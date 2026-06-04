from typing import Callable, Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase
import torch.nn.functional as F

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)

    full_ids_list = []
    response_mask_list = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        
        response_ids = tokenizer(
            output,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = prompt_ids + response_ids
        full_response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        full_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        response_mask_list.append(torch.tensor(full_response_mask, dtype=torch.bool))

    pad_token_id = tokenizer.pad_token_id
    full_input_ids = torch.nn.utils.rnn.pad_sequence(
        full_ids_list,
        batch_first=True,
        padding_value=pad_token_id,
    )
    full_response_mask = torch.nn.utils.rnn.pad_sequence(
        response_mask_list,
        batch_first=True,
        padding_value=False,
    )

    input_ids = full_input_ids[:, :-1]
    labels = full_input_ids[:, 1:]

    response_mask = full_response_mask[:, 1:]
    response_mask = response_mask & (labels != pad_token_id)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
    
    
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    seq_length = labels.shape[1]
    log_probs = log_probs[:, :seq_length, :]
    response_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    token_entropy = None
    if return_token_entropy:
        token_entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1)
    return {
        "log_probs": response_log_probs,
        "token_entropy": token_entropy,
    }
        

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards_list = []
    for rollout_response, repeated_ground_truth in zip(rollout_responses, repeated_ground_truths):
        rewards = reward_fn(rollout_response, repeated_ground_truth)
        rewards_list.append(rewards)
    
    reward_keys = rewards_list[0].keys()
    reward_tensors = {key: torch.tensor([rewards[key] for rewards in rewards_list], dtype=torch.float) for key in reward_keys}
    raw_rewards = reward_tensors["reward"]
    mean_total_rewards = raw_rewards.mean().item()
    mean_format_rewards = reward_tensors["format_reward"].mean().item()
    meta_data = {
        "mean_total_rewards": mean_total_rewards,
        "mean_format_rewards": mean_format_rewards,
    }
    return raw_rewards, meta_data


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    positive_group_bias: float = None,
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = raw_rewards.reshape(-1, group_size)
    
    if baseline == "mean":
        baseline_values = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - baseline_values
    elif baseline == "none":
        advantages = rewards
    
    if positive_group_bias is not None:
        tilt = 1.0 + positive_group_bias * torch.sign(advantages)
        advantages = advantages * tilt
    
    if advantage_normalizer == "std":
        normalizer_values = advantages.std(dim=1, keepdim=True) + advantage_eps
        normalized_advantages = advantages / normalizer_values
    elif advantage_normalizer == "mean":
        normalizer_values = advantages.abs().mean(dim=1, keepdim=True) + advantage_eps
        normalized_advantages = advantages / normalizer_values
    elif advantage_normalizer == "none":
        normalized_advantages = advantages
    
    meta_data = {
        "reward_mean": rewards.mean(),
        "reward_std": rewards.std(),
        "reward_min": rewards.min().values,
        "reward_max": rewards.max().values,
    }
    
    return normalized_advantages.reshape(-1), meta_data


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    masked_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        loss = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # avoid division by zero, which might happen when the response is empty
        return loss.mean()
    elif loss_normalization == "constant":
        loss = masked_loss.sum() / normalization_constant
        return loss


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    metadata = {}
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1)
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -policy_log_probs * raw_rewards_or_advantages
        return per_token_policy_gradient_loss, metadata

    log_ratios = policy_log_probs - old_log_probs
    ratios = torch.exp(log_ratios)
    if importance_reweighting_method == "noclip":
        per_token_policy_gradient_loss = -ratios * raw_rewards_or_advantages
        return per_token_policy_gradient_loss, metadata
    if importance_reweighting_method == "grpo":
        clipped_ratios = torch.clamp(
            ratios,
            1.0 - cliprange,
            1.0 + cliprange,
        )
        per_token_policy_gradient_loss = -torch.min(
            ratios * raw_rewards_or_advantages,
            clipped_ratios * raw_rewards_or_advantages,
        )
        is_clipped = (ratios > 1.0 + cliprange) | (ratios < 1.0 - cliprange)
        if response_mask is not None:
            mask = response_mask.bool()
            clip_numerator = (is_clipped & mask).sum()
            clip_denominator = mask.sum()
        else:
            clip_numerator = is_clipped.sum()
            clip_denominator = torch.tensor(is_clipped.numel(), device=is_clipped.device)
        metadata["clip_numerator"] = clip_numerator.detach()
        metadata["clip_denominator"] = clip_denominator.detach()
        metadata["clip_fraction"] = (clip_numerator / clip_denominator.clamp(min=1)).item()
        return per_token_policy_gradient_loss, metadata
    if importance_reweighting_method == "gspo":
        token_count = response_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        sequence_log_ratio = (log_ratios * response_mask).sum(dim=1, keepdim=True) / token_count
        sequence_ratio = torch.exp(sequence_log_ratio)
        clipped_sequence_ratio = torch.clamp(
            sequence_ratio,
            1.0 - cliprange,
            1.0 + cliprange,
        )
        selected_obj = torch.min(
            sequence_ratio * raw_rewards_or_advantages,
            clipped_sequence_ratio * raw_rewards_or_advantages,
        )
        per_token_policy_gradient_loss = -selected_obj.expand_as(policy_log_probs)  # give all tokens the same sequence-level loss
        is_clipped = (sequence_ratio > 1.0 + cliprange) | (sequence_ratio < 1.0 - cliprange)
        clip_numerator = is_clipped.sum()
        clip_denominator = torch.tensor(is_clipped.numel(), device=is_clipped.device)
        metadata["clip_numerator"] = clip_numerator.detach()
        metadata["clip_denominator"] = clip_denominator.detach()
        metadata["clip_fraction"] = (clip_numerator / clip_denominator.clamp(min=1)).item()
        return per_token_policy_gradient_loss, metadata
 
    
def compute_old_log_probs(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    repeated_prompts: list[str],
    rollout_responses: list[str],
    microbatch_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    tokenizer_out = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenizer_out["input_ids"]
    labels = tokenizer_out["labels"]
    batch_size = input_ids.shape[0]

    was_training = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, batch_size, microbatch_size):
            log_probs = get_response_log_probs(
                model,
                input_ids[i:i + microbatch_size].to(device),
                labels[i:i + microbatch_size].to(device),
            )["log_probs"]
            chunks.append(log_probs.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    positive_group_bias: float = None,
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    device = next(model.parameters()).device
    batch_size = len(rollout_responses)
    microbatch_size = batch_size // gradient_accumulation_steps
    avg_loss = torch.tensor(0.0, device=device)
    avg_total_rewards = 0.0
    avg_format_rewards = 0.0
    total_entropy = 0.0
    total_active_tokens = 0
    total_clip_numerator = 0.0
    total_clip_denominator = 0.0

    keep = []
    pruned_repeated_prompts = []
    pruned_rollout_responses = []
    pruned_normalized_advantages = []
    for i in range(0, batch_size, microbatch_size):
        rollout_responses_mb = rollout_responses[i:i+microbatch_size]
        repeated_ground_truths_mb = repeated_ground_truths[i:i+microbatch_size]
        mb_size = len(rollout_responses_mb)

        rewards_mb, reward_meta_data = compute_rollout_rewards(
            reward_fn,
            rollout_responses_mb,
            repeated_ground_truths_mb,
        )
        normalized_advantages_mb, advantage_meta_data = compute_group_normalized_rewards(
            rewards_mb,
            group_size,
            baseline=baseline,
            positive_group_bias=positive_group_bias,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
        )
        
        keep_mb = normalized_advantages_mb != 0
        keep.extend(keep_mb.cpu().numpy().tolist())
        if keep_mb.sum() > 0:
            pruned_repeated_prompts.extend([repeated_prompts[i + j] for j in range(mb_size) if keep_mb[j].item()])
            pruned_rollout_responses.extend([rollout_responses_mb[j] for j in range(mb_size) if keep_mb[j].item()])
            pruned_normalized_advantages.append(normalized_advantages_mb[keep_mb])
        avg_total_rewards += reward_meta_data["mean_total_rewards"] * mb_size / batch_size
        avg_format_rewards += reward_meta_data["mean_format_rewards"] * mb_size / batch_size
    # Handle the edge case where all advantagesa are zero
    if len(pruned_normalized_advantages) == 0:
        print("[EDGE CASE]: All advantages are zero, skipping gradient update for this batch.")
        meta_data = {
            "total_rewards": avg_total_rewards,
            "format_rewards": avg_format_rewards,
            "token_entropy": 0.0,
            "grad_norm": None,
            "clip_fraction": None,
        }
        optimizer.zero_grad()
        return torch.tensor(0.0, device=device), meta_data
    pruned_normalized_advantages = torch.cat(pruned_normalized_advantages, dim=0)
    pruned_batch_size = pruned_normalized_advantages.shape[0]
    old_log_probs = old_log_probs[keep] if old_log_probs is not None else None
    # Accumulate gradients across the pruned batch
    for i in range(0, pruned_batch_size, microbatch_size):
        pruned_repeated_prompts_mb = pruned_repeated_prompts[i:i+microbatch_size]
        rollout_responses_mb = pruned_rollout_responses[i:i+microbatch_size]
        normalized_advantages_mb = pruned_normalized_advantages[i:i+microbatch_size]
        tokenizer_out = tokenize_prompt_and_output(
            pruned_repeated_prompts_mb,
            rollout_responses_mb,
            tokenizer,
        )
        inputs_ids_mb = tokenizer_out["input_ids"].to(device)
        labels_mb = tokenizer_out["labels"].to(device)
        response_mask_mb = tokenizer_out["response_mask"].to(device)
        normalized_advantages_mb = normalized_advantages_mb.to(device)
        
        mb_size = inputs_ids_mb.shape[0]

        log_prob_and_token_entropy = get_response_log_probs(
            model,
            inputs_ids_mb,
            labels_mb,
            return_token_entropy=True,
        )
        log_probs_mb = log_prob_and_token_entropy["log_probs"]
        token_entropy_mb = log_prob_and_token_entropy["token_entropy"]
        
        per_token_policy_gradient_loss_mb, loss_metadata = compute_policy_gradient_loss(
            normalized_advantages_mb,
            log_probs_mb,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs[i:i+mb_size, :log_probs_mb.shape[1]].to(device) if old_log_probs is not None else None,
            cliprange=cliprange,
            response_mask=response_mask_mb if importance_reweighting_method in ["grpo", "gspo"] else None,
        )
        loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss_mb,
            response_mask_mb,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )
        # "sequence" returns a per-microbatch mean, so reweight into a batch-level
        # mean. "constant" already divides by a fixed global constant, so the
        # microbatch contributions just sum -- no reweighting.
        if loss_normalization == "sequence":
            loss = loss * mb_size / batch_size
        # Backward pass.
        loss.backward()
        avg_loss += loss.detach()
        
        total_entropy += (token_entropy_mb * response_mask_mb).sum().detach().item()
        total_active_tokens += response_mask_mb.sum().item()
        if "clip_numerator" in loss_metadata:
            total_clip_numerator += loss_metadata["clip_numerator"].item()
            total_clip_denominator += loss_metadata["clip_denominator"].item()
    # Update weights once across entire batch.
    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm = grad_norm.item()
    if grad_norm is not None and torch.isnan(torch.tensor(grad_norm)):  # skip optimizer step if grad norm is NaN
        optimizer.zero_grad()
    else:
        optimizer.step()
        # Zero gradients once across entire batch.
        optimizer.zero_grad()
    
    meta_data = {
        "total_rewards": avg_total_rewards,
        "format_rewards": avg_format_rewards,
        "token_entropy": total_entropy / total_active_tokens if total_active_tokens > 0 else 0.0,
        "grad_norm": grad_norm,
        "clip_fraction": (total_clip_numerator / total_clip_denominator) if total_clip_denominator > 0 else None,
    }
    return avg_loss, meta_data
        

# def run_grpo_train_step_unoptimized(
#     model: torch.nn.Module,
#     tokenizer: PreTrainedTokenizerBase,
#     optimizer: torch.optim.Optimizer,
#     gradient_accumulation_steps: int,
#     max_grad_norm: float | None,
#     reward_fn: Callable[[str, str], dict[str, float]],
#     repeated_prompts: list[str],
#     rollout_responses: list[str],
#     repeated_ground_truths: list[str],
#     group_size: int,
#     baseline: Literal["mean", "none"] = "mean",
#     advantage_eps: float = 1e-6,
#     advantage_normalizer: Literal["std", "none", "mean"] = "std",
#     loss_normalization: Literal["sequence", "constant"] = "sequence",
#     importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
#     old_log_probs: torch.Tensor | None = None,
#     cliprange: float | None = None,
#     normalization_constant: int | None = None,
# ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
#     device = next(model.parameters()).device
#     batch_size = len(rollout_responses)
#     microbatch_size = batch_size // gradient_accumulation_steps
#     avg_loss = torch.tensor(0.0, device=device)
#     avg_total_rewards = 0.0
#     avg_format_rewards = 0.0
#     total_entropy = 0.0
#     total_active_tokens = 0
#     for i in range(0, batch_size, microbatch_size):
#         rollout_responses_mb = rollout_responses[i:i+microbatch_size]
#         repeated_ground_truths_mb = repeated_ground_truths[i:i+microbatch_size]
#         mb_size = len(rollout_responses_mb)

#         rewards_mb, reward_meta_data = compute_rollout_rewards(
#             reward_fn,
#             rollout_responses_mb,
#             repeated_ground_truths_mb,
#         )
#         normalized_advantages_mb, advantage_meta_data = compute_group_normalized_rewards(
#             rewards_mb,
#             group_size,
#             baseline=baseline,
#             advantage_eps=advantage_eps,
#             advantage_normalizer=advantage_normalizer,
#         )
#         tokenizer_out = tokenize_prompt_and_output(
#             repeated_prompts[i:i+microbatch_size],
#             rollout_responses_mb,
#             tokenizer,
#         )
#         inputs_ids_mb = tokenizer_out["input_ids"].to(device)
#         labels_mb = tokenizer_out["labels"].to(device)
#         response_mask_mb = tokenizer_out["response_mask"].to(device)
#         normalized_advantages_mb = normalized_advantages_mb.to(device)

#         log_prob_and_token_entropy = get_response_log_probs(
#             model,
#             inputs_ids_mb,
#             labels_mb,
#             return_token_entropy=True,
#         )
#         log_probs_mb = log_prob_and_token_entropy["log_probs"]
#         token_entropy_mb = log_prob_and_token_entropy["token_entropy"]
        
#         per_token_policy_gradient_loss_mb = -log_probs_mb * normalized_advantages_mb.unsqueeze(1)
#         loss = aggregate_loss_across_microbatch(
#             per_token_policy_gradient_loss_mb,
#             response_mask_mb,
#             loss_normalization=loss_normalization,
#             normalization_constant=normalization_constant,
#         )
#         # "sequence" returns a per-microbatch mean, so reweight into a batch-level mean. "constant" already divides by a fixed global constant, so the microbatch contributions just sum.
#         if loss_normalization == "sequence":
#             loss = loss * mb_size / batch_size
#         # Backward pass.
#         loss.backward()
#         avg_loss += loss.detach()
        
#         avg_total_rewards += reward_meta_data["mean_total_rewards"] * mb_size / batch_size
#         avg_format_rewards += reward_meta_data["mean_format_rewards"] * mb_size / batch_size
        
#         total_entropy += (token_entropy_mb * response_mask_mb).sum().detach().item()
#         total_active_tokens += response_mask_mb.sum().item()
#     # Update weights once across entire batch.
#     grad_norm = None
#     if max_grad_norm is not None:
#         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#         grad_norm = grad_norm.item()
#     if grad_norm is not None and torch.isnan(torch.tensor(grad_norm)):  # skip optimizer step if grad norm is NaN
#         optimizer.zero_grad()
#     else:
#         optimizer.step()
#         # Zero gradients once across entire batch.
#         optimizer.zero_grad()
    
#     meta_data = {
#         "total_rewards": avg_total_rewards,
#         "format_rewards": avg_format_rewards,
#         "token_entropy": total_entropy / total_active_tokens if total_active_tokens > 0 else 0.0,
#         "grad_norm": grad_norm,
#     }
#     return avg_loss, meta_data