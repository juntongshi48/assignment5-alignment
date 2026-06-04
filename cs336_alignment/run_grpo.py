"""GRPO training script for OLMo-2-0425-1B on GSM8K."""
import argparse
import json
import random
from pathlib import Path

from cs336_alignment import grpo
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

# ------------------------------------------------------------------
# Fixed hyperparameters
# ------------------------------------------------------------------
MODEL_ID = "allenai/OLMo-2-0425-1B"
TRAIN_DATA_PATH = "data/gsm8k/train.jsonl"
VAL_DATA_PATH = "data/gsm8k/test.jsonl"
PROMPTS_DIR = Path("cs336_alignment") / "prompts"

N_TRAIN_EXAMPLES = 6400
N_VAL_EXAMPLES = 1024
NUM_ROLLOUT_STEPS = 200
ROLLOUT_BATCH_SIZE = 256   # counts responses: 32 prompts * 8 rollouts
TRAIN_BATCH_SIZE = 256
GROUP_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 32
SAMPLING_TEMPERATURE = 1.0
SAMPLING_MAX_TOKENS = 512
MAX_GRAD_NORM = 1.0

EVAL_EVERY_N_STEPS = 10
LOG_ROLLOUTS_EVERY_N_STEPS = 40

VLLM_PORT = 8000
VLLM_GPU = 1         # physical GPU index for the vLLM subprocess
POLICY_DEVICE = "cuda:0"  # GPU for policy training

# ------------------------------------------------------------------
# Prompt configs
# ------------------------------------------------------------------
PROMPT_CONFIGS = {
    "question_only": {
        "template_file": "question_only",
        "reward_fn": question_only_reward_fn,
        "use_stop": False,
    },
    "r1_zero": {
        "template_file": "r1_zero",
        "reward_fn": r1_zero_reward_fn,
        "use_stop": True,
    },
    "r1_zero_three_shot": {
        "template_file": "r1_zero_three_shot_gsm8k",
        "reward_fn": r1_zero_reward_fn,
        "use_stop": True,
    },
}

NORMALIZATION_CONSTANT = ROLLOUT_BATCH_SIZE * SAMPLING_MAX_TOKENS
GRPO_CONFIGS = {
    "GRPO": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
    },
    "GRPO_constant": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "Dr_GRPO": {
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "RFT": {
        "baseline": "none",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "MaxRL": {
        "baseline": "mean",
        "advantage_normalizer": "mean",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_prompt_template(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.prompt").read_text()


def format_prompt(template: str, question: str) -> str:
    return template.replace("{question}", question)


def extract_gt_answer(raw_answer: str) -> str:
    return raw_answer.split("####")[-1].strip()


def load_dataset(path: str, max_examples: int | None = None) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    if max_examples is not None:
        data = data[:max_examples]
    return data


def make_sampling_params(use_stop: bool, n: int, seed: int) -> dict:
    params = {
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": 1.0,
        "max_tokens": SAMPLING_MAX_TOKENS,
        "n": n,
        "seed": seed,
    }
    if use_stop:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def run_grpo(
    learning_rate: float,
    prompt_style: str,
    seed: int,
    grpo_cfg_name: str = "GRPO",
    exp_name: str = "grpo",
) -> None:
    import torch
    import wandb
    from cs336_alignment.checkpoint import get_model_and_tokenizer
    from cs336_alignment.grpo import compute_rollout_rewards, run_grpo_train_step
    from cs336_alignment.vllm_utils import VLLMServer

    random.seed(seed)
    torch.manual_seed(seed)

    cfg = PROMPT_CONFIGS[prompt_style]
    grpo_cfg = GRPO_CONFIGS[grpo_cfg_name]
    reward_fn = cfg["reward_fn"]
    template = load_prompt_template(cfg["template_file"])
    use_stop = cfg["use_stop"]

    train_data = load_dataset(TRAIN_DATA_PATH, N_TRAIN_EXAMPLES)
    val_data = load_dataset(VAL_DATA_PATH, N_VAL_EXAMPLES)

    wandb.init(
        project=f"cs336-alignment-{exp_name}",
        name=f"{grpo_cfg_name}_{prompt_style}_lr{learning_rate}_seed{seed}",
        config={
            "model_id": MODEL_ID,
            "learning_rate": learning_rate,
            "prompt_style": prompt_style,
            "seed": seed,
            "n_train_examples": N_TRAIN_EXAMPLES,
            "n_val_examples": N_VAL_EXAMPLES,
            "num_rollout_steps": NUM_ROLLOUT_STEPS,
            "rollout_batch_size": ROLLOUT_BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "sampling_max_tokens": SAMPLING_MAX_TOKENS,
            "max_grad_norm": MAX_GRAD_NORM,
            **grpo_cfg
        },
    )

    # vLLM on GPU 1; policy on GPU 0
    server = VLLMServer(model_id=MODEL_ID, port=VLLM_PORT, launch_server=True, gpu=VLLM_GPU)
    server.start()

    policy, tokenizer = get_model_and_tokenizer(MODEL_ID, device=POLICY_DEVICE)

    server.init_weight_sync(policy_device=POLICY_DEVICE)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    n_prompts_per_rollout_batch = ROLLOUT_BATCH_SIZE // GROUP_SIZE

    try:
        for step in range(NUM_ROLLOUT_STEPS):
            batch = random.sample(train_data, n_prompts_per_rollout_batch)
            questions = [ex["question"] for ex in batch]
            gt_answers = [extract_gt_answer(ex["answer"]) for ex in batch]
            prompts = [format_prompt(template, q) for q in questions]

            # Sync weights, then generate group_size rollouts per prompt
            server.sync_policy_weights(policy)
            completions = server.generate_completions(
                prompts=prompts,
                sampling_params=make_sampling_params(use_stop, n=GROUP_SIZE, seed=step),
                batch_size=n_prompts_per_rollout_batch,
            )
            # vLLM returns (prompt[0]*GROUP_SIZE), (prompt[1]*GROUP_SIZE), ...
            rollout_responses = [c.text for c in completions]
            repeated_prompts = [p for p in prompts for _ in range(GROUP_SIZE)]
            repeated_ground_truths = [gt for gt in gt_answers for _ in range(GROUP_SIZE)]

            policy.train()
            loss, meta = run_grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                max_grad_norm=MAX_GRAD_NORM,
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=GROUP_SIZE,
                **grpo_cfg
            )

            log = {
                "train/loss": loss.item(),
                "train/mean_reward": meta["total_rewards"],
                "train/mean_format_reward": meta["format_rewards"],
                "train/token_entropy": meta["token_entropy"],
            }
            if meta["grad_norm"] is not None:
                log["train/grad_norm"] = meta["grad_norm"]
            wandb.log(log, step=step)
            grad_norm_str = f"{meta['grad_norm']:.4f}" if meta["grad_norm"] is not None else "None"
            print(
                f"Step {step}/{NUM_ROLLOUT_STEPS}: loss={loss.item():.4f}, "
                f"reward={meta['total_rewards']:.4f}, "
                f"grad_norm={grad_norm_str}",
                flush=True,
            )

            if (step + 1) % LOG_ROLLOUTS_EVERY_N_STEPS == 0:
                table = wandb.Table(columns=["question", "gt_answer", "response"])
                for i in range(min(8, n_prompts_per_rollout_batch)):
                    table.add_data(questions[i], gt_answers[i], rollout_responses[i * GROUP_SIZE])
                wandb.log({"train/rollouts": table}, step=step)

            if (step + 1) % EVAL_EVERY_N_STEPS == 0:
                policy.eval()
                val_questions = [ex["question"] for ex in val_data]
                val_gt_answers = [extract_gt_answer(ex["answer"]) for ex in val_data]
                val_prompts = [format_prompt(template, q) for q in val_questions]

                server.sync_policy_weights(policy)
                val_completions = server.generate_completions(
                    prompts=val_prompts,
                    sampling_params=make_sampling_params(use_stop, n=1, seed=step),
                    batch_size=256,
                )
                val_responses = [c.text for c in val_completions]
                _, val_reward_meta = compute_rollout_rewards(reward_fn, val_responses, val_gt_answers)
                mean_val_reward = val_reward_meta["mean_total_rewards"]
                mean_val_fmt = val_reward_meta["mean_format_rewards"]
                mean_val_resp_len = sum(len(c.token_ids) for c in val_completions) / len(val_completions)
                wandb.log(
                    {
                        "val/mean_reward": mean_val_reward,
                        "val/mean_format_reward": mean_val_fmt,
                        "val/mean_response_length": mean_val_resp_len,
                    },
                    step=step,
                )
                print(
                    f"  Val reward: {mean_val_reward:.4f}, format: {mean_val_fmt:.4f}, "
                    f"resp_len: {mean_val_resp_len:.1f}",
                    flush=True,
                )
                policy.train()
    finally:
        server.stop()
        wandb.finish()



# ------------------------------------------------------------------
# Modal
# ------------------------------------------------------------------
from cs336_alignment.modal_utils import app, image, wandb_secret, VOLUME_MOUNTS


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 60 * 4,
    secrets=[wandb_secret],
    max_containers=4,
    volumes=VOLUME_MOUNTS,
)
def run_grpo_on_modal(
    learning_rate: float = 1e-5,
    prompt_style: str = "r1_zero",
    seed: int = 0,
    grpo_cfg_name: str = "GRPO",
    exp_name: str = "grpo",
) -> None:
    run_grpo(learning_rate=learning_rate, prompt_style=prompt_style, seed=seed, grpo_cfg_name=grpo_cfg_name, exp_name=exp_name)


@app.local_entrypoint()
def modal_main(
    learning_rate: float = 1e-5,
    prompt_style: str = "r1_zero",
    seed: int = 3,
) -> None:
    print(f"Launching GRPO: lr={learning_rate}, prompt={prompt_style}, seed={seed}")
    run_grpo_on_modal.remote(learning_rate=learning_rate, prompt_style=prompt_style, seed=seed)


@app.local_entrypoint()
def modal_sweep_seeds(
    learning_rate: float = 1e-5,
    prompt_style: str = "r1_zero",
) -> None:
    seeds = [0, 1, 2, 3]
    print(f"Launching seed sweep: lr={learning_rate}, prompt={prompt_style}, seeds={seeds}")
    futures = [
        run_grpo_on_modal.spawn(learning_rate=learning_rate, prompt_style=prompt_style, seed=s)
        for s in seeds
    ]
    for s, f in zip(seeds, futures):
        f.get()
        print(f"Seed {s} finished.")


@app.local_entrypoint()
def modal_sweep_lr(
    prompt_style: str = "r1_zero",
) -> None:
    learning_rate = [1e-6, 3e-6, 3e-5, 1e-4, 3e-4]
    print(f"Launching lr sweep: lr={learning_rate}, prompt={prompt_style}")
    futures = [
        run_grpo_on_modal.spawn(learning_rate=lr, prompt_style=prompt_style, seed=0)
        for lr in learning_rate
    ]
    for lr, f in zip(learning_rate, futures):
        f.get()
        print(f"lr {lr} finished.")

@app.local_entrypoint()
def modal_sweep_prompt(
    learning_rate: float = 1e-5,
) -> None:
    seeds = [0, 1]
    prompt_styles = ["question_only", "r1_zero_three_shot"]
    print(f"Launching prompt sweep: lr={learning_rate}, prompt={prompt_styles}, seeds={seeds}")
    futures = [
        run_grpo_on_modal.spawn(learning_rate=learning_rate, prompt_style=prompt_style, seed=s)
        for prompt_style in prompt_styles for s in seeds 
    ]
        
        
@app.local_entrypoint()
def modal_sweep_grpo_variants(
    learning_rate: float = 1e-5,
    prompt_style: str = "r1_zero",
) -> None:
    exp_name = "grpo_variant_sweep"
    seeds = [0, 1, 2, 3]
    grpo_cfg_names = list(GRPO_CONFIGS.keys())
    # seeds = [0]
    # grpo_cfg_names = ["GRPO"]
    print(f"Launching grpo variant sweep: lr={learning_rate}, prompt={prompt_style}, grpo_variants={grpo_cfg_names}, seeds={seeds}")
    futures = [
        run_grpo_on_modal.spawn(learning_rate=learning_rate, prompt_style=prompt_style, seed=s, grpo_cfg_name=grpo_cfg_name, exp_name=exp_name)
        for grpo_cfg_name in grpo_cfg_names for s in seeds
    ]