import argparse
import json
from pathlib import Path

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PROMPTS_DIR = Path("cs336_alignment") / "prompts"
MODEL_ID = "allenai/OLMo-2-0425-1B"
RESULTS_PATH = Path("results/prompting_baselines_results.json")
MAX_SAVED_EXAMPLES = 10  # examples per category saved to JSON


# ------------------------------------------------------------------
# Per-prompt config
# ------------------------------------------------------------------
PROMPT_CONFIGS = [
    {
        "name": "question_only",
        "template_file": "question_only",
        "reward_fn": question_only_reward_fn,
        "use_stop": False,
    },
    {
        "name": "r1_zero",
        "template_file": "r1_zero",
        "reward_fn": r1_zero_reward_fn,
        "use_stop": True,
    },
    {
        "name": "r1_zero_three_shot",
        "template_file": "r1_zero_three_shot_gsm8k",
        "reward_fn": r1_zero_reward_fn,
        "use_stop": True,
    },
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_prompt_template(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.prompt").read_text()


def format_prompt(template: str, question: str) -> str:
    return template.replace("{question}", question)


def extract_gt_answer(raw_answer: str) -> str:
    """GSM8K answers look like: '...rationale... #### 72'"""
    return raw_answer.split("####")[-1].strip()


def make_sampling_params(use_stop: bool) -> dict:
    params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 42,
    }
    if use_stop:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


def categorize(fmt_reward: float, ans_reward: float) -> int:
    """Return category 1/2/3 as defined in the problem."""
    if fmt_reward == 1.0 and ans_reward == 1.0:
        return 1
    if fmt_reward == 1.0 and ans_reward == 0.0:
        return 2
    return 3  # fmt_reward == 0



# ------------------------------------------------------------------
# Core evaluation logic
# ------------------------------------------------------------------
def evaluate_prompt(
    server: VLLMServer,
    cfg: dict,
    questions: list[str],
    gt_answers: list[str],
) -> tuple[dict, dict]:
    """Run one prompt config. Returns (counts, cat_examples)."""
    template = load_prompt_template(cfg["template_file"])
    prompts = [format_prompt(template, q) for q in questions]
    sampling_params = make_sampling_params(cfg["use_stop"])

    print(f"\n{'='*60}")
    print(f"Prompt: {cfg['name']}")
    print(f"{'='*60}")
    print(f"Generating {len(prompts)} completions...")

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=256,
    )

    counts = {1: 0, 2: 0, 3: 0}
    cat_examples: dict[int, list[dict]] = {1: [], 2: [], 3: []}

    for q, gt, completion in zip(questions, gt_answers, completions):
        response = completion.text
        rewards = cfg["reward_fn"](response, gt)
        cat = categorize(rewards["format_reward"], rewards["answer_reward"])
        counts[cat] += 1
        if len(cat_examples[cat]) < MAX_SAVED_EXAMPLES:
            cat_examples[cat].append({
                "question": q,
                "gt_answer": gt,
                "response": response,
                "format_reward": rewards["format_reward"],
                "answer_reward": rewards["answer_reward"],
            })

    total = sum(counts.values())

    return counts, cat_examples


# ------------------------------------------------------------------
# Modal
# ------------------------------------------------------------------
from cs336_alignment.modal_utils import (
    GPU,
    RUN_TIMEOUT_SECONDS,
    app,
    image,
    wandb_secret,
    VOLUME_MOUNTS,
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=60*60,
    secrets=[wandb_secret],
    max_containers=4,
    volumes=VOLUME_MOUNTS,
)
def run_single_prompt_on_modal(prompt_name: str, max_samples: int | None = None) -> str:
    cfg = next(c for c in PROMPT_CONFIGS if c["name"] == prompt_name)

    print(f"Loading GSM8K test set for prompt '{prompt_name}'...")
    dataset = []
    with open("data/gsm8k/train.jsonl") as f:
        for line in f:
            dataset.append(json.loads(line))
    if max_samples is not None:
        dataset = dataset[:max_samples]

    questions = [ex["question"] for ex in dataset]
    gt_answers = [extract_gt_answer(ex["answer"]) for ex in dataset]
    print(f"Loaded {len(questions)} questions.")

    server = VLLMServer(model_id=MODEL_ID, port=8000, launch_server=True, gpu=0)
    server.start()
    try:
        counts, examples = evaluate_prompt(server, cfg, questions, gt_answers)
    finally:
        server.stop()

    result = {
        "counts": {str(k): v for k, v in counts.items()},
        "examples": {str(k): v for k, v in examples.items()},
    }

    # Save on the container as a safety net
    out_path = RESULTS_PATH.parent / f"prompting_baselines_{prompt_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved on container to {out_path}")

    return json.dumps(result, indent=2)


@app.local_entrypoint()
def modal_main(max_samples: int = 0) -> None:
    n = max_samples if max_samples > 0 else None
    prompt_names = [cfg["name"] for cfg in PROMPT_CONFIGS]
    print(f"Launching {len(prompt_names)} Modal containers in parallel: {prompt_names}")

    # starmap fans out one container per (prompt_name, n) pair
    all_results = {}
    for prompt_name, result_json in zip(
        prompt_names,
        run_single_prompt_on_modal.starmap([(name, n) for name in prompt_names]),
    ):
        all_results[prompt_name] = json.loads(result_json)
        print(f"  Received results for '{prompt_name}'")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, data in all_results.items():
        c = {int(k): v for k, v in data["counts"].items()}
        total = sum(c.values())
        print(f"{name:<25}  cat1={c[1]} ({100*c[1]/total:.0f}%)  "
              f"cat2={c[2]} ({100*c[2]/total:.0f}%)  "
              f"cat3={c[3]} ({100*c[3]/total:.0f}%)")

    # Merge and save locally
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results merged and saved locally to {RESULTS_PATH}")

