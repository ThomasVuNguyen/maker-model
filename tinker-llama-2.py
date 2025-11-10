#!/usr/bin/env python
"""Higher-quality Tinker LoRA trainer inspired by train_cadmonkey.py."""

import os
import random
from math import ceil
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

import tinker
from tinker import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAMES = [
    "ThomasTheMaker/Synthetic-Object-v0",
    "ThomasTheMaker/Synthetic-Object",
]

BASE_MODEL = "meta-llama/Llama-3.2-1B"
MODEL_NAME = "cadmonkey-model"

NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
CHECKPOINT_INTERVAL_STEPS = 100
SEED = 3407

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))

EVAL_PROMPTS = [
    "hey cadmonkey, create me a sphere",
    "hey cadmonkey, create me a cat",
    "hey cadmonkey, create me a cylinder",
    "hey cadmonkey, create me a cube",
    "hey cadmonkey, create me a house",
    "hey cadmonkey, create me a car",
    "hey cadmonkey, create me a tree",
    "hey cadmonkey, create me a pyramid",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_env_file(path: str = ".env") -> None:
    """Minimal .env loader."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def chunk_list(items, chunk_size):
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def format_example(name: str, code: str) -> dict:
    return {
        "input": f"hey cadmonkey, create me a {name}",
        "output": code,
    }


def process_example(example: dict, tokenizer) -> types.Datum:
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_system_prompt=False,
    )
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
    user_text = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=True,
    )
    user_tokens = tokenizer.encode(user_text, add_special_tokens=True)

    prompt_weights = [0] * len(user_tokens)
    completion_weights = [1] * (len(full_tokens) - len(user_tokens))
    weights = prompt_weights + completion_weights

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def compute_weighted_loss(loss_outputs, batch):
    logprobs = np.concatenate([item["logprobs"].tolist() for item in loss_outputs])
    weights = np.concatenate(
        [example.loss_fn_inputs["weights"].tolist() for example in batch]
    )
    return -np.dot(logprobs, weights) / weights.sum()


def run_evaluation(sampling_client, tokenizer):
    print("\n" + "=" * 60)
    print("Evaluation Prompts")
    print("=" * 60)

    for idx, prompt_text in enumerate(EVAL_PROMPTS, 1):
        messages = [{"role": "user", "content": prompt_text}]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = types.ModelInput.from_ints(tokenizer.encode(chat_prompt))
        params = types.SamplingParams(
            max_tokens=1024,
            temperature=0.7,
            repetition_penalty=1.1,
            stop=["<|eot_id|>", "<|end|>", "</s>"],
        )
        future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        result = future.result()
        output = tokenizer.decode(result.sequences[0].tokens)
        print(f"\n[{idx}/{len(EVAL_PROMPTS)}] Prompt: {prompt_text}")
        print("-" * 60)
        print(output.strip())
        print("-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    load_env_file()
    set_seed(SEED)

    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("Missing TINKER_API_KEY. Set it in .env or the shell.")

    print("=" * 60)
    print("Tinker CadMonkey Trainer (multi-epoch)")
    print("=" * 60)

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    tokenizer = training_client.get_tokenizer()

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    loaded = []
    for name in DATASET_NAMES:
        try:
            print(f"Loading dataset: {name}")
            ds = load_dataset(name, split="train")
            loaded.append(ds)
            print(f"  ✓ {len(ds)} examples")
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")

    if not loaded:
        raise RuntimeError("No datasets could be loaded. Check DATASET_NAMES.")

    dataset = loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)
    print(f"\nTotal training examples: {len(dataset)}")

    # ------------------------------------------------------------------
    # Preprocess examples
    # ------------------------------------------------------------------
    print("\nFormatting examples...")
    examples = [format_example(row["name"], row["code"]) for row in dataset]

    # Quick stats
    token_counts = []
    for ex in examples:
        text = ex["input"] + "\n" + ex["output"]
        token_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
    over_limit = sum(1 for count in token_counts if count > MAX_CONTEXT_TOKENS)
    print(f"Average tokens/example: {np.mean(token_counts):.2f}")
    print(f"Examples over {MAX_CONTEXT_TOKENS} tokens: {over_limit}")

    print("\nTokenizing into Tinker datums...")
    processed_examples = [process_example(ex, tokenizer) for ex in tqdm(examples)]
    total_steps_per_epoch = ceil(len(processed_examples) / BATCH_SIZE)
    print(f"Steps per epoch (@ batch {BATCH_SIZE}): {total_steps_per_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print("\n" + "-" * 60)
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print("-" * 60)

        random.shuffle(processed_examples)
        epoch_bar = tqdm(
            list(chunk_list(processed_examples, BATCH_SIZE)),
            desc=f"Epoch {epoch}",
            unit="batch",
        )

        for batch in epoch_bar:
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=LEARNING_RATE)
            )

            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            loss = compute_weighted_loss(fwdbwd_result.loss_fn_outputs, batch)
            losses.append(loss)
            global_step += 1

            epoch_bar.set_postfix(loss=f"{loss:.4f}")

            if global_step % CHECKPOINT_INTERVAL_STEPS == 0:
                ckpt_name = f"{MODEL_NAME}-step-{global_step}"
                print(f"\n💾 Saving checkpoint: {ckpt_name}")
                training_client.save_weights_and_get_sampling_client(name=ckpt_name)

        epoch_loss = np.mean(losses[-total_steps_per_epoch:])
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    final_name = f"{MODEL_NAME}-final"
    print(f"💾 Saving final model: {final_name}")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=final_name
    )
    print("✓ Final weights saved")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    run_evaluation(sampling_client, tokenizer)


if __name__ == "__main__":
    main()
