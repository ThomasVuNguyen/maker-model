#!/usr/bin/env python
"""Standalone Python equivalent of the tinker-llama notebook."""

dataset_name = "ThomasTheMaker/Synthetic-Object-v0"
base_model = "meta-llama/Llama-3.2-1B"
# "Qwen/Qwen3-8B-Base"
# "openai/gpt-oss-20b"
# 
model_name = "cadmonkey-model"
unsloth_model = "unsloth/Llama-3.2-1B"
# "Qwen/Qwen3-8B-Base"
# "openai/gpt-oss-20b"
# 

import os
from pathlib import Path


def load_env_file(path: str = ".env") -> None:
    """Lightweight .env loader to avoid extra dependencies."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env_file()

tinker_api_key = os.getenv("TINKER_API_KEY")
if not tinker_api_key:
    raise RuntimeError("Missing TINKER_API_KEY. Set it in .env or your shell environment.")

max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))

import tinker
service_client = tinker.ServiceClient()
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)

training_client = service_client.create_lora_training_client(
    base_model=base_model
)


from datasets import load_dataset

# Load the dataset
dataset = load_dataset(dataset_name)

# Create examples from the dataset
examples = [
    {
        "input": f"hey cadmonkey, make me a {row['name']}",
        # row['prompt'],
        
        "output": row['code']
    }
    for row in dataset['train']  # Adjust split name if needed (e.g., 'test', 'validation')
]

# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()

# Count total tokens and find longest
total_tokens = 0
token_counts: list[int] = []
longest_example_tokens = 0
max_tokens_example = None

for ex in examples:
    text = ex["input"] + "\n" + ex["output"]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(tokens)
    total_tokens += token_count
    token_counts.append(token_count)
    
    # Track the longest example
    if token_count > longest_example_tokens:
        longest_example_tokens = token_count
        max_tokens_example = ex

print(f"Total examples: {len(examples)}")
print(f"Total tokens: {total_tokens:,}")
print(f"Average tokens per example: {total_tokens / len(examples):.2f}")
print(f"\nLongest example:")
print(f"  Tokens: {longest_example_tokens:,}")
print(f"  Input: {max_tokens_example['input']}")
output_token_length = len(tokenizer.encode(max_tokens_example["output"], add_special_tokens=False))
print(f"  Output characters: {len(max_tokens_example['output'])}")
print(f"  Output tokens: {output_token_length}")

examples_over_limit = sum(1 for count in token_counts if count > max_context_tokens)
print(f"\nExamples exceeding {max_context_tokens} tokens: {examples_over_limit}")

# --- Code Block 3: Process Examples ---
from tinker import types
# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()

def process_example(example: dict, tokenizer) -> types.Datum:
    messages = [
        {"role": "user", "content": example['input']},
        {"role": "assistant", "content": example['output']}
    ]
    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_system_prompt=False  # Add this if supported
    )
    # Tokenize full conversation
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
    # Tokenize user-only portion to find split point
    user_only = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=True
    )
    user_tokens = tokenizer.encode(user_only, add_special_tokens=True)
    # Assign weights
    prompt_weights = [0] * len(user_tokens)
    completion_weights = [1] * (len(full_tokens) - len(user_tokens))
    weights = prompt_weights + completion_weights
    # Shift tokens for next-token prediction
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    weights = weights[1:]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]

'''
# Visualize the first example for debugging
datum0 = processed_examples[0]
print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
print("-" * 50)
for i, (inp, tgt, wgt) in enumerate(
    zip(
        datum0.model_input.to_ints(),
        datum0.loss_fn_inputs['target_tokens'].tolist(),
        datum0.loss_fn_inputs['weights'].tolist()
    )
):
    print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")
    
'''
# --- Code Block 4: Training with checkpoints and progress bar ---
from tqdm import tqdm
import numpy as np

# Training configuration
num_steps = 30  # Adjust as needed
checkpoint_interval = 5  # Save checkpoint every 20 steps
model_name_prefix = model_name

print(f"\nTraining for {num_steps} steps")
print(f"Checkpoints will be saved every {checkpoint_interval} steps")
print("="*60)

for step in tqdm(range(num_steps), desc="Training progress", unit="step"):
    # Forward + backward pass
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
    
    # Wait for results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()
    
    # Compute weighted average log loss per token
    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    loss = -np.dot(logprobs, weights) / weights.sum()
    
    tqdm.write(f"Step {step + 1}/{num_steps} | Loss per token: {loss:.4f}")
    
    # Save checkpoint at intervals
    if (step + 1) % checkpoint_interval == 0:
        checkpoint_name = f"{model_name_prefix}-checkpoint-{step + 1}"
        tqdm.write(f"💾 Saving checkpoint: {checkpoint_name}")
        checkpoint_client = training_client.save_weights_and_get_sampling_client(name=checkpoint_name)
        tqdm.write(f"✓ Checkpoint saved: {checkpoint_name}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# --- Save Final Model ---
print(f"\n💾 Saving final model: {model_name_prefix}-final")
final_model_name = f"{model_name_prefix}-final"
sampling_client = training_client.save_weights_and_get_sampling_client(name=final_model_name)
print(f"✓ Final model saved: {final_model_name}")

# --- Code Block 5: Sampling and output ---
test_messages = [{"role": "user", "content": "hey cadmonkey, make me a whale"}]
prompt_text = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

params = types.SamplingParams(max_tokens=4096, temperature=0.8, stop=["<|eot_id|>", "<|end|>", "</s>"])
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=3)
result = future.result()

print("\n" + "="*80)
print("FULL MODEL OUTPUTS:")
print("="*80)
for i, seq in enumerate(result.sequences):
    generated_text = tokenizer.decode(seq.tokens)
    full_output = prompt_text + generated_text
    
    print(f"\n--- Sample {i} ---")
    print(f"Generated only: {repr(generated_text)}")
    print(f"Full output:\n{full_output}")
    print("-" * 80)

print("\n" + "="*60)
print("SAVED MODELS:")
print("="*60)
print(f"Final model: {final_model_name}")
print(f"Checkpoints saved at steps: {', '.join([str(i) for i in range(checkpoint_interval, num_steps + 1, checkpoint_interval)])}")
print("="*60)


TINKER_MODEL_NAME = model_name + "-final"  # The name you used when saving in Tinker
TINKER_PATH = "tinker://908f5273-02d4-4e23-a685-3d65e61bdcbb/sampler_weights/cadmonkey-model-final"  # Your Tinker path


# --- Download LoRA from Tinker, Merge, and Upload to HF Hub ---

import tinker
import urllib.request
import tarfile
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo

# Configuration
HF_USERNAME = "ThomasTheMaker"  # Replace with your Hugging Face username
MODEL_NAME_ON_HUB = model_name  # Name for merged model on HF Hub
ADAPTER_NAME_ON_HUB = model_name + "-lora"  # Name for LoRA adapters on HF Hub
BASE_MODEL = unsloth_model  # The base model you trained from (change this to match your base model)


print("="*60)
print("DOWNLOADING, MERGING, AND UPLOADING MODEL")
print("="*60)

# Step 1: Download LoRA adapters from Tinker
print(f"\n📥 Step 1: Downloading LoRA adapters from Tinker...")
rc = service_client.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path(TINKER_PATH)
checkpoint_archive_url_response = future.result()

print(f"   Downloading from: {checkpoint_archive_url_response.url}")
print(f"   URL expires: {checkpoint_archive_url_response.expires}")

urllib.request.urlretrieve(checkpoint_archive_url_response.url, "cadmonkey.tar")
print("   ✓ Downloaded cadmonkey.tar")

# Step 2: Extract the tar file
print(f"\n📦 Step 2: Extracting LoRA adapters...")
os.makedirs("./lora_adapters", exist_ok=True)
with tarfile.open("cadmonkey.tar", "r") as tar:
    tar.extractall("./lora_adapters")
print("   ✓ Extracted to ./lora_adapters")

# Step 3: Load base model and tokenizer
print(f"\n🔧 Step 3: Loading base model...")
print(f"   Base model: {BASE_MODEL}")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("   ✓ Base model loaded")

# Step 4: Load LoRA adapters and merge
print(f"\n🔀 Step 4: Merging LoRA adapters with base model...")
model_with_lora = PeftModel.from_pretrained(base_model, "./lora_adapters")
merged_model = model_with_lora.merge_and_unload()
print("   ✓ LoRA adapters merged into base model")

# Step 5: Save merged model locally
print(f"\n💾 Step 5: Saving merged model locally...")
os.makedirs("./merged_model", exist_ok=True)
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
print("   ✓ Merged model saved to ./merged_model")

# Get HF token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Step 6: Upload LoRA adapters to Hugging Face Hub
print(f"\n🚀 Step 6: Uploading LoRA adapters to Hugging Face Hub...")
adapter_repo_id = f"{HF_USERNAME}/{ADAPTER_NAME_ON_HUB}"
print(f"   Repository: {adapter_repo_id}")

if not hf_token:
    print("   ⚠️  HF_TOKEN not found in environment")
    print("   Please set it: export HF_TOKEN='your_token_here'")
    print("   Or get it from: https://huggingface.co/settings/tokens")
else:
    try:
        # Create adapter repository
        api = HfApi()
        try:
            create_repo(adapter_repo_id, token=hf_token, exist_ok=True)
            print(f"   ✓ Adapter repository created/verified: {adapter_repo_id}")
        except Exception as e:
            print(f"   Note: {e}")
        
        # Add base_model info to adapter_config.json for proper metadata
        import json
        adapter_config_path = "./lora_adapters/adapter_config.json"
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            adapter_config['base_model_name_or_path'] = BASE_MODEL
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f, indent=2)
        
        # Upload LoRA adapter files
        print("   Uploading LoRA adapter files...")
        api.upload_folder(
            folder_path="./lora_adapters",
            repo_id=adapter_repo_id,
            token=hf_token,
            commit_message=f"Upload LoRA adapters for {TINKER_MODEL_NAME}"
        )
        
        print(f"   ✓ LoRA adapters uploaded successfully!")
        print(f"   📦 Adapters available at: https://huggingface.co/{adapter_repo_id}")
        
    except Exception as e:
        print(f"   ✗ Error uploading adapters to Hub: {e}")

# Step 7: Upload merged model to Hugging Face Hub
print(f"\n🚀 Step 7: Uploading merged model to Hugging Face Hub...")
merged_repo_id = f"{HF_USERNAME}/{MODEL_NAME_ON_HUB}"
print(f"   Repository: {merged_repo_id}")

if hf_token:
    try:
        # Create merged model repository
        api = HfApi()
        try:
            create_repo(merged_repo_id, token=hf_token, exist_ok=True)
            print(f"   ✓ Merged model repository created/verified: {merged_repo_id}")
        except Exception as e:
            print(f"   Note: {e}")
        
        # Upload all files from merged_model directory
        print("   Uploading merged model files...")
        api.upload_folder(
            folder_path="./merged_model",
            repo_id=merged_repo_id,
            token=hf_token,
            commit_message=f"Upload merged {TINKER_MODEL_NAME} model"
        )
        
        print(f"   ✓ Merged model uploaded successfully!")
        print(f"   🎉 Model available at: https://huggingface.co/{merged_repo_id}")
        
    except Exception as e:
        print(f"   ✗ Error uploading merged model to Hub: {e}")

# Cleanup (optional)
print(f"\n🧹 Cleanup...")
import shutil
try:
    os.remove("cadmonkey.tar")
    shutil.rmtree("./lora_adapters")
    print("   ✓ Cleaned up temporary files (adapters)")
except Exception as e:
    print(f"   Note: {e}")

print("\n" + "="*60)
print("PROCESS COMPLETE!")
print("="*60)
print(f"Local merged model: ./merged_model")
print(f"\nHugging Face Hub:")
print(f"  - LoRA Adapters: https://huggingface.co/{adapter_repo_id}")
print(f"  - Merged Model:  https://huggingface.co/{merged_repo_id}")
print("\nTo use the LoRA adapters later:")
print(f"  from peft import PeftModel")
print(f"  from transformers import AutoModelForCausalLM")
print(f"  base = AutoModelForCausalLM.from_pretrained('{BASE_MODEL}')")
print(f"  model = PeftModel.from_pretrained(base, '{adapter_repo_id}')")
