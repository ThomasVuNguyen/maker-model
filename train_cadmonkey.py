"""
CadMonkey Training Script
Fine-tune Llama 3.2 1B on OpenSCAD code generation
"""

import os
import glob
import torch
import random
import subprocess
from datetime import datetime
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, TextStreamer
from trl import SFTConfig, SFTTrainer

# ============================================================================
# Configuration
# ============================================================================

MAX_SEQ_LENGTH = 4096
DTYPE = None  # None for auto detection
LOAD_IN_4BIT = True
MODEL_NAME = "unsloth/Llama-3.2-1B"

# LoRA Configuration
LORA_R = 256
LORA_ALPHA = 512
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training Configuration
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 50  # Increased for more gradual warmup
NUM_EPOCHS = 3  # Train for 3 epochs instead of 1
LEARNING_RATE = 5e-5  # Lower learning rate for more stable training
LOGGING_STEPS = 10
WEIGHT_DECAY = 0.01  # Increased weight decay to prevent overfitting
LR_SCHEDULER_TYPE = "cosine"  # Cosine decay instead of linear

# Generation Configuration
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.1
MAX_NEW_TOKENS = 4096  # Allow longer code generation

# Paths
OUTPUT_DIR = "outputs"
LORA_MODEL_DIR = "lora_model"
EVAL_OUTPUT_DIR = "eval_outputs"

# Dataset Configuration - Add or remove dataset names as needed
DATASET_NAMES = [
    "ThomasTheMaker/Synthetic-Openscad-v1",
    "ThomasTheMaker/Synthetic-Openscad-v0",
    "ThomasTheMaker/Synthetic-Openscad-v2",
    "ThomasTheMaker/Synthetic-Openscad-v3",
]

# OpenSCAD rendering configuration
OPENSCAD_BINARY = "openscad"  # Change to full path if not in PATH
RENDER_TIMEOUT = 30  # seconds per render
IMAGE_SIZE = "800,600"  # width,height
ENABLE_RENDERING = False  # Set to True when xvfb is installed

# Evaluation prompts
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

# ============================================================================
# Helper Functions
# ============================================================================

def check_training_status():
    """Check if training has already been completed or can be resumed."""
    if os.path.exists(OUTPUT_DIR) and os.path.exists(LORA_MODEL_DIR):
        print("⚠️  Training appears to be already completed!")
        print(f"   Found '{OUTPUT_DIR}' and '{LORA_MODEL_DIR}' directories.")
        print("   To retrain from scratch, manually delete these directories.")
        return "completed"
    elif os.path.exists(OUTPUT_DIR):
        checkpoint_dirs = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
        if checkpoint_dirs:
            print(f"✓ Found {len(checkpoint_dirs)} checkpoint(s) in '{OUTPUT_DIR}' directory.")
            print("  Training will resume from the last checkpoint.")
            return "resume"
        else:
            print(f"✓ Found '{OUTPUT_DIR}' directory but no checkpoints.")
            return "fresh"
    else:
        print("✓ Starting fresh training...")
        return "fresh"

def formatting_prompts_func(examples, tokenizer):
    """Format dataset examples into chat format."""
    names = examples["name"]
    codes = examples["code"]
    texts = []

    for name, code in zip(names, codes):
        convo = [
            {"role": "user", "content": f"hey cadmonkey, create me a {name}"},
            {"role": "assistant", "content": code}
        ]
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    return {"text": texts}

def show_memory_stats(stage="current"):
    """Display GPU memory statistics."""
    gpu_stats = torch.cuda.get_device_properties(0)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\n{'='*60}")
    print(f"GPU Memory Stats ({stage})")
    print(f"{'='*60}")
    print(f"GPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Reserved memory: {used_memory} GB")
    print(f"{'='*60}\n")
    return used_memory, max_memory

def generate_code(model, tokenizer, prompt, stream=True):
    """Generate OpenSCAD code from a prompt."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    if stream:
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        return None
    else:
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        result = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return result[0] if result else ""

def create_eval_folder():
    """Create a uniquely numbered evaluation output folder."""
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # Generate a random 6-digit number + timestamp for uniqueness
    random_id = random.randint(100000, 999999)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"eval_{timestamp}_{random_id}"
    folder_path = os.path.join(EVAL_OUTPUT_DIR, folder_name)

    os.makedirs(folder_path, exist_ok=True)
    print(f"\n📁 Created evaluation folder: {folder_path}")
    return folder_path

def extract_code_from_response(response):
    """Extract just the OpenSCAD code from the model response."""
    import re

    # Step 1: Extract content after assistant header if present
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        parts = response.split("<|start_header_id|>assistant<|end_header_id|>")
        response = parts[-1] if len(parts) > 1 else response

    # Step 2: Remove the chat template header junk (system/user text that appears as regular tokens)
    # Look for where the actual code starts
    lines = response.split('\n')
    start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines, system/user/assistant keywords, and metadata
        if not stripped or stripped in ['system', 'user', 'assistant'] or \
           'Cutting Knowledge Date:' in line or 'Today Date:' in line or \
           stripped.endswith('assistant'):
            continue
        # Found first line of actual content
        start_idx = i
        break

    lines = lines[start_idx:]
    response = '\n'.join(lines)

    # Step 3: Remove special tokens
    for token in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>", "<|reserved_special_token_"]:
        if token in response:
            response = response.split(token)[0]

    # Step 4: Remove garbage at the end by checking for non-ASCII or nonsense
    lines = response.split('\n')
    valid_lines = []

    for line in lines:
        # Check if line is mostly ASCII printable characters
        if not line.strip():  # Keep empty lines
            valid_lines.append(line)
            continue

        try:
            # Calculate ratio of ASCII printable characters
            ascii_count = sum(1 for c in line if 32 <= ord(c) < 127)
            ascii_ratio = ascii_count / max(len(line), 1)

            # If line has lots of non-ASCII or looks like garbage, stop
            if ascii_ratio < 0.8:
                break

            # Check for common garbage patterns
            if re.search(r'[а-яА-ЯａристиндивидуkrvldkfилактиЎыџN]{5,}', line):  # Cyrillic/garbage strings
                break

            valid_lines.append(line)
        except:
            break

    return '\n'.join(valid_lines).strip()

def render_scad_file(scad_path, output_image_path):
    """Render a .scad file to a PNG image using OpenSCAD."""
    try:
        # Set environment variables for headless rendering
        env = os.environ.copy()
        env['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering
        env['GALLIUM_DRIVER'] = 'softpipe'  # Use software pipe driver

        # Try to use xvfb-run if available, otherwise use DISPLAY=:99
        xvfb_available = subprocess.run(['which', 'xvfb-run'],
                                       capture_output=True).returncode == 0

        if xvfb_available:
            cmd = [
                'xvfb-run', '-a', '-s', '-screen 0 1024x768x24',
                OPENSCAD_BINARY,
                "-o", output_image_path,
                "--imgsize", IMAGE_SIZE,
                "--colorscheme", "BeforeDawn",
                "--viewall",
                "--autocenter",
                scad_path
            ]
        else:
            # Fallback: try with software rendering and fake display
            cmd = [
                OPENSCAD_BINARY,
                "-o", output_image_path,
                "--imgsize", IMAGE_SIZE,
                "--colorscheme", "BeforeDawn",
                "--viewall",
                "--autocenter",
                "--render",  # Force render mode
                scad_path
            ]
            env['DISPLAY'] = ':99'  # Fake display

        result = subprocess.run(
            cmd,
            timeout=RENDER_TIMEOUT,
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode == 0 and os.path.exists(output_image_path):
            return True, None
        else:
            error_msg = result.stderr if result.stderr else "Unknown error"
            # If it's just a display error, report it differently
            if "DISPLAY" in error_msg or "X server" in error_msg:
                return False, "No display available (install xvfb: sudo apt install xvfb)"
            return False, error_msg
    except subprocess.TimeoutExpired:
        return False, "Rendering timeout"
    except FileNotFoundError:
        return False, f"OpenSCAD not found at '{OPENSCAD_BINARY}'"
    except Exception as e:
        return False, str(e)

def save_eval_results(model, tokenizer, eval_folder):
    """Generate code for evaluation prompts and save to files."""
    print(f"\n🧪 Running evaluation on {len(EVAL_PROMPTS)} prompts...")
    print("="*60)

    render_results = []

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        # Extract object name from prompt
        object_name = prompt.replace("hey cadmonkey, create me a ", "").replace(" ", "_")

        print(f"\n[{i}/{len(EVAL_PROMPTS)}] Generating: {object_name}")
        print("-"*60)

        # Generate code (non-streaming for saving)
        response = generate_code(model, tokenizer, prompt, stream=False)
        code = extract_code_from_response(response)

        # Print preview
        preview = code[:200] + "..." if len(code) > 200 else code
        print(preview)

        # Save to file
        filename = f"{i:02d}_{object_name}.scad"
        filepath = os.path.join(eval_folder, filename)

        with open(filepath, 'w') as f:
            f.write(f"// Generated by CadMonkey\n")
            f.write(f"// Prompt: {prompt}\n")
            f.write(f"// Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(code)

        print(f"✓ Saved to: {filename}")

        # Try to render the code (only if enabled)
        if ENABLE_RENDERING:
            image_filename = f"{i:02d}_{object_name}.png"
            image_path = os.path.join(eval_folder, image_filename)

            print(f"🎨 Rendering {object_name}...", end=" ")
            success, error = render_scad_file(filepath, image_path)

            if success:
                print("✓ Rendered successfully")
                render_results.append({
                    "name": object_name,
                    "filename": filename,
                    "image": image_filename,
                    "rendered": True,
                    "error": None
                })
            else:
                print(f"✗ Failed: {error}")
                render_results.append({
                    "name": object_name,
                    "filename": filename,
                    "image": None,
                    "rendered": False,
                    "error": error
                })
        else:
            print(f"⏭️  Rendering disabled")
            render_results.append({
                "name": object_name,
                "filename": filename,
                "image": None,
                "rendered": False,
                "error": "Rendering disabled (set ENABLE_RENDERING=True to enable)"
            })

    # Generate review markdown file
    create_review_markdown(eval_folder, render_results)

    print("\n" + "="*60)
    print(f"✅ All {len(EVAL_PROMPTS)} evaluations saved to: {eval_folder}")
    rendered_count = sum(1 for r in render_results if r["rendered"])
    print(f"🎨 Successfully rendered: {rendered_count}/{len(render_results)} models")
    print("="*60)

def create_review_markdown(eval_folder, render_results):
    """Create a markdown review template."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Calculate rendering success rate
    total = len(render_results)
    rendered = sum(1 for r in render_results if r["rendered"])
    render_rate = (rendered / total * 100) if total > 0 else 0

    markdown_content = f"""# CadMonkey Evaluation Report

**Generated at:** {timestamp}
**Total prompts:** {total}

---

## Rendering Statistics

- **Rendering Rate:** {render_rate:.1f}% ({rendered}/{total} models rendered successfully)
- **Resemblance Rate:** __% (fill in after manual review)

---

## Model Outputs

"""

    for i, result in enumerate(render_results, 1):
        markdown_content += f"### {i}. {result['name'].replace('_', ' ').title()}\n\n"
        markdown_content += f"- **File:** `{result['filename']}`\n"

        if result['rendered']:
            markdown_content += f"- **Preview:** ![{result['name']}]({result['image']})\n"
            markdown_content += "- **Rendered:** ✅ Success\n"
        else:
            markdown_content += "- **Rendered:** ❌ Failed\n"
            markdown_content += f"- **Error:** `{result['error']}`\n"

        markdown_content += "- **Quality:** __ / 5 (fill in after review)\n"
        markdown_content += "- **Resemblance:** __ / 5 (fill in after review)\n"
        markdown_content += "- **Notes:** _[Add your observations here]_\n\n"
        markdown_content += "---\n\n"

    markdown_content += """
## Review Guide

### Quality Rating (1-5)
- **5:** Perfect, production-ready code
- **4:** Good, minor improvements needed
- **3:** Acceptable, some issues present
- **2:** Poor, major issues
- **1:** Broken or unusable

### Resemblance Rating (1-5)
- **5:** Perfectly matches the prompt
- **4:** Good representation
- **3:** Recognizable but simplified
- **2:** Loosely related
- **1:** Doesn't match prompt

### Overall Scoring
- Calculate average quality and resemblance scores
- Update the "Resemblance Rate" at the top with the average resemblance score as a percentage (avg/5 * 100)
"""

    # Save markdown file
    markdown_path = os.path.join(eval_folder, "REVIEW.md")
    with open(markdown_path, 'w') as f:
        f.write(markdown_content)

    print(f"\n📝 Review template saved to: REVIEW.md")

# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    print("\n" + "="*60)
    print("CadMonkey Training Script - Llama 3.2 1B")
    print("="*60 + "\n")

    # Check training status
    training_status = check_training_status()

    if training_status == "completed":
        print("\n⏭️  Skipping training - loading existing model for inference...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=LORA_MODEL_DIR,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        FastLanguageModel.for_inference(model)

        print("\n✓ Model loaded! Ready for inference.")

        # Create evaluation folder and save results
        eval_folder = create_eval_folder()
        save_eval_results(model, tokenizer, eval_folder)

        print(f"\n📂 Evaluation files saved to: {eval_folder}")
        return

    # Load base model
    print("\n📥 Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Add LoRA adapters
    print("\n🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Setup tokenizer with chat template
    print("\n💬 Setting up chat template...")
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Load and combine datasets
    print("\n📊 Loading datasets...")
    print(f"   Attempting to load {len(DATASET_NAMES)} dataset(s)...\n")

    loaded_datasets = []

    for i, dataset_name in enumerate(DATASET_NAMES, 1):
        try:
            print(f"   [{i}/{len(DATASET_NAMES)}] Loading: {dataset_name}...")
            ds = load_dataset(dataset_name, split="train")
            loaded_datasets.append(ds)
            print(f"       ✓ Loaded {len(ds)} examples")
        except Exception as e:
            print(f"       ✗ Failed to load: {e}")
            continue

    if not loaded_datasets:
        print("\n❌ Error: No datasets could be loaded!")
        print("   Please check DATASET_NAMES configuration and network connection.")
        return

    # Combine all loaded datasets
    if len(loaded_datasets) == 1:
        dataset = loaded_datasets[0]
        print(f"\n   Using single dataset: {len(dataset)} examples")
    else:
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(loaded_datasets)
        print(f"\n   Combined {len(loaded_datasets)} datasets: {len(dataset)} total examples")

    print("\n🔄 Formatting dataset...")
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
    )

    # Example data inspection
    print("\n📋 Example from dataset:")
    print(f"   Name: {dataset[5]['name']}")
    print(f"   Code preview: {dataset[5]['code'][:100]}...")

    # Setup trainer
    print("\n⚙️  Setting up trainer...")
    checkpoint_dirs = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
    resume_from_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None

    if resume_from_checkpoint:
        print(f"   ✓ Resuming from: {resume_from_checkpoint}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=True,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=LOGGING_STEPS,
            optim="adamw_8bit",
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
            resume_from_checkpoint=resume_from_checkpoint,
            save_strategy="epoch",  # Save at end of each epoch
            save_total_limit=2,  # Keep only last 2 checkpoints
        ),
    )

    # Train only on assistant responses
    print("\n🎯 Configuring to train only on assistant responses...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Show memory before training
    start_memory, max_memory = show_memory_stats("before training")

    # Train
    print("\n🚀 Starting training...\n")
    trainer_stats = trainer.train()

    # Show training stats
    print("\n✅ Training completed!\n")
    end_memory, _ = show_memory_stats("after training")

    if trainer_stats and hasattr(trainer_stats, 'metrics'):
        metrics = trainer_stats.metrics
        runtime = metrics['train_runtime']
        print(f"⏱️  Training time: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        print(f"📈 Memory used for training: {end_memory - start_memory:.3f} GB")
        print(f"📊 Memory usage: {end_memory/max_memory*100:.1f}% of GPU capacity")

    # Save model
    print(f"\n💾 Saving model to '{LORA_MODEL_DIR}'...")
    model.save_pretrained(LORA_MODEL_DIR)
    tokenizer.save_pretrained(LORA_MODEL_DIR)
    print("   ✓ Model saved!")

    # Test inference and save evaluation results
    print("\n🧪 Running evaluation and saving results...\n")
    FastLanguageModel.for_inference(model)

    # Create evaluation folder
    eval_folder = create_eval_folder()

    # Generate and save all evaluation examples
    save_eval_results(model, tokenizer, eval_folder)

    print("\n" + "="*60)
    print("✨ All done!")
    print(f"📂 Evaluation files saved to: {eval_folder}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
