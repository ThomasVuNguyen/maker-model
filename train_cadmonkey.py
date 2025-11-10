"""
CadMonkey Training Script
Fine-tune Llama 3.2 1B on OpenSCAD code generation
"""

import os
import glob
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, TextStreamer
from trl import SFTConfig, SFTTrainer

# ============================================================================
# Configuration
# ============================================================================

MAX_SEQ_LENGTH = 2048
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
MAX_NEW_TOKENS = 2048  # Allow longer code generation

# Paths
OUTPUT_DIR = "outputs"
LORA_MODEL_DIR = "lora_model"

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
        result = tokenizer.batch_decode(outputs)
        return result

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
        print("\nExample generations:")
        print("\n--- Sphere ---")
        generate_code(model, tokenizer, "hey cadmonkey, create me a sphere")
        print("\n--- Cat ---")
        generate_code(model, tokenizer, "hey cadmonkey, create me a cat")
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
    dataset_v0 = load_dataset("ThomasTheMaker/Synthetic-Object-v0", split="train")
    print(f"   Dataset v0 size: {len(dataset_v0)} examples")

    try:
        dataset_v1 = load_dataset("ThomasTheMaker/Synthetic-Object", split="train")
        print(f"   Dataset v1 size: {len(dataset_v1)} examples")

        # Combine datasets
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([dataset_v0, dataset_v1])
        print(f"   Combined dataset size: {len(dataset)} examples")
    except Exception as e:
        print(f"   ⚠️  Could not load v1 dataset: {e}")
        print(f"   Using only v0 dataset")
        dataset = dataset_v0

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

    # Test inference
    print("\n🧪 Testing inference...\n")
    FastLanguageModel.for_inference(model)

    print("--- Generating sphere ---")
    generate_code(model, tokenizer, "hey cadmonkey, create me a sphere")

    print("\n--- Generating cat ---")
    generate_code(model, tokenizer, "hey cadmonkey, create me a cat")

    print("\n" + "="*60)
    print("✨ All done!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
