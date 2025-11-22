#!/usr/bin/env python3
"""
Dataset Generator for CadMonkey
Downloads ThomasTheMaker/Synthetic-Object-v0, generates OpenSCAD code using Ollama,
validates the code with OpenSCAD, and saves to a new dataset JSON file.
"""

import os
import json
import subprocess
import tempfile
import time
import threading
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configuration
DATASET_NAME = "ThomasTheMaker/Synthetic-Object"
MODEL_NAME = "moonshotai/Kimi-K2-Instruct"  # API model to use
OUTPUT_FILE = "generated_dataset_v10.json"
OPENSCAD_TIMEOUT = 10  # seconds
NUM_SAMPLES = None  # Number of samples to process (set to None for all)
MAX_RETRIES = 5  # Maximum number of regeneration attempts if validation fails
PARALLEL_WORKERS = 10  # Number of parallel API calls

# Initialize OpenAI client
RIFT_API_KEY = os.getenv("RIFT_API_KEY")
if not RIFT_API_KEY:
    print("❌ RIFT_API_KEY not found in .env file")
    exit(1)

client = openai.OpenAI(
    api_key=RIFT_API_KEY,
    base_url="https://inference.cloudrift.ai/v1"
)

# Thread lock for safe checkpoint saving
checkpoint_lock = threading.Lock()

# Prompt template for generating OpenSCAD code
PROMPT_TEMPLATE = """You are an expert OpenSCAD programmer. Generate valid OpenSCAD code to create a 3D model of: {name}

Requirements:
- Use only standard OpenSCAD primitives and operations
- Code must be syntactically correct and compilable
- Include appropriate transformations and positioning
- Add comments explaining the design
- Make it visually recognizable as a {name}

Generate ONLY the OpenSCAD code, no explanations:"""


def call_api(prompt, model=MODEL_NAME):
    """Call OpenAI-compatible API to generate code."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
            timeout=60
        )

        # Get the response content
        if hasattr(completion, 'choices') and len(completion.choices) > 0:
            message = completion.choices[0].message
            if hasattr(message, 'content') and message.content:
                return message.content.strip()

        print(f"❌ API returned empty response")
        return None
    except Exception as e:
        print(f"❌ API error: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_openscad_code(response):
    """Extract OpenSCAD code from the LLM response."""
    if not response:
        return None

    # Remove markdown code blocks if present
    code = response
    if "```openscad" in code:
        parts = code.split("```openscad")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0]

    # Remove common prefixes/explanations
    lines = code.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip explanation lines
        if stripped.startswith("Here") or stripped.startswith("Sure") or \
           stripped.startswith("I'll") or stripped.startswith("This"):
            continue
        # Found code start
        if stripped and (stripped.startswith("//") or
                        stripped.startswith("module") or
                        stripped.startswith("union") or
                        stripped.startswith("difference") or
                        any(kw in stripped for kw in ["sphere", "cube", "cylinder", "translate", "rotate"])):
            start_idx = i
            break

    code = '\n'.join(lines[start_idx:])
    return code.strip()


def validate_openscad_code(code):
    """Validate OpenSCAD code by trying to compile it."""
    if not code:
        return False, "Empty code"

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Try to compile with OpenSCAD
        result = subprocess.run(
            ["openscad", "--render", "-o", "/tmp/test_compile.3mf", temp_file],
            capture_output=True,
            text=True,
            timeout=OPENSCAD_TIMEOUT
        )

        # Clean up
        os.unlink(temp_file)
        if os.path.exists("/tmp/test_compile.3mf"):
            os.unlink("/tmp/test_compile.3mf")

        # Check result
        if result.returncode == 0:
            # Check for warnings
            if "WARNING" in result.stdout or "WARNING" in result.stderr:
                return True, f"Valid (with warnings)"
            return True, "Valid"
        else:
            # Extract error message
            error_lines = [line for line in result.stderr.split('\n')
                          if "ERROR" in line or "error" in line]
            error_msg = error_lines[0] if error_lines else "Compilation failed"
            return False, error_msg
    except subprocess.TimeoutExpired:
        os.unlink(temp_file)
        return False, "Compilation timeout"
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False, str(e)


def generate_and_validate(name, max_retries=MAX_RETRIES):
    """Generate OpenSCAD code for a given name and validate it.

    Retries up to max_retries times if validation fails.
    """
    print(f"\n{'='*60}")
    print(f"Generating code for: {name}")
    print('='*60)

    # Generate prompt
    prompt = PROMPT_TEMPLATE.format(name=name)

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"\n🔄 Retry attempt {attempt}/{max_retries}...")

        # Call LLM
        print(f"🤖 Calling {MODEL_NAME}...")
        response = call_api(prompt)

        if not response:
            print(f"❌ Failed to generate code")
            if attempt < max_retries:
                continue
            return None, f"Failed to generate code after {max_retries} attempts"

        # Extract code
        code = extract_openscad_code(response)
        if not code:
            print(f"❌ Failed to extract code")
            if attempt < max_retries:
                continue
            return None, f"Failed to extract code after {max_retries} attempts"

        print(f"📝 Generated {len(code)} characters of code")
        preview = code[:150] + "..." if len(code) > 150 else code
        print(f"Preview: {preview}")

        # Validate with OpenSCAD
        print(f"✓ Validating with OpenSCAD...")
        is_valid, message = validate_openscad_code(code)

        if is_valid:
            if attempt > 1:
                print(f"✅ {message} (succeeded on attempt {attempt})")
            else:
                print(f"✅ {message}")
            return code, message
        else:
            print(f"❌ Validation failed: {message}")
            if attempt < max_retries:
                print(f"   Will retry with a new generation...")
            else:
                print(f"   Max retries ({max_retries}) reached, giving up.")

    return None, f"Failed validation after {max_retries} attempts"


def load_checkpoint():
    """Load existing results from checkpoint file if it exists."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)
            completed_names = {r["name"] for r in results}
            print(f"📂 Found existing checkpoint: {len(results)} items already completed")
            return results, completed_names
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            return [], set()
    return [], set()


def save_checkpoint(results):
    """Save current results to checkpoint file (thread-safe)."""
    with checkpoint_lock:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)


def process_item(example):
    """Process a single dataset item (designed to run in parallel)."""
    name = example["name"]
    original_code = example.get("code", "")

    # Generate new code with retries
    generated_code, status = generate_and_validate(name)

    if generated_code:
        return {
            "success": True,
            "data": {
                "name": name,
                "code": generated_code,
                "original_code": original_code,
                "status": status,
                "valid": True
            },
            "needed_retry": "attempt" in status.lower()
        }
    else:
        return {
            "success": False,
            "name": name,
            "error": status
        }


def main():
    print("\n" + "="*60)
    print("CadMonkey Dataset Generator")
    print("="*60 + "\n")

    # Check if OpenSCAD is installed
    try:
        result = subprocess.run(["openscad", "--version"], capture_output=True)
        if result.returncode != 0:
            print("❌ OpenSCAD not found. Install: sudo apt install openscad")
            return
    except FileNotFoundError:
        print("❌ OpenSCAD not found. Install: sudo apt install openscad")
        return

    # Load checkpoint if exists
    results, completed_names = load_checkpoint()

    # Load dataset
    print(f"📥 Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    # Take first N samples
    if NUM_SAMPLES:
        dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

    # Filter out already completed items
    if completed_names:
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: x["name"] not in completed_names)
        skipped = original_size - len(dataset)
        print(f"⏭️  Skipping {skipped} already completed items")

    print(f"📊 Processing {len(dataset)} remaining samples with {PARALLEL_WORKERS} parallel workers\n")

    # Generate and validate
    stats = {
        "total": len(dataset),
        "success": 0,
        "failed": 0,
        "retries": 0,  # Track total number of retries
        "lock": threading.Lock()  # Lock for thread-safe stats updates
    }

    # Track timing for ETA
    start_time = time.time()

    # Process items in parallel
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_item, example): example for example in dataset}

        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1

            try:
                result = future.result()

                # Update stats in thread-safe manner
                with stats["lock"]:
                    if result["success"]:
                        results.append(result["data"])
                        stats["success"] += 1
                        if result["needed_retry"]:
                            stats["retries"] += 1

                        # Save checkpoint after each success
                        try:
                            save_checkpoint(results)
                            print(f"💾 Checkpoint saved: {result['data']['name']} ({len(results)} items total)")
                        except Exception as e:
                            print(f"⚠️  Failed to save checkpoint: {e}")
                    else:
                        stats["failed"] += 1
                        print(f"⏭️ Skipping {result['name']} due to validation failure: {result['error']}")

                # Display progress
                elapsed = timedelta(seconds=int(time.time() - start_time))
                avg_time = (time.time() - start_time) / completed
                remaining = len(dataset) - completed
                eta_seconds = avg_time * remaining
                eta = timedelta(seconds=int(eta_seconds))

                print(f"\n⏱️  Progress: {completed}/{len(dataset)} | Elapsed: {elapsed} | ETA: {eta} | Avg: {avg_time:.1f}s/item")
                print(f"📊 Stats: ✅ {stats['success']} success | ❌ {stats['failed']} failed | 🔄 {stats['retries']} retries")

            except Exception as e:
                print(f"❌ Error processing item: {e}")
                with stats["lock"]:
                    stats["failed"] += 1

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    # Final save
    print(f"\n{'='*60}")
    print("Finalizing results...")
    print('='*60)

    try:
        save_checkpoint(results)
        print(f"✅ Final dataset saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Failed to save final results: {e}")

    # Calculate overall statistics (including previously completed items)
    total_in_dataset = len(results)
    success_this_run = stats['success']
    failed_this_run = stats['failed']

    print(f"\n⏱️  Session time: {total_time_str}")
    print(f"📊 Session Statistics:")
    print(f"   Processed:       {stats['total']} items this session")
    print(f"   Success:         {success_this_run} ({success_this_run/max(stats['total'], 1)*100:.1f}%)")
    print(f"   Failed:          {failed_this_run} ({failed_this_run/max(stats['total'], 1)*100:.1f}%)")
    print(f"   Needed retries:  {stats['retries']} items required regeneration")
    if times:
        avg_time = sum(times) / len(times)
        print(f"   Avg time:        {avg_time:.1f}s per item")

    print(f"\n📊 Total Dataset Statistics:")
    print(f"   Total items:     {total_in_dataset}")
    print(f"   Dataset file:    {OUTPUT_FILE}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
