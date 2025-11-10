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
from datetime import timedelta
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_NAME = "ThomasTheMaker/Synthetic-Object-v0"
MODEL_NAME = "gemma3:27b"  # Ollama model to use
OUTPUT_FILE = "generated_dataset.json"
OPENSCAD_TIMEOUT = 10  # seconds
NUM_SAMPLES = None  # Number of samples to process (set to None for all)
MAX_RETRIES = 5  # Maximum number of regeneration attempts if validation fails

# Prompt template for generating OpenSCAD code
PROMPT_TEMPLATE = """You are an expert OpenSCAD programmer. Generate valid OpenSCAD code to create a 3D model of: {name}

Requirements:
- Use only standard OpenSCAD primitives and operations
- Code must be syntactically correct and compilable
- Include appropriate transformations and positioning
- Add comments explaining the design
- Make it visually recognizable as a {name}

Generate ONLY the OpenSCAD code, no explanations:"""


def call_ollama(prompt, model=MODEL_NAME):
    """Call Ollama to generate code."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"❌ Ollama error: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("⏱️ Ollama timeout")
        return None
    except FileNotFoundError:
        print("❌ Ollama not found. Install it from https://ollama.ai")
        return None
    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
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
        response = call_ollama(prompt)

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


def main():
    print("\n" + "="*60)
    print("CadMonkey Dataset Generator")
    print("="*60 + "\n")

    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True)
        if result.returncode != 0:
            print("❌ Ollama not found. Install from https://ollama.ai")
            return
    except FileNotFoundError:
        print("❌ Ollama not found. Install from https://ollama.ai")
        return

    # Check if OpenSCAD is installed
    try:
        result = subprocess.run(["openscad", "--version"], capture_output=True)
        if result.returncode != 0:
            print("❌ OpenSCAD not found. Install: sudo apt install openscad")
            return
    except FileNotFoundError:
        print("❌ OpenSCAD not found. Install: sudo apt install openscad")
        return

    # Load dataset
    print(f"📥 Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    # Take first N samples
    if NUM_SAMPLES:
        dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

    print(f"📊 Processing {len(dataset)} samples\n")

    # Generate and validate
    results = []
    stats = {
        "total": len(dataset),
        "success": 0,
        "failed": 0,
        "retries": 0  # Track total number of retries
    }

    # Track timing for ETA
    start_time = time.time()
    times = []

    for i, example in enumerate(dataset):
        item_start = time.time()

        name = example["name"]
        original_code = example.get("code", "")

        # Calculate and display ETA
        if i > 0:
            avg_time = sum(times) / len(times)
            remaining = len(dataset) - i
            eta_seconds = avg_time * remaining
            eta = timedelta(seconds=int(eta_seconds))
            elapsed = timedelta(seconds=int(time.time() - start_time))

            print(f"\n⏱️  Progress: {i}/{len(dataset)} | Elapsed: {elapsed} | ETA: {eta} | Avg: {avg_time:.1f}s/item")
            print(f"📊 Current stats: ✅ {stats['success']} success | ❌ {stats['failed']} failed | 🔄 {stats['retries']} retries")

        # Generate new code with retries
        generated_code, status = generate_and_validate(name)

        if generated_code:
            results.append({
                "name": name,
                "code": generated_code,
                "original_code": original_code,
                "status": status,
                "valid": True
            })
            stats["success"] += 1

            # Track if retries were needed (check if status mentions "attempt")
            if "attempt" in status.lower():
                stats["retries"] += 1
        else:
            stats["failed"] += 1
            print(f"⏭️ Skipping {name} due to validation failure: {status}")

        # Track time for this item
        item_time = time.time() - item_start
        times.append(item_time)

        # Keep only last 10 times for rolling average
        if len(times) > 10:
            times.pop(0)

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print('='*60)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Dataset saved to: {OUTPUT_FILE}")
    print(f"⏱️  Total time: {total_time_str}")
    print(f"📊 Statistics:")
    print(f"   Total:           {stats['total']}")
    print(f"   Success:         {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"   Failed:          {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"   Needed retries:  {stats['retries']} items required regeneration")
    if times:
        avg_time = sum(times) / len(times)
        print(f"   Avg time:        {avg_time:.1f}s per item")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
