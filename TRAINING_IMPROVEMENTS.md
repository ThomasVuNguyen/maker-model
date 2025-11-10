# Training Improvements for CadMonkey Model

## Problem
The model was generating reserved special tokens (`<|reserved_special_token_XXX|>`) after the actual OpenSCAD code, indicating the model hadn't properly learned when to stop generation.

## Root Causes
1. **Too few training iterations** (only 1 epoch)
2. **Learning rate too high** (2e-4) - caused unstable learning
3. **Not enough data diversity** (only one dataset)
4. **Insufficient warmup** (only 10 steps)

## Solutions Implemented

### 1. Extended Training Duration
- **Before**: 1 epoch (~207 steps)
- **After**: 3 epochs (~621 steps)
- **Why**: Model needs more iterations to learn proper stopping behavior and reduce overfitting to weird patterns

### 2. Lower Learning Rate
- **Before**: 2e-4
- **After**: 5e-5 (2.5x lower)
- **Why**: Lower LR allows more stable convergence and prevents the model from learning spurious correlations with reserved tokens

### 3. Better Learning Rate Schedule
- **Before**: Linear decay
- **After**: Cosine decay
- **Why**: Cosine provides smoother transitions and often leads to better final performance

### 4. Increased Warmup Steps
- **Before**: 10 steps
- **After**: 50 steps
- **Why**: Gradual warmup prevents early instability that could cause the model to latch onto bad patterns

### 5. Higher Weight Decay
- **Before**: 0.001
- **After**: 0.01 (10x higher)
- **Why**: Stronger regularization prevents overfitting to noise in the data

### 6. Combined Datasets
- **Before**: Only `ThomasTheMaker/Synthetic-Object-v0` (~26k examples)
- **After**: Both `v0` + `Synthetic-Object` (potentially ~50k+ examples)
- **Why**: More diverse training data helps the model generalize better and reduces memorization of artifacts

### 7. Checkpoint Strategy
- **New**: Save checkpoints at end of each epoch
- **New**: Keep only last 2 checkpoints to save disk space
- **Why**: Allows evaluation of model quality after each epoch

## Expected Results

After retraining with these settings:
- ✅ Model should stop cleanly at `<|eot_id|>` without reserved tokens
- ✅ More consistent code quality across different prompts
- ✅ Better handling of edge cases
- ✅ Reduced repetition and artifacts

## How to Retrain

1. Delete existing training artifacts:
   ```bash
   rm -rf outputs lora_model
   ```

2. Run training script:
   ```bash
   python train_cadmonkey.py
   ```

3. Training will take approximately 3x longer (but still manageable on RTX 6000)

## Monitoring Training

Watch for these signs during training:
- **Loss should decrease smoothly** without sudden spikes
- **No warning messages** about attention masks or pad tokens
- **Memory usage should be stable** around 40-45% of GPU

## Alternative Approaches (if this doesn't work)

If reserved tokens still appear after retraining:

1. **Add explicit EOS token forcing**: Modify the dataset formatting to ensure every example ends with exactly `<|eot_id|>`
2. **Filter training data**: Remove any examples that might have artifacts
3. **Try base Llama-3.2-1B-Instruct**: It's already instruction-tuned and might have better stopping behavior
4. **Add stopping criteria during inference**: Use a custom stopping function that halts at first reserved token
