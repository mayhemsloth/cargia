# Training Configuration Files

This directory contains YAML configuration files for different training scenarios in the Cargia training sequence.

## Why YAML?

YAML was chosen over JSON for configuration files because it offers:
- **Better readability**: Clean, indented structure without quotes and brackets
- **Comments support**: Can add explanatory comments to configurations
- **Easier editing**: Less syntax overhead, especially for nested structures
- **Human-friendly**: More intuitive for manual configuration editing

## Configuration Structure

Each configuration file follows this structure:
- `training`: Core training parameters including the new `data_sample_maximum_limit`
- `augmentation`: Data augmentation settings
- `training_args`: SFTTrainer arguments
- `paths`: Environment-specific paths (local vs cloud)

## New Features

### `data_sample_maximum_limit`
- **Type**: `Optional[int]`
- **Default**: `null` (use full dataset)
- **Purpose**: Limits the total number of training samples for controlled testing

**Usage Examples**:
- `"data_sample_maximum_limit": 1` - Test with single sample (overfitting test)
- `"data_sample_maximum_limit": 8` - Test with 8 samples (small dataset test)
- `"data_sample_maximum_limit": null` - Use full dataset (production training)

## Testing Sequence

### Step 1: Overfit Single Sample
```bash
python train_cli.py --config configs/step_1_overfit_single.yaml --local
```
- **Purpose**: Ensures core training code loop works
- **Data**: 1 sample
- **Augmentations**: Disabled
- **Expected**: Should overfit quickly

### Step 2: Overfit Eight Samples
```bash
python train_cli.py --config configs/step_2_overfit_eight.yaml --local
```
- **Purpose**: Further ensures core training code loop works
- **Data**: 8 samples
- **Augmentations**: Disabled
- **Expected**: Should overfit, but may take longer

### Step 3: Overfit Single Sample with Augmentations
```bash
python train_cli.py --config configs/step_3_overfit_single_with_aug.yaml --local
```
- **Purpose**: Ensures augmentations code works as expected
- **Data**: 1 sample
- **Augmentations**: Enabled (color, character, spatial)
- **Expected**: Should overfit but take longer than without augmentations

### Step 4: Overfit Eight Samples with Augmentations
```bash
python train_cli.py --config configs/step_4_overfit_eight_with_aug.yaml --local
```
- **Purpose**: Tests model's ability to memorize with LoRA training
- **Data**: 8 samples
- **Augmentations**: Enabled
- **Expected**: May not overfit completely - depends on LoRA capacity

### Step 5: Full Dataset Training
```bash
python train_cli.py --config configs/step_5_full_dataset.yaml --cloud
```
- **Purpose**: Production training after overfitting tests pass
- **Data**: Full dataset (no limit)
- **Augmentations**: Enabled
- **Expected**: General training, not overfitting

## Cloud Deployment

For cloud deployment, use the `--cloud` flag:
```bash
python train_cli.py --config configs/step_X_config.yaml --cloud
```

The cloud configuration will automatically use the appropriate paths for your cloud environment.

## LoRA Settings

**Important**: LoRA settings remain constant throughout all testing steps:
- `lora_alpha`: 16
- `lora_dropout`: 0.05
- `r`: 2 (rank)
- `target_modules`: ["q_proj"] (minimal for testing)

Only change these settings after all tests pass and you're ready for production training.

## Custom Configurations

To create custom configurations:
1. Copy an existing config file
2. Modify the `data_sample_maximum_limit` as needed
3. Adjust `max_steps` based on expected training time
4. Enable/disable augmentations as needed
5. Use appropriate environment paths 