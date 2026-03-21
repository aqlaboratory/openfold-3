# OpenFold3 Training

Welcome to the training documentation for OpenFold3. This guide covers how to train OpenFold3 on the PDB dataset from scratch or fine-tune from an existing checkpoint.

## 1. Prerequisites

- OpenFold3 Conda environment. See [OpenFold3 Installation](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/Installation.md) for instructions on how to build the environment.

## 2. Download the Dataset

The pre-processed PDB training dataset is hosted on AWS S3. Download it using the AWS CLI:

```bash
aws s3 sync s3://openfold3-data/pdb_training_set/ /shared/openfold3/pdb_training_set/ --no-sign-request

```

## 3. Prepare the Training Config

The training configuration is stored in a YAML file that controls all aspects of training: model settings, dataset configuration, distributed training parameters, logging, and checkpointing.

**Note:** Make sure you update the paths to match your file locations. The examples below assume a `/shared/openfold3` directory that's accessible from all your training nodes.

Complete example YAML configurations for all stages of training are available in [examples/training_yamls/](https://github.com/aqlaboratory/openfold-3/tree/main/examples/training_yamls).

### 3.1 Basic Training Config

Here's a minimal configuration for single-GPU training:

```yaml
experiment_settings:
  mode: train
  output_dir: ./test_train_output 
  restart_checkpoint_path: last

data_module_args:
  batch_size: 1
  num_workers: 1
  epoch_len: 500  # Ckpt every 500 steps (effective batch_size * # of steps)

logging_config:
  log_lr: false
  wandb_config: null

pl_trainer_args:
  devices: 1
  num_nodes: 1
  precision: bf16-mixed
  max_epochs: -1
  log_every_n_steps: 50

checkpoint_config:
  every_n_epochs: 1
  auto_insert_metric_name: false
  save_last: true
  save_top_k: -1

model_update:
  presets: 
    - train
  custom:
    architecture:
      shared:
        use_confidence_emb_prob: 0.8
        diffusion:
          use_conditioning_prob: 0.8

dataset_configs:
  train:
    weighted-pdb:
      dataset_class: WeightedPDBDataset
      weight: 1.0
      config:
        debug_mode: true
        template:
          n_templates: 4
          take_top_k: false
        crop:
          token_crop:
            enabled: true
            token_budget: 384
            crop_weights:
              contiguous: 0.2
              spatial: 0.4
              spatial_interface: 0.4
          chain_crop:
            enabled: true

  validation:
    val-weighted-pdb:
      dataset_class: ValidationPDBDataset
      config:
        debug_mode: true
        msa:
          subsample_main: false
        template:
          n_templates: 4
          take_top_k: true
        crop:
          token_crop:
            enabled: false

dataset_paths:
  weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none 
    alignment_array_directory: /shared/openfold3/pdb_training_set/alignment_arrays
    dataset_cache_file: /shared/openfold3/pdb_training_set/dataset_caches/training_cache_with_templates.json
    target_structures_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/reference_mols
    template_cache_directory: /shared/openfold3/pdb_training_set/templates/train_template_cache
    template_structure_array_directory: /shared/openfold3/pdb_training_set/templates/template_structure_arrays
    template_structures_directory: none
    template_file_format: npz
    ccd_file: null

  val-weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none 
    alignment_array_directory: /shared/openfold3/pdb_training_set/alignment_arrays
    dataset_cache_file: /shared/openfold3/pdb_training_set/dataset_caches/validation_cache_with_templates.json
    target_structures_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/reference_mols
    template_cache_directory: /shared/openfold3/pdb_training_set/templates/val_template_cache
    template_structure_array_directory: /shared/openfold3/pdb_training_set/templates/template_structure_arrays
    template_structures_directory: none
    template_file_format: npz
    ccd_file: null
```

For example configurations for all stages of training, please see [examples/training_yamls/](https://github.com/aqlaboratory/openfold-3/tree/main/examples/training_yamls):
- `initial_training.yml`: Standard initial training configuration
- `finetune_1.yml`: Fine-tuning stage 1 configuration
- `finetune_2.yml`: Fine-tuning stage 2 configuration
- `finetune_3.yml`: Fine-tuning stage 3 configuration (used in OF3p)

## 4. Launch Training

### 4.1 Single-GPU Training

For testing or debugging:

```bash
run_openfold train --runner-yaml training.yaml --seed 42
```

### 4.2 Multi-GPU Training

To train on multiple GPUs within a single node, configure your YAML:

```yaml
pl_trainer_args:
  devices: 8      # GPUs per node
  num_nodes: 1
```

For multi-node distributed training, update your config as follows:

```yaml
pl_trainer_args:
  devices: 8       # GPUs per node
  num_nodes: 32    # Total number of nodes
```

Then launch training with:

```bash
run_openfold train --runner_yaml training.yaml --seed 42
```

## 5. Monitoring Training

Enable Weights & Biases logging by configuring `wandb_config` in your YAML:

```yaml
logging_config:
  log_lr: true
  wandb_config:
    project: openfold3-training
    entity: your-wandb-entity
    group: null
    id: null
    experiment_name: my-training-run
```

To use W&B, ensure you're logged in:

```bash
wandb login
```

## 6. Checkpointing and Resuming

### 6.1 Checkpoint Configuration

```yaml
checkpoint_config:
  every_n_epochs: 1              # Save checkpoint every N epochs
  auto_insert_metric_name: false # Don't add metric to filename
  save_last: true                # Always save 'last.ckpt'
  save_top_k: -1                 # Keep all checkpoints (-1) or top K
```

Checkpoints are saved to `{output_dir}/checkpoints/`.

### 6.2 Resuming Training

To resume from the last checkpoint:

```yaml
experiment_settings:
  restart_checkpoint_path: last
```

To resume from a specific checkpoint:

```yaml
experiment_settings:
  restart_checkpoint_path: /path/to/checkpoint.ckpt
```

## 7. Fine-tuning

To fine-tune from a pre-trained checkpoint, specify the checkpoint path and adjust training parameters as needed:

```yaml
experiment_settings:
  mode: train
  output_dir: ./finetune_output
  seed: 42
  restart_checkpoint_path: /path/to/pretrained.ckpt
```