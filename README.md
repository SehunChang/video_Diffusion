# Minimal Diffusion

A minimal implementation of diffusion models with a modular architecture.

## Project Structure

The project is organized as follows:

```
minimal-diffusion/
├── architectures/       # Neural network architectures
│   ├── __init__.py      # Architecture registry
│   ├── unets.py         # Multiple UNet variants (UNet, UNetBig, UNetSmall)
│   └── resnet.py        # ResNet architecture
├── trainers/            # Training loop implementations
│   ├── __init__.py      # Trainer registry
│   ├── base_trainer.py  # Base trainer class
│   ├── standard_trainer.py  # Standard diffusion training
│   └── shared_epsilon_trainer.py  # Shared epsilon training for video
├── data.py              # Dataset loading utilities
├── main.py              # Main training script
```

## Features

- Multiple UNet variants (standard, big, small)
- Support for both image and video diffusion
- Distributed training support
- Comprehensive logging and checkpointing
- DDIM sampling
- Class-conditional generation
- EMA model averaging
- Flexible trainer system

## Usage

### Training a model

```bash
python main.py \
  --arch unet \
  --trainer standard \
  --dataset cifar10 \
  --data-dir ./dataset/ \
  --batch-size 128 \
  --epochs 500 \
  --save-dir ./trained_models/ \
  --lr 0.0001 \
  --ema_w 0.9995
```

### Training with distributed data parallel

```bash
python -m torch.distributed.launch --nproc_per_node=N main.py \
  --arch unet \
  --trainer standard \
  --dataset cifar10 \
  --data-dir ./dataset/ \
  --batch-size 128 \
  --epochs 500 \
  --save-dir ./trained_models/
```

### Training with shared epsilon (for video)

```bash
python main.py \
  --arch unet \
  --trainer shared_epsilon \
  --dataset video_dataset \
  --data-dir ./dataset/ \
  --batch-size 32 \
  --epochs 500 \
  --reg-weight 0.1 \
  --save-dir ./trained_models/ \
  --seq-len 16
```

### Sampling from a trained model

```bash
python main.py \
  --arch unet \
  --dataset cifar10 \
  --sampling-only \
  --num-sampled-images 1000 \
  --pretrained-ckpt ./trained_models/checkpoint_epoch_500.pt \
  --sample-out-dir ./sampled_images/ \
  --sampling-steps 250 \
  --ddim
```

### Resuming training from a checkpoint

```bash
python main.py \
  --arch unet \
  --trainer standard \
  --dataset cifar10 \
  --data-dir ./dataset/ \
  --resume ./trained_models/checkpoint_epoch_100.pt \
  --save-dir ./trained_models/
```

## Command Line Arguments

### Model Configuration
- `--arch`: Architecture to use (unet, unet_big, unet_small, unet_custom, resnet)
- `--class-cond`: Train a class-conditioned model
- `--diffusion-steps`: Number of timesteps in diffusion process (default: 1000)
- `--sampling-steps`: Number of steps for sampling (default: 250)
- `--ddim`: Use DDIM sampling instead of DDPM

### Dataset Configuration
- `--dataset`: Dataset to use
- `--data-dir`: Directory containing the dataset
- `--motion-dir`: Directory for motion data (for video datasets)
- `--num-training-data`: Number of images to use for training
- `--seq-len`: Number of frames in each sequence (for video)

### Training Configuration
- `--trainer`: Training loop to use (standard, shared_epsilon)
- `--batch-size`: Batch size per GPU
- `--lr`: Learning rate (default: 0.0001)
- `--epochs`: Number of training epochs
- `--ema_w`: EMA model weight (default: 0.9995)
- `--reg-weight`: Weight for regularization loss (for shared_epsilon trainer)
- `--save-every`: Save model every n epochs (default: 1)
- `--resume`: Path to checkpoint to resume training from

### Sampling Configuration
- `--sampling-only`: No training, just sample images
- `--num-sampled-images`: Number of images to sample
- `--pretrained-ckpt`: Path to pretrained model checkpoint
- `--delete-keys`: Keys to delete from pretrained checkpoint
- `--sample-out-dir`: Directory to save sampled images

### Misc Configuration
- `--save-dir`: Directory to save model checkpoints
- `--local_rank`: Local rank for distributed training
- `--seed`: Random seed (default: 112233)

## Adding New Components

### Adding a new architecture

1. Create a new file in the `architectures/` directory (e.g., `architectures/new_arch.py`)
2. Implement your architecture as a PyTorch module
3. Add your architecture to `architectures/__init__.py`

### Adding a new trainer

1. Create a new file in the `trainers/` directory (e.g., `trainers/new_trainer.py`)
2. Inherit from `BaseTrainer` and implement the `train_one_epoch` method
3. Add your trainer to `trainers/__init__.py`
4. Add any trainer-specific arguments with the `--trainer_` prefix
