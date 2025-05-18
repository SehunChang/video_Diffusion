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
  --save-dir ./trained_models/
```

### Training with a larger UNet model

```bash
python main.py \
  --arch unet_big \
  --trainer standard \
  --dataset cifar10 \
  --data-dir ./dataset/ \
  --batch-size 64 \
  --epochs 500 \
  --save-dir ./trained_models/
```

### Training with a smaller UNet model

```bash
python main.py \
  --arch unet_small \
  --trainer standard \
  --dataset cifar10 \
  --data-dir ./dataset/ \
  --batch-size 256 \
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
  --save-dir ./trained_models/
```

### Sampling from a trained model

```bash
python main.py \
  --arch unet \
  --dataset cifar10 \
  --sampling-only \
  --num-sampled-images 1000 \
  --pretrained-ckpt ./trained_models/unet_cifar10-epoch_500-timesteps_1000-class_condn_False.pt \
  --sample-out-dir ./sampled_images/
```

## Adding New Components

### Adding a new architecture

1. Create a new file in the `architectures/` directory (e.g., `architectures/new_arch.py`)
2. Implement your architecture as a PyTorch module
3. Add your architecture to `architectures/__init__.py`

### Adding a new trainer

1. Create a new file in the `trainers/` directory (e.g., `trainers/new_trainer.py`)
2. Inherit from `BaseTrainer` and implement the `train_one_epoch` method
3. Add your trainer to `trainers/__init__.py`

## Command Line Arguments

- `--arch`: Architecture to use (unet, unet_big, unet_small, unet_custom, resnet)
- `--trainer`: Training loop to use (e.g., standard, shared_epsilon)
- `--dataset`: Dataset to use
- `--data-dir`: Directory containing the dataset
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--diffusion-steps`: Number of steps in the diffusion process
- `--sampling-steps`: Number of steps for sampling
- `--ddim`: Use DDIM sampling instead of DDPM
- `--class-cond`: Train a class-conditioned model
- `--reg-weight`: Weight for regularization loss (for shared_epsilon trainer)
- `--save-dir`: Directory to save model checkpoints
- `--sample-out-dir`: Directory to save sampled images
