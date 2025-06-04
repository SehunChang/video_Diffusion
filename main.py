import os
import cv2
import copy
import math
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from easydict import EasyDict
import glob
import json
import logging
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data import get_metadata, get_dataset, fix_legacy_dict
from architectures import get_architecture
from trainers import get_trainer

unsqueeze3x = lambda x: x[..., None, None, None]


class GuassianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t, eps=None):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        if eps is None:
            eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                # print("current_t", current_t)
                # print("current_sub_t", current_sub_t)
                pred_epsilon = model(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
                # print(f"[sample_from_reverse_process] final image stats: min={final.min().item():.3f}, max={final.max().item():.3f}")

        return final


class LossLogger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.loss = []
        self.start_time = time()
        self.ema_loss = None
        self.ema_w = 0.9
        self.current_step = 0
        self.metrics = {}

    def log(self, v, display=False):
        # Handle both single value and dictionary inputs
        if isinstance(v, dict):
            metrics_dict = v
        else:
            metrics_dict = {'loss': v}
            
        # Update metrics
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
            
            # Update EMA for each metric
            ema_key = f"{k}_ema"
            if not hasattr(self, ema_key):
                setattr(self, ema_key, v)
            else:
                current_ema = getattr(self, ema_key)
                setattr(self, ema_key, self.ema_w * current_ema + (1 - self.ema_w) * v)

        self.current_step += 1

        if display:
            elapsed_time = (time() - self.start_time) / 3600
            # Format all EMAs for display
            ema_strings = []
            for k in metrics_dict.keys():
                ema_value = getattr(self, f"{k}_ema")
                ema_strings.append(f"{k} (ema): {ema_value:.3f}")
            
            message = (
                f"Steps: {self.current_step}/{self.max_steps} "
                f"{', '.join(ema_strings)} "
                f"Time elapsed: {elapsed_time:.3f} hr"
            )
            if hasattr(self, 'logger'):
                self.logger.log_info(message)
            else:
                print(message)


def sample_N_images(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
    save_dir=None,
    resume=True,
):
    """Sample N images, saving each as soon as it is generated. Optionally resume from existing images in save_dir."""
    samples, labels, num_samples = [], [], 0
    if dist.is_available() and dist.is_initialized():
        num_processes = dist.get_world_size()
        group = dist.group.WORLD
    else:
        num_processes = 1
        group = None
    os.makedirs(save_dir, exist_ok=True)
    # Find already saved images
    existing_images = sorted(glob.glob(os.path.join(save_dir, 'sample_*.png')))
    start_idx = len(existing_images) if resume else 0
    if args.local_rank == 0:
        print(f"Resuming from {start_idx} images in {save_dir}")
    total_to_sample = N - start_idx
    idx = start_idx
    with tqdm(total=math.ceil(total_to_sample / (args.batch_size * num_processes))) as pbar:
        while idx < N:
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, image_size, image_size)
                    .float()
                    .to(args.device)
                )
            if args.class_cond:
                # For unet_aat, always use label 1 during sampling
                if args.arch == "unet_aat":

                    if args.trainer == "adjacent_attention":
                        y = torch.ones(len(xT), dtype=torch.int64).to(args.device)
                    elif args.trainer == "causal_attention":
                        y = torch.zeros(len(xT), dtype=torch.int64).to(args.device)
                else:
                    y = torch.randint(num_classes, (len(xT),), dtype=torch.int64).to(args.device)
            else:
                y = None
            gen_images = diffusion.sample_from_reverse_process(
                model, xT, sampling_steps, {"y": y}, args.ddim
            )
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            if args.class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                labels_batch = torch.cat(labels_list).detach().cpu().numpy()
            else:
                labels_batch = None
            if group is not None:
                dist.all_gather(samples_list, gen_images, group)
            else:
                samples_list[0] = gen_images
            batch_images = torch.cat(samples_list).detach().cpu().numpy()
            batch_images = (127.5 * (batch_images + 1)).astype(np.uint8)
            # Save each image
            for i in range(len(batch_images)):
                if idx >= N:
                    break
                img = batch_images[i].transpose(1, 2, 0)  # CHW to HWC
                img_path = os.path.join(save_dir, f"sample_{idx:06d}.png")
                cv2.imwrite(img_path, img[:, :, ::-1])
                idx += 1
            pbar.update(1)
    if args.class_cond:
        return None, None  # Labels are saved per image
    else:
        return None, None

class TrainingLogger:
    """Handles logging and directory organization for training."""
    
    def __init__(self, args):
        """
        Initialize the logger.
        
        Args:
            args: Training arguments
        """
        self.args = args
        # Skip directory setup if we're only sampling
        if not args.sampling_only:
            self.setup_directories()
            self.setup_logging()
        else:
            # Just ensure the save_dir exists for sampling
            os.makedirs(args.save_dir, exist_ok=True)
        
    def setup_directories(self):
        """Create necessary directories for training."""
        if self.args.resume:
            # Use the existing run directory from the checkpoint
            self.run_dir = get_base_dir_from_checkpoint(self.args.resume)
            self.log_info(f"Using existing run directory: {self.run_dir}")
        else:
            # Create new timestamp-based run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.args.arch}_{self.args.dataset}_{timestamp}"
            self.run_dir = os.path.join(self.args.save_dir, run_name)
        
        # Create subdirectories
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.sample_dir = os.path.join(self.run_dir, "samples")
        self.log_dir = os.path.join(self.run_dir, "logs")
        
        # Create directories if they don't exist
        for dir_path in [self.checkpoint_dir, self.sample_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def setup_logging(self):
        """Setup logging configuration."""
        if self.args.local_rank == 0:
            # Setup file handler
            log_file = os.path.join(self.log_dir, "training.log")
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            # Save args to JSON
            args_dict = vars(self.args)
            with open(os.path.join(self.log_dir, "args.json"), 'w') as f:
                json.dump(args_dict, f, indent=4)
                
    def log_info(self, message):
        """Log info message."""
        if self.args.local_rank == 0:
            logging.info(message)
            
    def log_metrics(self, metrics_dict, step):
        """Log metrics to file."""
        if self.args.local_rank == 0:
            metrics_file = os.path.join(self.log_dir, "metrics.jsonl")
            metrics_dict['step'] = step
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
                
    def get_checkpoint_path(self, epoch):
        """Get path for saving checkpoint."""
        return os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )
        
    def get_ema_checkpoint_path(self, epoch):
        """Get path for saving EMA checkpoint."""
        return os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}_ema.pt"
        )


def get_base_dir_from_checkpoint(checkpoint_path):
    """Extract the base directory from a checkpoint path.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Base directory containing the checkpoint (parent of checkpoints directory)
    """
    # Remove the filename and 'checkpoints' directory
    base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    return base_dir


def main():
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str, help="Neural network architecture: unet, unet_big, unet_small, unet_custom, or resnet")
    parser.add_argument(
        "--class-cond",
        action="store_true",
        default=False,
        help="train class-conditioned diffusion model",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=250,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
    )
    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-dir", type=str, default="./dataset/")
    parser.add_argument("--motion-dir", type=str, default="")
    parser.add_argument("--num-training-data", type=int, default=None, help="Number of images to use for training (from the dataset folder)")

    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    parser.add_argument("--reg-weight", type=float, default=0.1, 
                      help="Weight for the regularization loss between consecutive frames")
    # sampling/finetuning
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=50000,
        help="Number of images required to sample from the model",
    )
    parser.add_argument("--seq-len", type=int, default=1, help="Number of frames in each sequence")
    parser.add_argument("--use_normalized_flow", action="store_true", help="Use normalized optical flow CSV")

    # Add argument to accept any trainer-specific arguments
    parser.add_argument('--trainer_*', action='append', nargs='+', help='Trainer-specific arguments')

    # trainer selection
    parser.add_argument(
        "--trainer", 
        type=str, 
        default="standard",
        help="Trainer to use (standard, shared_epsilon)"
    )

    # misc
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--sample-out-dir", type=str, default="./sampled_images/online", help="Directory to save sampled images (overrides --save-dir for sampling)")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)
    parser.add_argument("--save-every", type=int, default=1, help="Save model every n epochs")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")

    # setup
    args, unknown = parser.parse_known_args()
    
    # Process unknown arguments that start with trainer_
    trainer_kwargs = {}
    for arg in unknown:
        if arg.startswith('--trainer_'):
            prefix, *rest = arg.split('--trainer_')
            key = rest[0]  # Get the part after --trainer_
            value = True  # Default to True for flags
            if '=' in key:
                key, value = key.split('=')
                # Try to convert value to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
            else:
            # If no value is provided, raise an error since we need a value for trainer-specific arguments
                raise ValueError(f"No value provided for trainer argument: {key}")
            print(f"key: {key}, value: {value}")
            trainer_kwargs[key] = value
    
    metadata = get_metadata(args.dataset)
    torch.backends.cudnn.benchmark = True
    args.device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    # Initialize logger
    logger = TrainingLogger(args)
    # Only log the args if not in sampling mode
    if not args.sampling_only and args.local_rank == 0:
        logger.log_info(f"Starting training with args: {args}")
        if trainer_kwargs:
            logger.log_info(f"Trainer-specific args: {trainer_kwargs}")

    # Create model and diffusion process
    model = get_architecture(
        args.arch,
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes if args.class_cond else None,
    ).to(args.device)
    
    logger.log_info("We are assuming that model input/output pixel range is [-1, 1]. Please adhere to it.")
    diffusion = GuassianDiffusion(args.diffusion_steps, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # load pre-trained model or resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.log_info(f"Resuming from checkpoint: {args.resume}")
        # Get the base directory from the checkpoint path
        base_dir = get_base_dir_from_checkpoint(args.resume)
        logger.log_info(f"Base directory: {base_dir}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        
        # Load model state
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New checkpoint format with training state
            state_dict = checkpoint['model_state_dict']
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'ema_dict' in checkpoint:
                args.ema_dict = checkpoint['ema_dict']
            logger.log_info(f"Resuming from epoch {start_epoch}")
        else:
            # Legacy format - just model state
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict)
            logger.log_info("Loaded legacy checkpoint format")
    elif args.pretrained_ckpt:
        logger.log_info(f"Loading pretrained model from {args.pretrained_ckpt}")
        checkpoint = torch.load(args.pretrained_ckpt, map_location=args.device)
        
        # Handle both new and legacy checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            d = checkpoint['model_state_dict']
        else:
            d = fix_legacy_dict(checkpoint)
            
        dm = model.state_dict()
        if args.delete_keys:
            for k in args.delete_keys:
                logger.log_info(
                    f"Deleting key {k} because its shape in ckpt ({d[k].shape}) doesn't match "
                    + f"with shape in model ({dm[k].shape})"
                )
                del d[k]
        model.load_state_dict(d, strict=False)
        logger.log_info(f"Mismatched keys in ckpt and model: {set(d.keys()) ^ set(dm.keys())}")

    # distributed training
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        logger.log_info(f"Using distributed training on {ngpus} gpus.")
        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # sampling
    if args.sampling_only:
        args.seq_len = 1
        model.seq_len = args.seq_len
        # print(f"[main.py] Sampling mode activated. Using seq_len = {model.seq_len}")
        sample_N_images(
            args.num_sampled_images,
            model,
            diffusion,
            None,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes,
            args,
            save_dir=args.save_dir,  # Use save_dir directly
            resume=True,
        )
        return

    # Load dataset
    train_set = get_dataset(args.dataset, args.data_dir, metadata, args)
    sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    logger.log_info(
        f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(train_set)}"
    )
    loss_logger = LossLogger(len(train_loader) * args.epochs)
    loss_logger.logger = logger  # Connect loss_logger with TrainingLogger

    # ema model
    args.ema_dict = copy.deepcopy(model.state_dict())
    
    # Initialize the trainer
    trainer_class = get_trainer(args.trainer, args)
    
    # # Get all arguments that start with 'trainer_'
    # trainer_kwargs = {k[8:]: v for k, v in vars(args).items() if k.startswith('trainer_')}
    
    trainer = trainer_class(model, diffusion, args, **trainer_kwargs)
    
    # Start training
    epoch_iter = range(start_epoch, args.epochs)
    if args.local_rank == 0:
        epoch_iter = tqdm(epoch_iter, desc='Epochs', total=args.epochs - start_epoch)
    for epoch in epoch_iter:
        if sampler is not None:
            sampler.set_epoch(epoch)
        trainer.train_one_epoch(train_loader, optimizer, loss_logger, None)
        
        # Sample images periodically
        if not epoch % 10:
            sampled_images, _ = sample_N_images(
                32,
                model,
                diffusion,
                None,
                args.sampling_steps,
                args.batch_size,
                metadata.num_channels,
                metadata.image_size,
                metadata.num_classes,
                args,
                save_dir=logger.sample_dir,
                resume=False
            )

        # Save checkpoints
        if args.local_rank == 0 and (epoch + 1) % args.save_every == 0:
            # Save regular checkpoint
            checkpoint_path = logger.get_checkpoint_path(epoch + 1)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_dict': args.ema_dict,
            }, checkpoint_path)
            logger.log_info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save EMA checkpoint
            ema_checkpoint_path = logger.get_ema_checkpoint_path(epoch + 1)
            torch.save(args.ema_dict, ema_checkpoint_path)
            logger.log_info(f"Saved EMA checkpoint to {ema_checkpoint_path}")
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'loss': loss_logger.ema_loss if loss_logger.ema_loss is not None else float('inf')
            }
            logger.log_metrics(metrics, epoch + 1)


if __name__ == "__main__":
    main()
