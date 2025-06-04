import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer

class SharedX0sTrainer(BaseTrainer):
    """Trainer for diffusion models with shared epsilon across frames."""
    
    def __init__(self, model, diffusion, args, **kwargs):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            diffusion: The diffusion process
            args: Training arguments
            **kwargs: Additional trainer-specific arguments that can be accessed via self.trainer_args
        """
        super().__init__(model, diffusion, args)
        # Store all kwargs as trainer-specific arguments
        self.trainer_args = kwargs
        
        # Set default values for common arguments
        self.use_flow_weighting = kwargs.get('use_flow_weighting', True)
        self.flow_weight_scale = kwargs.get('flow_weight_scale', 1.0)
        # Add timestep weighting configuration
        self.use_timestep_weighting = kwargs.get('use_timestep_weighting', False)
        self.timestep_weight_scale = kwargs.get('timestep_weight_scale', 1.0)
        # Store regularization weight
        self.reg_weight = getattr(args, "reg_weight", 0.1)

        print(f"use_timestep_weighting: {self.use_timestep_weighting}")
        print(f"timestep_weight_scale: {self.timestep_weight_scale}")
        print(f"use_flow_weighting: {self.use_flow_weighting}")
        print(f"flow_weight_scale: {self.flow_weight_scale}")
        print(f"reg_weight: {self.reg_weight}")
        # Print timestep weights for debugging
        if self.use_timestep_weighting and args.local_rank == 0:
            print("\nTimestep weights for all timesteps:")
            print("Timestep\tWeight")
            print("-" * 30)
            
            # Calculate weights for all timesteps
            all_timesteps = torch.arange(self.diffusion.timesteps, device=args.device)
            alpha_bar = self.diffusion.scalars.alpha_bar[all_timesteps]
            weights = self.timestep_weight_scale * (1.0 - alpha_bar)
            
            # Print every 100th timestep to avoid too much output
            for t in range(0, self.diffusion.timesteps, 100):
                print(f"{t}\t{weights[t].item():.4f}")
            # Always print the last timestep
            if self.diffusion.timesteps % 100 != 0:
                print(f"{self.diffusion.timesteps-1}\t{weights[-1].item():.4f}")
            print("-" * 30)
            print(f"Timestep weight scale: {self.timestep_weight_scale}")
            print(f"Min weight: {weights.min().item():.4f}")
            print(f"Max weight: {weights.max().item():.4f}")
            print(f"Mean weight: {weights.mean().item():.4f}")
            print()
    
    def train_one_epoch(self, dataloader, optimizer, logger, lrs):
        """Train for one epoch using the shared epsilon training approach."""
        self.model.train()
        data_iter = dataloader
        if self.args.local_rank == 0:
            data_iter = tqdm(enumerate(dataloader), total=len(dataloader), desc='Batches', leave=False)
        else:
            data_iter = enumerate(dataloader)
            
        for step, (video_frames, optical_flow) in data_iter:
            batch_size, num_frames, channels, height, width = video_frames.shape
            assert num_frames == 3, "Expected 3 consecutive frames per sample"
            assert (video_frames.max().item() <= 1) and (0 <= video_frames.min().item())
            
            # Convert to [-1, 1] pixel range and move to device
            video_frames = 2 * video_frames.to(self.args.device) - 1
            optical_flow = optical_flow.to(self.args.device)
            labels = labels.to(self.args.device) if self.args.class_cond else None
            
            # Sample the same timestep for all frames in a sequence
            t = torch.randint(self.diffusion.timesteps, (batch_size,), dtype=torch.int64).to(self.args.device)
            
            # Reshape video frames for processing
            frames_flat = video_frames.view(-1, channels, height, width)
            t_flat = t.repeat_interleave(num_frames)
            
            eps = torch.randn(batch_size, channels, height, width, device=self.args.device)
            eps_expanded = eps.unsqueeze(1).expand(-1, num_frames, -1, -1, -1)
            eps_flat = eps_expanded.reshape(-1, channels, height, width)
            
            # Forward diffusion process with shared noise
            xt_flat, _ = self.diffusion.sample_from_forward_process(frames_flat, t_flat, eps=eps_flat)
            
            # Model prediction of epsilon
            pred_eps_flat = self.model(xt_flat, t_flat, y=None)
            
            # Convert predicted epsilon to x0 using the diffusion process parameters
            alpha_bar = self.diffusion.scalars.alpha_bar[t_flat]
            alpha = self.diffusion.scalars.alpha[t_flat]
            beta = self.diffusion.scalars.beta[t_flat]
            
            # x0 = (xt - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
            pred_x0_flat = (xt_flat - torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1) * pred_eps_flat) / torch.sqrt(alpha_bar).view(-1, 1, 1, 1)
            # Clamp x0 to valid range
            pred_x0_flat = pred_x0_flat.clamp(-1, 1)
            
            # Reshape predictions back to sequences
            pred_x0 = pred_x0_flat.view(batch_size, num_frames, channels, height, width)
            
            # Get alpha_bar for current timesteps for weighting
            alpha_bar_t = self.diffusion.scalars.alpha_bar[t]
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            
            # Original MSE loss in epsilon space
            mse_loss = ((pred_eps_flat - eps_flat) ** 2).mean()
            
            # Regularization loss in x0 space
            x0_diff_0_1 = ((pred_x0[:, 0] - pred_x0[:, 1]) ** 2).mean(dim=[1, 2, 3])
            x0_diff_1_2 = ((pred_x0[:, 1] - pred_x0[:, 2]) ** 2).mean(dim=[1, 2, 3])
            
            # Weight by sqrt(alpha_bar)
            weighted_x0_diff_0_1 = (x0_diff_0_1 * sqrt_alpha_bar).mean()
            weighted_x0_diff_1_2 = (x0_diff_1_2 * sqrt_alpha_bar).mean()
            
            reg_loss = weighted_x0_diff_0_1 + weighted_x0_diff_1_2
            
            # Final loss
            loss = mse_loss + self.reg_weight * reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if lrs is not None:
                lrs.step()

            # update ema_dict
            if self.args.local_rank == 0:
                new_dict = self.model.state_dict()
                for (k, v) in self.args.ema_dict.items():
                    self.args.ema_dict[k] = (
                        self.args.ema_w * self.args.ema_dict[k] + (1 - self.args.ema_w) * new_dict[k]
                    )
                log_dict = {
                    'loss': loss.item(),
                    'reg_loss': reg_loss.item(),
                    'sqrt_alpha_bar': sqrt_alpha_bar.mean().item(),
                }
                if self.use_flow_weighting:
                    log_dict.update({
                        'flow_weight_0_1': self.flow_weight_scale / (1.0 + optical_flow[:, 0]).mean().item(),
                        'flow_weight_1_2': self.flow_weight_scale / (1.0 + optical_flow[:, 1]).mean().item(),
                        'flow_weight_scale': self.flow_weight_scale
                    })
                if self.use_timestep_weighting:
                    log_dict.update({
                        'timestep_weight': self.timestep_weight_scale * (1.0 - self.diffusion.scalars.alpha_bar[t].item()),
                        'timestep_weight_scale': self.timestep_weight_scale
                    })
                logger.log(log_dict, display=not step % 100)