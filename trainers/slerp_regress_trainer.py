import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer

class slerp_regression_trainer(BaseTrainer):
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
        # Store regularization weight and annealing parameters
        print(args.reg_weight)
        self.reg_weight = getattr(args, "reg_weight", 0.1)
        # Default to effectively disable annealing (start step set to unreachable number)
        self.anneal_start_step = kwargs.get('anneal_start_step', float('inf'))
        self.anneal_end_step = kwargs.get('anneal_end_step', float('inf'))
        self.anneal_start_weight = self.reg_weight
        self.anneal_end_weight = kwargs.get('anneal_end_weight', 0.01)
        self.current_step = 0

        print(f"use_timestep_weighting: {self.use_timestep_weighting}")
        print(f"timestep_weight_scale: {self.timestep_weight_scale}")
        print(f"use_flow_weighting: {self.use_flow_weighting}")
        print(f"flow_weight_scale: {self.flow_weight_scale}")
        print(f"reg_weight: {self.reg_weight}")
        print(f"anneal_start_step: {self.anneal_start_step}")
        print(f"anneal_end_step: {self.anneal_end_step}")
        print(f"anneal_end_weight: {self.anneal_end_weight}")
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
            
            # Model prediction
            pred_eps_flat = self.model(xt_flat, t_flat, y=None)
            
            # Reshape predictions back to sequences
            pred_eps = pred_eps_flat.view(batch_size, num_frames, channels, height, width)
            
            # Original MSE loss
            mse_loss = ((pred_eps_flat - eps_flat) ** 2).mean()
            
            # Reshape predictions and ground truth to sequences
            pred_eps = pred_eps_flat.view(batch_size, num_frames, channels, height, width)
            eps = eps_flat.view(batch_size, num_frames, channels, height, width)
            
            # Calculate direction vector d = (ε₊₁ - ε₋₁) / ||ε₊₁ - ε₋₁||
            eps_diff = eps[:, 2] - eps[:, 0]  # ε₊₁ - ε₋₁
            eps_diff_norm = torch.norm(eps_diff.view(eps_diff.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
            d = eps_diff / (eps_diff_norm + 1e-8)  # Add small epsilon for numerical stability
            
            # Reshape d for broadcasting
            d = d.unsqueeze(1)  # [batch_size, 1, channels, height, width]
            
            # Calculate orthogonal projection for each frame
            # (I - ddᵀ)(ε̂ₖ - ε₋₁)
            align_loss = 0
            for k in range(num_frames):
                # Get difference from first frame
                diff = pred_eps[:, k] - eps[:, 0]  # ε̂ₖ - ε₋₁
                
                # Calculate projection: (I - ddᵀ)(ε̂ₖ - ε₋₁)
                # For each spatial location, this is equivalent to:
                # diff - (d * (d * diff).sum()) where * is element-wise multiplication
                proj = (d * diff).sum(dim=[2, 3, 4], keepdim=True)
                ortho_diff = diff - d * proj
                
                # Add squared norm to loss
                align_loss = align_loss + (ortho_diff ** 2).mean(dim=[1, 2, 3])
            
            # Average over frames
            align_loss = align_loss.mean()
            
            # Apply timestep weighting if enabled
            if self.use_timestep_weighting:
                alpha_bar = self.diffusion.scalars.alpha_bar[t]
                timestep_weight = self.timestep_weight_scale * (1.0 - alpha_bar)
                align_loss = (align_loss * timestep_weight).mean()
            
            # Calculate current regularization weight based on step
            if self.current_step >= self.anneal_start_step:
                if self.current_step >= self.anneal_end_step:
                    current_reg_weight = self.anneal_end_weight
                else:
                    progress = (self.current_step - self.anneal_start_step) / (self.anneal_end_step - self.anneal_start_step)
                    current_reg_weight = self.anneal_start_weight + (self.anneal_end_weight - self.anneal_start_weight) * progress
            else:
                current_reg_weight = self.reg_weight

            # Final loss with annealed regularization weight
            loss = mse_loss + current_reg_weight * align_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if lrs is not None:
                lrs.step()

            # Increment step counter
            self.current_step += 1

            # update ema_dict
            if self.args.local_rank == 0:
                new_dict = self.model.state_dict()
                for (k, v) in self.args.ema_dict.items():
                    self.args.ema_dict[k] = (
                        self.args.ema_w * self.args.ema_dict[k] + (1 - self.args.ema_w) * new_dict[k]
                    )
                log_dict = {
                    'loss': loss.item(),
                    'align_loss': align_loss.item(),
                    'current_reg_weight': current_reg_weight,
                    'current_step': self.current_step,
                }
                if self.use_flow_weighting:
                    log_dict.update({
                        'flow_weight_0_1': d[:, 0, 0, 0, 0].mean().item(),
                        'flow_weight_1_2': d[:, 0, 0, 0, 1].mean().item(),
                        'flow_weight_scale': self.flow_weight_scale
                    })
                if self.use_timestep_weighting:
                    log_dict.update({
                        'timestep_weight': timestep_weight.mean().item(),
                        'timestep_weight_scale': self.timestep_weight_scale
                    })
                logger.log(log_dict, display=not step % 100)