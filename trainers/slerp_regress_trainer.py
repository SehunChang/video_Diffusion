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
    
    def calculate_direction_vector_fixed(self, eps_first, eps_last, num_frames, omega, device, optical_flow=None):
        """Calculate direction vector using finite differences for better stability, with optional flow-aware spacing."""
        batch_size, channels, height, width = eps_first.shape
        eps_interpolated = torch.zeros(batch_size, num_frames, channels, height, width, device=device)
        d = torch.zeros_like(eps_interpolated)

        if optical_flow is None:
            # Uniform weights
            weights = torch.linspace(0, 1, num_frames, device=device).unsqueeze(0).repeat(batch_size, 1)
        else:
            # Compute flow-aware weights
            flow = optical_flow + 1e-6
            flow_cumsum = torch.cat([torch.zeros(batch_size, 1, device=device), torch.cumsum(flow, dim=1)], dim=1)
            weights = flow_cumsum / flow_cumsum[:, -1:].clamp(min=1e-6)  # [batch_size, num_frames]

        for k in range(num_frames):
            weight = weights[:, k]  # [batch_size]
            sin_omega = torch.sin(omega)
            sin_omega = torch.clamp(sin_omega, 1e-6)
            sin_omega = sin_omega.view(batch_size, 1, 1, 1)
            w = weight.view(batch_size, 1, 1, 1)
            omega_expanded = omega.view(batch_size, 1, 1, 1)
            eps_interpolated[:, k] = (
                torch.sin((1 - w) * omega_expanded) / sin_omega * eps_first +
                torch.sin(w * omega_expanded) / sin_omega * eps_last
            )
            # Calculate direction using finite differences
            if k == 0:
                next_weight = weights[:, k + 1] if k + 1 < num_frames else weight
                next_weight_expanded = next_weight.view(batch_size, 1, 1, 1)
                eps_next = (
                    torch.sin((1 - next_weight_expanded) * omega_expanded) / sin_omega * eps_first +
                    torch.sin(next_weight_expanded * omega_expanded) / sin_omega * eps_last
                )
                d[:, k] = eps_next - eps_interpolated[:, k]
            elif k == num_frames - 1:
                prev_weight = weights[:, k - 1]
                prev_weight_expanded = prev_weight.view(batch_size, 1, 1, 1)
                eps_prev = (
                    torch.sin((1 - prev_weight_expanded) * omega_expanded) / sin_omega * eps_first +
                    torch.sin(prev_weight_expanded * omega_expanded) / sin_omega * eps_last
                )
                d[:, k] = eps_interpolated[:, k] - eps_prev
            else:
                prev_weight = weights[:, k - 1]
                next_weight = weights[:, k + 1]
                prev_weight_expanded = prev_weight.view(batch_size, 1, 1, 1)
                next_weight_expanded = next_weight.view(batch_size, 1, 1, 1)
                eps_prev = (
                    torch.sin((1 - prev_weight_expanded) * omega_expanded) / sin_omega * eps_first +
                    torch.sin(prev_weight_expanded * omega_expanded) / sin_omega * eps_last
                )
                eps_next = (
                    torch.sin((1 - next_weight_expanded) * omega_expanded) / sin_omega * eps_first +
                    torch.sin(next_weight_expanded * omega_expanded) / sin_omega * eps_last
                )
                d[:, k] = (eps_next - eps_prev) / 2.0
        return eps_interpolated, d

    def calculate_align_loss_fixed(self, pred_eps, eps_interpolated, d, num_frames):
        """Calculate align loss with proper scaling and normalization."""
        batch_size = pred_eps.shape[0]
        align_loss = 0
        
        for k in range(num_frames):
            # Get difference from corresponding interpolated epsilon
            diff = pred_eps[:, k] - eps_interpolated[:, k]  # ε̂ₖ - εₖ
            
            # Normalize direction vector properly
            d_flat = d[:, k].reshape(batch_size, -1)
            diff_flat = diff.reshape(batch_size, -1)
            
            # Calculate projection coefficient
            d_norm_sq = torch.sum(d_flat ** 2, dim=1, keepdim=True)
            d_norm_sq = torch.clamp(d_norm_sq, min=1e-8)
            
            proj_coeff = torch.sum(d_flat * diff_flat, dim=1, keepdim=True) / d_norm_sq
            
            # Calculate orthogonal component
            proj_flat = proj_coeff * d_flat
            ortho_flat = diff_flat - proj_flat
            
            # Calculate loss for this frame
            frame_loss = torch.mean(ortho_flat ** 2, dim=1)  # Mean over spatial dimensions
            align_loss = align_loss + frame_loss
        
        # Average over batch and frames
        align_loss = align_loss.mean() / num_frames

        # Apply timestep weighting if enabled
        if self.use_timestep_weighting:
            alpha_bar = self.diffusion.scalars.alpha_bar[t]
            timestep_weight = self.timestep_weight_scale * (1.0 - alpha_bar)
            align_loss = (align_loss * timestep_weight).mean()
        
        return align_loss

    def calculate_align_loss_cosine(self, pred_eps, eps_interpolated, d, num_frames):
        """Alternative approach using cosine similarity to encourage alignment."""
        batch_size = pred_eps.shape[0]
        align_loss = 0
        
        for k in range(num_frames):
            diff = pred_eps[:, k] - eps_interpolated[:, k]
            
            # Flatten for cosine similarity calculation
            diff_flat = diff.reshape(batch_size, -1)
            d_flat = d[:, k].reshape(batch_size, -1)
            
            # Calculate cosine similarity
            diff_norm = torch.norm(diff_flat, dim=1, keepdim=True)
            d_norm = torch.norm(d_flat, dim=1, keepdim=True)
            
            # Avoid division by zero
            diff_norm = torch.clamp(diff_norm, min=1e-8)
            d_norm = torch.clamp(d_norm, min=1e-8)
            
            cos_sim = torch.sum(diff_flat * d_flat, dim=1, keepdim=True) / (diff_norm * d_norm)
            
            # Loss is 1 - |cos_sim| to encourage alignment in either direction
            frame_loss = 1.0 - torch.abs(cos_sim).squeeze()
            align_loss = align_loss + frame_loss
        
        return align_loss.mean() / num_frames

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
            assert num_frames % 2 == 1, "Expected odd number of frames per sample"
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
            
            # Sample two different noise vectors for start and end frames
            eps_first = torch.randn(batch_size, channels, height, width, device=self.args.device)
            eps_last = torch.randn(batch_size, channels, height, width, device=self.args.device)
            
            # Calculate the angle between the two noise vectors
            eps_first_norm = eps_first.view(batch_size, -1)
            eps_last_norm = eps_last.view(batch_size, -1)
            
            # Normalize vectors
            eps_first_norm = eps_first_norm / torch.norm(eps_first_norm, dim=1, keepdim=True)
            eps_last_norm = eps_last_norm / torch.norm(eps_last_norm, dim=1, keepdim=True)
            
            # Calculate dot product
            dot_product = (eps_first_norm * eps_last_norm).sum(dim=1, keepdim=True)
            dot_product = torch.clamp(dot_product, -0.9995, 0.9995)  # Prevent numerical instability
            
            # Calculate angle
            omega = torch.acos(dot_product)
            
            # Create interpolation weights for each frame
            frame_weights = torch.linspace(0, 1, num_frames, device=self.args.device).view(1, -1, 1, 1, 1)
            
            # SLERP interpolation for each frame
            eps_interpolated, d = self.calculate_direction_vector_fixed(eps_first, eps_last, num_frames, omega, self.args.device, optical_flow)
            
            # Reshape for forward diffusion
            eps_flat = eps_interpolated.reshape(-1, channels, height, width)
            
            # Forward diffusion process with interpolated noise
            xt_flat, _ = self.diffusion.sample_from_forward_process(frames_flat, t_flat, eps=eps_flat)
            
            # Model prediction
            pred_eps_flat = self.model(xt_flat, t_flat, y=None)
            
            # Reshape predictions back to sequences
            pred_eps = pred_eps_flat.view(batch_size, num_frames, channels, height, width)
        
            # Original MSE loss
            mse_loss = ((pred_eps_flat - eps_flat) ** 2).mean()
            
            if torch.isnan(mse_loss) or torch.isinf(mse_loss):
                raise ValueError("NaN or Inf encountered in mse_loss")
            
            # Calculate align loss
            align_loss = self.calculate_align_loss_fixed(pred_eps, eps_interpolated, d, num_frames)
            
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
                        'timestep_weight': self.timestep_weight_scale,
                        'timestep_weight_scale': self.timestep_weight_scale
                    })
                logger.log(log_dict, display=not step % 100)