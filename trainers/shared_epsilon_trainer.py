import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer

class SharedEpsilonTrainer(BaseTrainer):
    """Trainer for diffusion models with shared epsilon across frames."""
    
    def train_one_epoch(self, dataloader, optimizer, logger, lrs):
        """Train for one epoch using the shared epsilon training approach."""
        self.model.train()
        data_iter = dataloader
        if self.args.local_rank == 0:
            data_iter = tqdm(enumerate(dataloader), total=len(dataloader), desc='Batches', leave=False)
        else:
            data_iter = enumerate(dataloader)
            
        for step, (video_frames, optical_flow, labels) in data_iter:
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
            pred_eps_flat = self.model(xt_flat, t_flat, y=labels.repeat_interleave(num_frames) if labels is not None else None)
            
            # Reshape predictions back to sequences
            pred_eps = pred_eps_flat.view(batch_size, num_frames, channels, height, width)
            
            # Original MSE loss
            mse_loss = ((pred_eps_flat - eps_flat) ** 2).mean()
            
            # Regularization loss
            eps_diff_0_1 = ((pred_eps[:, 0] - pred_eps[:, 1]) ** 2).mean(dim=[1, 2, 3])
            eps_diff_1_2 = ((pred_eps[:, 1] - pred_eps[:, 2]) ** 2).mean(dim=[1, 2, 3])
            
            flow_weight_0_1 = 1.0 / (1.0 + optical_flow[:, 0])
            flow_weight_1_2 = 1.0 / (1.0 + optical_flow[:, 1])
            
            weighted_eps_diff_0_1 = (eps_diff_0_1 * flow_weight_0_1).mean()
            weighted_eps_diff_1_2 = (eps_diff_1_2 * flow_weight_1_2).mean()
            
            reg_loss = weighted_eps_diff_0_1 + weighted_eps_diff_1_2
            
            # Final loss
            reg_weight = getattr(self.args, "reg_weight", 0.1)
            loss = mse_loss + reg_weight * reg_loss
            
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
                logger.log(loss.item(), display=not step % 100) 