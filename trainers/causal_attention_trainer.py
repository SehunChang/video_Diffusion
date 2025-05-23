import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
import torch.nn.functional as F


class CausalAttentionTrainer(BaseTrainer):
    """Trainer using adjacent-frame temporal attention loss with center-to-side attention guidance and relative position bias."""

    def __init__(self, model, diffusion, args, **kwargs):
        super().__init__(model, diffusion, args)
        self.trainer_args = kwargs

        self.use_flow_weighting = kwargs.get('use_flow_weighting', True)
        self.flow_weight_scale = kwargs.get('flow_weight_scale', 1.0)
        self.attn_supervise_weight = kwargs.get('attn_supervise_weight', 1.0)
        self.use_relative_position = kwargs.get('use_relative_position', True)

    def train_one_epoch(self, dataloader, optimizer, logger, lrs):
        self.model.train()
        data_iter = dataloader
        if self.args.local_rank == 0:
            data_iter = tqdm(enumerate(dataloader), total=len(dataloader), desc='Batches', leave=False)
        else:
            data_iter = enumerate(dataloader)

        for step, (video_frames, optical_flow, labels) in data_iter:
            B, T, C, H, W = video_frames.shape
            assert T == 3, "Expected 3 consecutive frames"

            # Convert to [-1, 1] pixel range and move to device
            video_frames = 2 * video_frames.to(self.args.device) - 1
            optical_flow = optical_flow.to(self.args.device)

            # Sample the same timestep for all frames in a sequence
            t = torch.randint(self.diffusion.timesteps, (B,), dtype=torch.int64).to(self.args.device)

            # Flatten: [B, T, C, H, W] -> [B*T, C, H, W]
            frames_flat = video_frames.view(-1, C, H, W)
            t_flat = t.repeat_interleave(T)

            # Shared noise for all 3 frames
            eps = torch.randn(B, C, H, W, device=self.args.device)
            eps_flat = eps[:, None, ...].expand(-1, T, -1, -1, -1).reshape(-1, C, H, W)

            # Diffusion forward process
            xt_flat, _ = self.diffusion.sample_from_forward_process(frames_flat, t_flat, eps=eps_flat)

            # Temporal label 생성: [0,1,2] 반복 (B times)
            # temporal_index = torch.arange(T, device=self.args.device).repeat(B)
            temporal_index = torch.tensor([1,0,2], device=self.args.device).repeat(B)
            # Model prediction
            pred_eps_flat, extra = self.model(
                xt_flat,
                t_flat,
                y=temporal_index,
                return_attn=False,
                return_rel_pos=self.use_relative_position
            )

            # Reshape back: [B*T, C, H, W] -> [B, T, C, H, W]
            pred_eps = pred_eps_flat.view(B, T, C, H, W)
            target_eps = eps_flat.view(B, T, C, H, W)

            # MSE loss
            mse_loss = F.mse_loss(pred_eps, target_eps)

            # Attention supervision loss
            # attn_weights_1 = extra.get("attn_weights_1")
            # attn_weights_3 = extra.get("attn_weights_3")

            # attn_supervision_1 = torch.tensor(0.0, device=self.args.device)
            # attn_supervision_3 = torch.tensor(0.0, device=self.args.device)

            # if attn_weights_1 is not None and attn_weights_3 is not None:
            #     B_attn, H, Q, K = attn_weights_1.shape
            #     assert Q == 1 and K == 1, f"Expected [B, H, 1, 1], got {attn_weights_1.shape}"

            #     # x1 ← x2 → attention ground-truth = all attention to x2 (index 0)
            #     attn_gt_1 = torch.ones_like(attn_weights_1)  # shape: [B, H, 1, 1]
            #     attn_gt_3 = torch.ones_like(attn_weights_3)

            #     attn_supervision_1 = F.kl_div(attn_weights_1.log(), attn_gt_1, reduction="batchmean")
            #     attn_supervision_3 = F.kl_div(attn_weights_3.log(), attn_gt_3, reduction="batchmean")

            # attn_supervision = (attn_supervision_1 + attn_supervision_3) / 2

            # Relative position bias loss
            # rel_pos_bias = extra.get('rel_pos_bias')
            # if self.use_relative_position and rel_pos_bias is not None:
            #     if rel_pos_bias.dim() == 4:
            #         rel_target = rel_pos_bias[:, :, 0, 0] + rel_pos_bias[:, :, 0, 1]
            #     elif rel_pos_bias.dim() == 3:
            #         rel_target = rel_pos_bias[:, 0, 0] + rel_pos_bias[:, 0, 1]
            #     else:
            #         raise ValueError("Unexpected shape for rel_pos_bias")
            #     rel_loss = (rel_target ** 2).mean()
            # else:
            #     rel_loss = torch.tensor(0.0, device=self.args.device)

            # Flow-weighted regularization loss
            # if self.use_flow_weighting:
            #     flow_weight_0_1 = self.flow_weight_scale / (1.0 + optical_flow[:, 0])
            #     flow_weight_1_2 = self.flow_weight_scale / (1.0 + optical_flow[:, 1])

            #     eps_diff_0_1 = ((pred_eps[:, 0] - pred_eps[:, 1]) ** 2).mean(dim=[1, 2, 3])
            #     eps_diff_1_2 = ((pred_eps[:, 1] - pred_eps[:, 2]) ** 2).mean(dim=[1, 2, 3])

            #     reg_loss = (eps_diff_0_1 * flow_weight_0_1 + eps_diff_1_2 * flow_weight_1_2).mean()
            # else:
            #     reg_loss = torch.tensor(0.0, device=self.args.device)

            # # Total loss
            # reg_weight = getattr(self.args, "reg_weight", 0.1)
            # loss = mse_loss + self.attn_supervise_weight * attn_supervision + reg_weight * reg_loss + rel_loss
            loss = mse_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lrs is not None:
                lrs.step()

            # update ema_dict
            if self.args.local_rank == 0:
                new_dict = self.model.state_dict()
                for k in self.args.ema_dict:
                    self.args.ema_dict[k] = self.args.ema_w * self.args.ema_dict[k] + (1 - self.args.ema_w) * new_dict[k]

                log_dict = {
                    'loss': loss.item(),
                    'mse_loss': mse_loss.item(),
                    # 'attn_supervision': attn_supervision.item(),
                    # 'rel_loss': rel_loss.item(),
                    # 'reg_loss': reg_loss.item(),
                    'flow_weight_scale': self.flow_weight_scale
                }
                logger.log(log_dict, display=not step % 100)
