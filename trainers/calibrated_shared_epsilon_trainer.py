import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer


class CalibratedSharedEpsilonTrainer(BaseTrainer):
    """
    Diffusion trainer that
      • shares the same ε across 3 consecutive frames
      • applies a confidence-weighted, directional Laplacian
        – low-confidence frames are pulled toward high-confidence ones
        – high-confidence frames receive (almost) no gradient from their neighbours
    """

    # ------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------
    def __init__(self, model, diffusion, args, **kwargs):
        super().__init__(model, diffusion, args)

        # Hyper-parameters (all have sensible defaults)
        self.use_flow_weighting      = kwargs.get("use_flow_weighting", True)
        self.flow_weight_scale       = kwargs.get("flow_weight_scale", 1.0)

        self.use_timestep_weighting  = kwargs.get("use_timestep_weighting", False)
        self.timestep_weight_scale   = kwargs.get("timestep_weight_scale", 1.0)

        self.reg_weight              = getattr(args, "reg_weight", 0.1)
        self.alpha_confidence        = kwargs.get("alpha_confidence", 10.0)   # confidence "temperature"
        self.confidence_clip         = kwargs.get("confidence_clip", (0.1, 10.0))  # more conservative clip range
        self.rho_ema                 = kwargs.get("fidelity_ema_decay", 0.99)
        self.temperature             = kwargs.get("temperature", 1)  # temperature for softmax confidence calculation

        # Linear annealing of the reg-weight
        self.anneal_start_step       = kwargs.get("anneal_start_step", float("inf"))
        self.anneal_end_step         = kwargs.get("anneal_end_step", float("inf"))
        self.anneal_end_weight       = kwargs.get("anneal_end_weight", 0.01)

        self.current_step            = 0
        self.fidelity_stats          = None   # filled lazily (mean/var per frame)

    # ------------------------------------------------------------
    #  EMA helper for residual statistics
    # ------------------------------------------------------------
    def _update_fidelity_stats(self, residuals):
        """residuals: tensor [B, 3]"""
        if self.fidelity_stats is None:
            self.fidelity_stats = {
                "mean": residuals.mean(0),
                "var":  residuals.var(0, unbiased=False) + 1e-8,
            }
            return

        rho = self.rho_ema
        batch_mean = residuals.mean(0)
        batch_var  = residuals.var(0, unbiased=False) + 1e-8

        self.fidelity_stats["mean"].mul_(rho).add_((1 - rho) * batch_mean)
        self.fidelity_stats["var"].mul_(rho).add_((1 - rho) * batch_var)

    # ------------------------------------------------------------
    #  One-epoch training loop
    # ------------------------------------------------------------
    def train_one_epoch(self, dataloader, optimizer, logger, lrs):
        self.model.train()
        loader = (
            tqdm(enumerate(dataloader), total=len(dataloader),
                 desc="Batches", leave=False)
            if self.args.local_rank == 0
            else enumerate(dataloader)
        )

        for step, (video_frames, optical_flow) in loader:
            # ----------------------------------------------------
            # (0)  Shapes & basic sanity
            # ----------------------------------------------------
            B, T, C, H, W = video_frames.shape
            assert T == 3, "trainer expects exactly 3 frames per sample"

            video_frames = 2 * video_frames.to(self.args.device) - 1
            optical_flow = optical_flow.to(self.args.device)

            # ----------------------------------------------------
            # (1)  Sample shared timestep & ε
            # ----------------------------------------------------
            t   = torch.randint(self.diffusion.timesteps, (B,),
                                device=self.args.device)
            eps = torch.randn(B, C, H, W, device=self.args.device)
            eps_exp = eps[:, None].expand(-1, T, -1, -1, -1)   # [B,3,C,H,W]

            # ----------------------------------------------------
            # (2)  Forward diffusion
            # ----------------------------------------------------
            frames_flat = video_frames.view(-1, C, H, W)
            t_flat      = t.repeat_interleave(T)
            eps_flat    = eps_exp.reshape(-1, C, H, W)

            xt_flat, _ = self.diffusion.sample_from_forward_process(
                frames_flat, t_flat, eps=eps_flat
            )

            # ----------------------------------------------------
            # (3)  Network prediction
            # ----------------------------------------------------
            pred_eps_flat = self.model(xt_flat, t_flat, y=None)
            pred_eps = pred_eps_flat.view(B, T, C, H, W)

            # ----------------------------------------------------
            # (4)  Core losses
            # ----------------------------------------------------
            # 4a. basic MSE on ε prediction
            mse_loss = (pred_eps_flat - eps_flat).pow(2).mean()

            # 4b. residuals → confidences
            with torch.no_grad():
                residuals = (pred_eps - eps_exp).pow(2).mean(dim=[2, 3, 4])  # [B,3]
                self._update_fidelity_stats(residuals)

                mu = self.fidelity_stats["mean"]
                std = self.fidelity_stats["var"].sqrt()
                # Calculate relative residuals compared to mu
                relative_residuals = residuals / (mu + 1e-8)
                # Use softmax with temperature to get more concentrated weights
                confidence = torch.softmax(-relative_residuals / self.temperature, dim=1)  # [B,3]

                # directional weights (soft, but always sum to 1 per edge)
                c0, c1, c2 = confidence[:, 0], confidence[:, 1], confidence[:, 2]
                # Add small epsilon to prevent division by zero and ensure weights are well-behaved
                w_0_to_1 = c1 / (c0 + c1 + 1e-6)   # [B]
                w_1_to_0 = 1.0 - w_0_to_1
                w_1_to_2 = c2 / (c1 + c2 + 1e-6)
                w_2_to_1 = 1.0 - w_1_to_2
                # helpers for broadcasting
                w01 = w_0_to_1.view(-1, 1, 1, 1)
                w10 = w_1_to_0.view(-1, 1, 1, 1)
                w12 = w_1_to_2.view(-1, 1, 1, 1)
                w21 = w_2_to_1.view(-1, 1, 1, 1)

            # 4c. confidence-weighted, *asymmetric* Laplacian
            #     (stop-grad on the "target" side of each edge)
            reg_loss = (
                (w01 * (pred_eps[:, 0] - pred_eps[:, 1].detach()).pow(2)).mean()
              + (w10 * (pred_eps[:, 1] - pred_eps[:, 0].detach()).pow(2)).mean()
              + (w12 * (pred_eps[:, 1] - pred_eps[:, 2].detach()).pow(2)).mean()
              + (w21 * (pred_eps[:, 2] - pred_eps[:, 1].detach()).pow(2)).mean()
            )

            # 4d. optional timestep / flow weighting
            timestep_weight = torch.ones_like(t, dtype=torch.float32)
            if self.use_timestep_weighting:
                alpha_bar = self.diffusion.scalars.alpha_bar[t]          # [B]
                timestep_weight = self.timestep_weight_scale * (1.0 - alpha_bar)

            flow_weight = torch.ones_like(t)
            if self.use_flow_weighting:
                fw01 = self.flow_weight_scale / (1.0 + optical_flow[:, 0])
                fw12 = self.flow_weight_scale / (1.0 + optical_flow[:, 1])
                flow_weight = (fw01 + fw12) * 0.5                       # [B]

            reg_loss = reg_loss * (timestep_weight * flow_weight).mean()

            # ----------------------------------------------------
            # (5)  Annealed total loss
            # ----------------------------------------------------
            if self.current_step < self.anneal_start_step:
                cur_reg_w = self.reg_weight
            elif self.current_step >= self.anneal_end_step:
                cur_reg_w = self.anneal_end_weight
            else:
                prog = ((self.current_step - self.anneal_start_step) /
                        (self.anneal_end_step - self.anneal_start_step))
                cur_reg_w = self.reg_weight + prog * (self.anneal_end_weight - self.reg_weight)

            loss = mse_loss + cur_reg_w * reg_loss

            # ----------------------------------------------------
            # (6)  Optimiser step
            # ----------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lrs is not None:
                lrs.step()
            self.current_step += 1

            # ----------------------------------------------------
            # (7)  EMA and logging
            # ----------------------------------------------------
            if self.args.local_rank == 0:
                # EMA update
                new_sd = self.model.state_dict()
                for k, v in self.args.ema_dict.items():
                    self.args.ema_dict[k] = (
                        self.args.ema_w * v + (1 - self.args.ema_w) * new_sd[k]
                    )

                logger.log(
                    {
                        "loss": loss.item(),
                        "mse": mse_loss.item(),
                        "reg": reg_loss.item(),
                        "cur_reg_w": cur_reg_w,
                        "conf_mean": confidence.mean().item(),
                        "pi_0_to_1": w_0_to_1.mean().item(),
                        "pi_1_to_2": w_1_to_2.mean().item(),
                        "step": self.current_step,
                    },
                    display=not step % 100,
                )
