import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base_trainer import BaseTrainer

# -----------------------------------------------------------------------------
#  Randomised SVD – GPU‑friendly, memory‑light  (Halko et al. 2011)
# -----------------------------------------------------------------------------

def _randomised_svd(mat: torch.Tensor, rank: int = 8, oversample: int = 4):
    """Return the first *rank* right‑singular vectors of **mat** ∈ ℝ^{d×k}.
    Works on CPU or GPU; d can be 200 k+.
    """
    d, k_in = mat.shape
    k = min(rank + oversample, k_in)

    # 1. Gaussian projection (k_in × k)
    Omega = torch.randn(k_in, k, device=mat.device, dtype=mat.dtype)
    Y = mat @ Omega                          # (d × k)

    # 2. Orthonormalise → Q
    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (d × k)

    # 3. Project A onto subspace
    B = Q.T @ mat                             # (k × k_in)

    # 4. Small SVD of B
    _, _, Vt = torch.linalg.svd(B, full_matrices=False)

    # 5. Keep first *rank* right‑singular vectors
    V = Vt[:rank].T                           # (k_in × rank)
    return V


class PixelTangentTrainer(BaseTrainer):
    """Shared‑epsilon trainer + pixel‑space Manifold‑Tangent Regularisation (MTR).

    Tangent directions are estimated directly in **input pixel space** from the
    two neighbouring frames by randomised SVD.  No auxiliary encoder needed.
    """

    # ----------------------- initialisation ----------------------------------
    def __init__(self, model, diffusion, args, **kwargs):
        super().__init__(model, diffusion, args)
        self.kwargs = kwargs

        # Original hyper‑parameters ------------------------------------------------
        self.use_flow_weighting     = kwargs.get("use_flow_weighting", True)
        self.flow_weight_scale      = kwargs.get("flow_weight_scale", 1.0)
        self.use_timestep_weighting = kwargs.get("use_timestep_weighting", False)
        self.timestep_weight_scale  = kwargs.get("timestep_weight_scale", 1.0)
        self.reg_weight            = getattr(args, "reg_weight", 0.1)

        # MTR hyper‑parameters -----------------------------------------------------
        self.mtr_weight      = kwargs.get("mtr_weight", 0.05)   # λ
        self.mtr_rank        = kwargs.get("mtr_rank", 6)        # m (≤ neigh.)
        self.mtr_beta        = kwargs.get("mtr_beta", 4.0)      # weight decay
        self.mtr_start_step  = kwargs.get("mtr_start_step", 0)  # warm‑up steps

        if args.local_rank == 0:
            print("[MTR‑pixel] weight =", self.mtr_weight,
                  "rank =", self.mtr_rank, "beta =", self.mtr_beta)

    # ----------------------- tangent helpers ----------------------------------
    @torch.no_grad()
    def _estimate_tangent(self, ref: torch.Tensor, neigh: torch.Tensor):
        """Ref ∈ [B,C,H,W], neigh ∈ [B,k,C,H,W] → V ∈ [B,N,m]   (N = C·H·W)."""
        B, k, C, H, W = neigh.shape
        N = C * H * W
        ref_flat  = ref.view(B, 1, N)             # (B,1,N)
        neigh_flat = neigh.view(B, k, N)          # (B,k,N)

        # Differences D = (neigh − ref)  → (B, N, k)
        D = (neigh_flat - ref_flat).permute(0, 2, 1).contiguous()

        m = min(self.mtr_rank, k)
        V_list = []
        for b in range(B):
            # Transpose D[b] to get (k, N) shape for _randomised_svd
            V_b = _randomised_svd(D[b].T, rank=m)   # (k,N)ᵀ → (N,m)
            V_list.append(V_b)
        V = torch.stack(V_list, 0)                # (B,N,m)
        return V

    def _mtr_loss(self, pred_score: torch.Tensor, V: torch.Tensor):
        """pred_score [B,C,H,W]    V [B,N,m] → scalar."""
        B, C, H, W = pred_score.shape
        N = C * H * W
        s_flat = pred_score.view(B, N)            # (B,N)
        # projections onto each tangent dir
        proj = torch.einsum("bn,bnm->bm", s_flat, V)  # (B,m)
        loss = (proj ** 2).sum(dim=1).mean()      # scalar
        return loss

    # ----------------------- training loop ------------------------------------
    def train_one_epoch(self, dataloader, optimizer, logger, lrs, *, global_step=0):
        self.model.train()
        iterator = (tqdm(enumerate(dataloader), total=len(dataloader), desc="Batches", leave=False)
                    if self.args.local_rank == 0 else enumerate(dataloader))

        for step, (video_frames, optical_flow) in iterator:
            optimizer.zero_grad(set_to_none=True)

            # -------- data prep --------
            B, T, C, H, W = video_frames.shape
            assert T == 3, "Need exactly 3 consecutive frames"
            dev = self.args.device
            video_frames = video_frames.to(dev).mul(2.0).sub_(1.0)   # [−1,1]
            optical_flow  = optical_flow.to(dev)
            ref = video_frames[:, 1]        # (B,C,H,W)
            neigh = torch.cat([video_frames[:, :1], video_frames[:, 2:]], dim=1)     # (B,2,C,H,W)
            V = self._estimate_tangent(ref, neigh)  # (B,N,m)

            # -------- diffusion step --------
            t = torch.randint(self.diffusion.timesteps, (B,), device=dev)
            t_flat = t.repeat_interleave(T)
            noise = torch.randn(B, C, H, W, device=dev)
            noise_exp = noise.unsqueeze(1).expand(-1, T, -1, -1, -1)
            noise_flat = noise_exp.reshape(-1, C, H, W)
            x_flat = video_frames.view(-1, C, H, W)
            xt_flat, _ = self.diffusion.sample_from_forward_process(x_flat, t_flat, eps=noise_flat)

            # -------- model pred --------
            pred_flat = self.model(xt_flat, t_flat, y=None)
            pred = pred_flat.view(B, T, C, H, W)
            mse_loss = F.mse_loss(pred_flat, noise_flat)

            # # temporal reg as before ------------------------------------------
            # d01 = ((pred[:,0]-pred[:,1])**2).mean(dim=[1,2,3])
            # d12 = ((pred[:,1]-pred[:,2])**2).mean(dim=[1,2,3])
            # if self.use_timestep_weighting:
            #     alpha_bar = self.diffusion.scalars.alpha_bar[t]
            #     t_w = self.timestep_weight_scale * (1.0 - alpha_bar)
            # else:
            #     t_w = torch.ones_like(t, dtype=torch.float32)
            # if self.use_flow_weighting:
            #     f01 = self.flow_weight_scale / (1.0 + optical_flow[:,0])
            #     f12 = self.flow_weight_scale / (1.0 + optical_flow[:,1])
            #     reg_loss = (d01*f01*t_w).mean() + (d12*f12*t_w).mean()
            # else:
            #     reg_loss = (d01*t_w).mean() + (d12*t_w).mean()

            # -------- pixel‑space tangent loss ---------------------------------
            if global_step >= self.mtr_start_step:
                # t_norm = t.to(torch.float32) / self.diffusion.timesteps
                # w_mtr = torch.exp(-self.mtr_beta * t_norm).mean()
                # print(w_mtr)
                mtr_loss = self._mtr_loss(pred[:,0], V)
                # print(mtr_loss)
                # tan_loss = w_mtr * self._mtr_loss(pred[:,0], V)
                tan_loss = mtr_loss
            else:
                raise
                tan_loss = torch.tensor(0.0, device=dev)

            # -------- total + step --------------------------------------------
            # loss = mse_loss + self.reg_weight*reg_loss + self.mtr_weight*tan_loss
            loss = mse_loss  + self.mtr_weight*tan_loss
            loss.backward()
            optimizer.step()
            if lrs: lrs.step()

            # -------- EMA + logging -------------------------------------------
            if self.args.local_rank == 0:
                sd = self.model.state_dict()
                for k, v in self.args.ema_dict.items():
                    self.args.ema_dict[k] = self.args.ema_w * v + (1.0 - self.args.ema_w) * sd[k]
                log_dict = {
                    'loss': loss.item(),
                    'mse_loss': mse_loss.item(),
                    # 'reg_loss': reg_loss.item(),
                    'tan_loss': tan_loss.item(),
                    'flow_weight_scale': self.flow_weight_scale
                }
                logger.log(log_dict, display=not step % 100)

            global_step += 1
