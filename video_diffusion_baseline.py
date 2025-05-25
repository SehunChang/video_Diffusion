# video_diffusion_train.py
import os, glob, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ---------- Utilities ----------
def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))
    return out

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# ---------- Dataset ----------
class MultiCamVideoDataset(Dataset):
    def __init__(self, video_root, flow_root, seq_len=3, transform=None):
        self.video_root = video_root
        self.flow_root = flow_root
        self.seq_len = seq_len
        self.half = seq_len // 2
        self.transform = transform
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for video_id in sorted(os.listdir(self.video_root)):
            video_path = os.path.join(self.video_root, video_id)
            motion_csv_path = os.path.join(self.flow_root, video_id, f"{video_id}_motion.csv")
            if not os.path.exists(motion_csv_path): continue
            for cam in [f"cam{i}" for i in range(8)]:
                cam_path = os.path.join(video_path, cam)
                frame_paths = sorted(glob.glob(os.path.join(cam_path, "*.jpg")))
                if len(frame_paths) < self.seq_len: continue
                for center in range(self.half, len(frame_paths) - self.half):
                    indices = list(range(center - self.half, center + self.half + 1))
                    samples.append({
                        "frame_paths": [frame_paths[i] for i in indices],
                        "flow_csv": motion_csv_path,
                        "flow_indices": list(range(indices[0], indices[-1])),
                        "cam": cam
                    })
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = [self.transform(Image.open(p).convert("RGB")) for p in sample["frame_paths"]]
        image_seq = torch.stack(images).permute(1, 0, 2, 3)  # (3, T, H, W)
        flow_df = pd.read_csv(sample["flow_csv"])
        flow = flow_df[sample["cam"]].iloc[sample["flow_indices"]].values.astype("float32")
        return image_seq, torch.tensor(flow)

# ---------- Denoiser ----------
class SimpleDenoiser(nn.Module):
    def __init__(self, channels=3, base_dim=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, base_dim * 2),
            nn.ReLU()
        )
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, base_dim, 3, 1, 1), nn.ReLU(),
            nn.Conv3d(base_dim, base_dim * 2, 3, 1, 1), nn.ReLU()
        )
        self.middle = nn.Conv3d(base_dim * 2, base_dim * 2, 3, 1, 1)
        self.decoder = nn.Sequential(
            nn.ReLU(), nn.Conv3d(base_dim * 2, base_dim, 3, 1, 1), nn.ReLU(),
            nn.Conv3d(base_dim, channels, 3, 1, 1)
        )

    def forward(self, x, t):
        temb = self.time_mlp(t).view(t.size(0), -1, 1, 1, 1)
        x = self.encoder(x) + temb
        x = self.middle(x)
        x = self.decoder(x)
        return x

# ---------- Diffusion ----------
class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.register_buffer('posterior_variance', betas * (1. - F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)) / (1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise, noise

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def forward(self, x):
        b, device = x.size(0), x.device
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        x_t, noise = self.q_sample(x, t)
        pred = self.model(x_t, t)
        return F.mse_loss(pred, noise)

# ---------- Trainer ----------
class Trainer:
    def __init__(self, model, dataloader, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            print(f"[Epoch {epoch + 1}/{epochs}]")
            for x, _ in tqdm(self.dataloader):
                x = x.to(self.device)
                loss = self.model(x)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                tqdm.write(f"loss: {loss.item():.4f}")

# ---------- Main ----------
def main():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = MultiCamVideoDataset(
        video_root="/WD2/video_Diffusion-main/preprocessed_50k_camfilter_train_",
        flow_root="all_motion_csv",
        seq_len=3,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    device = torch.device("cuda:0")

    denoiser = SimpleDenoiser()
    diffusion = GaussianDiffusion(denoiser)
    trainer = Trainer(diffusion, dataloader, device)
    trainer.train()

if __name__ == '__main__':
    main()
