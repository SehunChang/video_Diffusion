from __future__ import annotations
import os, torch, torch.nn.functional as F, numpy as np
from PIL import Image
from torchvision import transforms
from typing import Dict
from utils import resolve_folder_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AestheticPredictor:
    def __init__(
        self,
        clip_arch: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        #https://github.com/christophschuhmann/improved-aesthetic-predictor 여기서 다운로드해서 저장해둬야 함함
        linear_weights: str = "sac+logos+ava1-l14-linearMSE.pth",
        batch_size: int = 64,
    ):
        import open_clip  # pip install open_clip_torch
        self.batch_size = batch_size

        # 1. CLIP 비전 백본
        self.clip, _, clip_preprocess = open_clip.create_model_and_transforms(
            clip_arch, device=device, pretrained=clip_pretrained
        )
        self.clip.eval()

        # 2. aesthetic 선형 회귀 헤드
        if not os.path.isfile(linear_weights):
            raise FileNotFoundError(
                f"Linear weights '{linear_weights}' not found.\n"
                "다운로드: https://github.com/christophschuhmann/improved-aesthetic-predictor"
            )
        self.head = torch.nn.Linear(self.clip.visual.output_dim, 1).to(device)
        self.head.load_state_dict(torch.load(linear_weights, map_location=device))
        self.head.eval()

        # 3. PIL → Tensor 전처리 (repo preprocess 그대로)
        self.pil_to_tensor = clip_preprocess  # 224 resize + CLIP 정규화

    # ───────────────────────────────────────────────────────
    def _predict_batches(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """img_tensor: (N,3,224,224) CLIP-정규화 완료 상태"""
        scores = []
        for i in range(0, len(img_tensor), self.batch_size):
            batch = img_tensor[i : i + self.batch_size]
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                feats = self.clip.encode_image(batch)
                s = self.head(feats).squeeze(1)
            scores.append(s)
        return torch.cat(scores, 0)

    # ───────────────────────────────────────────────────────
    def score_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N,C,H,W) 0‒1,  C ∈ {1,3}
        return: (N,)  텐서
        """
        if imgs.shape[1] == 1:  # grayscale → RGB
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = F.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
        imgs = imgs.clamp(0, 1)

        # CLIP 정규화
        norm = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        imgs = norm(imgs)
        return self._predict_batches(imgs.to(device))

    # ───────────────────────────────────────────────────────
    def score_folder(self, folder: str) -> torch.Tensor:
        paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        if not paths:
            raise RuntimeError(f"No images found in {folder}")

        pil_imgs = [Image.open(p).convert("RGB") for p in paths]
        tensor_imgs = torch.stack([self.pil_to_tensor(img) for img in pil_imgs]).to(
            device
        )
        return self._predict_batches(tensor_imgs)


# ═══════════════════════════════════════════════════════════
def evaluate(fake_images: torch.Tensor, real_dir: str,n_fake=None,n_real=None) -> Dict[str, float]:
    """
    fake_images : (N,1/3,64,64) 0‒1
    real_dir    : 폴더 경로 (하위 class 폴더여도 OK)
    returns     : {"Aesthetic_fake_mean":…, "Aesthetic_real_mean":…, "Aesthetic_gap":…}
    """
    # fake 이미지 개수 제한
    if n_fake is not None and n_fake < len(fake_images):
        fake_images = fake_images[:n_fake]
    pred = AestheticPredictor()
    fake_scores = pred.score_tensor(fake_images)
    real_scores = pred.score_folder(real_dir,num_samples=n_real)

    fake_mean = fake_scores.mean().item()
    real_mean = real_scores.mean().item()
    return {
        "Aesthetic_fake_mean": fake_mean,
        "Aesthetic_real_mean": real_mean,
        "Aesthetic_gap": real_mean - fake_mean,
    }