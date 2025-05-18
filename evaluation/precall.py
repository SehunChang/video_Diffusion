"""
VGG‑16 기반 Precision / Recall v2 평가
fake_images : Tensor  (N,1,64,64)   0~1 스케일
real_image_dir : 원본 이미지 폴더
return : dict  {"Precision": p, "Recall": r}
"""
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import os, numpy as np, torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from utils import resolve_folder_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 질문에 주신 두 번째 코드의 함수 그대로 ---------- #
def preprocess_images(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(images.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(images.device)
    return (images - mean) / std

def extract_features(images, model, batch_size=50):
    model.eval()
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            x = model.features(batch)
            x = x.view(x.size(0), -1)
            x = model.classifier[:4](x)
        feats.append(x.cpu().numpy())
    return np.concatenate(feats, axis=0)

def extract_features_from_folder(path, model, num_samples = None, batch_size=50):
    #path = resolve_folder_path(path)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    ds = ImageFolder(path, transform=tfm)
    if num_samples is not None:
        ds.samples = ds.samples[:num_samples]

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats = []
    model.eval()
    for batch, _ in tqdm(loader, desc="VGG feats (real)"):
        batch = batch.to(device)
        with torch.no_grad():
            x = model.features(batch)
            x = x.view(x.size(0), -1)
            x = model.classifier[:4](x)
        feats.append(x.cpu().numpy())
    return np.concatenate(feats, axis=0)

def compute_distances(A, B):
    A2 = np.sum(A**2, axis=1, keepdims=True)
    B2 = np.sum(B**2, axis=1, keepdims=True)
    return np.sqrt(A2 - 2*A@B.T + B2.T + 1e-6)

def compute_radii(feats, k=3):
    D = compute_distances(feats, feats)
    safe_k = min(k+1, D.shape[1]-1)
    return np.partition(D, safe_k, axis=1)[:, safe_k]

def compute_precision_recall(real_feats, fake_feats, real_r, fake_r):
    dist_rf = compute_distances(real_feats, fake_feats)
    dist_fr = dist_rf.T
    precision = np.mean(np.any(dist_fr < real_r[None, :], axis=1))
    recall    = np.mean(np.any(dist_rf < fake_r[None, :], axis=1))
    return precision, recall
# -------------------------------------------------------------- #

def evaluate(fake_images, real_image_dir,n_fake=None,n_real=None) -> dict[str, float]:
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).to(device)
    # fake 이미지 개수 제한
    if n_fake is not None and n_fake < len(fake_images):
        fake_images = fake_images[:n_fake]
    fake_proc = preprocess_images(fake_images.to(device))
    fake_feats = extract_features(fake_proc, model)

    real_feats = extract_features_from_folder(real_image_dir, model,num_samples=n_real)
    real_r, fake_r = compute_radii(real_feats), compute_radii(fake_feats)
    p, r = compute_precision_recall(real_feats, fake_feats, real_r, fake_r)
    return {"Precision": float(p), "Recall": float(r)}
