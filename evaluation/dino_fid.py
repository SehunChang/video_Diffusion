import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from scipy.linalg import sqrtm
import hashlib, pathlib, pickle
import numpy as np
from utils import resolve_folder_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _cache_file(cache_dir: str, real_dir: str, n_real: int, model: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(f"{real_dir}_{n_real}_{model}".encode()).hexdigest()
    return os.path.join(cache_dir, f"fid_cache_{h}.npz")


class DINOv2FeatureExtractor:
    def __init__(self, model_name='facebook/dinov2-base', batch_size=64):
        self.processor = AutoImageProcessor.from_pretrained(model_name,use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.batch_size = batch_size

    def extract_from_tensor(self, images):
        if images.ndim == 1:
            images = images.repeat(1,3,1,1)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = images.clamp(0, 1).cpu()
        feats = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size]
            inputs = self.processor(images=list(batch), return_tensors='pt').to(device)
            with torch.no_grad():
                cls = self.model(**inputs).last_hidden_state[:, 0, :]
            feats.append(cls.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def extract_from_dir(self, path,num_samples=None):
        path = resolve_folder_path(path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image_paths = [os.path.join(path, f) for f in os.listdir(path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if num_samples:
            rng = np.random.default_rng(0)
            image_paths = rng.choice(image_paths, size = min(num_samples,len(image_paths)), replace=False)
        feats = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            images = [transform(Image.open(p).convert('RGB')) for p in batch_paths]
            inputs = self.processor(images=images, return_tensors='pt').to(device)
            with torch.no_grad():
                cls = self.model(**inputs).last_hidden_state[:, 0, :]
            feats.append(cls.cpu().numpy())
        return np.concatenate(feats, axis=0)

def compute_mean_and_cov(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_dino_fid(fake_images, real_image_dir,n_fake=None,n_real=None,cache_dir='.fid_cache'):
    extractor = DINOv2FeatureExtractor()

    #real 이미지 feature stats cache코드 추가
    cash_path = _cache_file(cache_dir, real_image_dir, n_real, "dinov2-base")
    if os.path.exists(cash_path):
        data=np.load(cash_path)
        mu_r,sigma_r = data['mu'],data['sigma']
        print(f"[dino_fid] loaded cache stat-> {cash_path}")
    else:
        print("Extracting features from real images...")
        real_feats = extractor.extract_from_dir(real_image_dir, num_samples=n_real)
        mu_r, sigma_r = compute_mean_and_cov(real_feats)

        np.savez(cash_path, mu=mu_r, sigma=sigma_r)
        print(f"[DINO FID] saved stats cache → {cash_path}")
    
    real_feats = extractor.extract_from_dir(real_image_dir)
    mu_r, sigma_r = compute_mean_and_cov(real_feats)

    print("Extracting features from fake images...")
    fake_feats = extractor.extract_from_tensor(fake_images)
    mu_g, sigma_g = compute_mean_and_cov(fake_feats)

    fid = compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    print(f"DINOv2 FID: {fid:.4f}")
    return fid

def evaluate(fake_images, real_image_dir, n_fake=None, n_real=None, cache_dir='.fid_cache') -> dict[str, float]:
    """
    Wrapper function to compute DINOv2 FID and return results in a dictionary format.
    """
    # fake 이미지 개수 제한
    if n_fake is not None and n_fake < len(fake_images):
        fake_images = fake_images[:n_fake]
    fid = compute_dino_fid(fake_images, real_image_dir, n_fake=n_fake, n_real=n_real, cache_dir=cache_dir)
    return {"DINOv2_FID": fid}

if __name__ == "__main__":
    # 생성된 이미지 예시 (64x64 grayscale)
    fake = torch.randn(3, 1, 64, 64).to(device)

    # 실제 이미지 폴더
    real_dir = './temp/cls0'  # 실제 경로로 수정

    compute_dino_fid(fake, real_dir)
