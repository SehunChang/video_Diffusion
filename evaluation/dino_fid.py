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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DINOv2FeatureExtractor:
    def __init__(self, model_name='facebook/dinov2-base', batch_size=64):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
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

    def extract_from_dir(self, path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image_paths = [os.path.join(path, f) for f in os.listdir(path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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

def compute_dino_fid(fake_images, real_image_dir):
    extractor = DINOv2FeatureExtractor()

    print("Extracting features from real images...")
    real_feats = extractor.extract_from_dir(real_image_dir)
    mu_r, sigma_r = compute_mean_and_cov(real_feats)

    print("Extracting features from fake images...")
    fake_feats = extractor.extract_from_tensor(fake_images)
    mu_g, sigma_g = compute_mean_and_cov(fake_feats)

    fid = compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    print(f"DINOv2 FID: {fid:.4f}")
    return fid

if __name__ == "__main__":
    # 생성된 이미지 예시 (64x64 grayscale)
    fake = torch.randn(3, 1, 64, 64).to(device)

    # 실제 이미지 폴더
    real_dir = './temp/cls0'  # 실제 경로로 수정

    compute_dino_fid(fake, real_dir)