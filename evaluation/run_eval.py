from __future__ import annotations
import argparse, csv, os, torch
from typing import Callable, Dict
from PIL import Image
from torchvision import transforms

import dino_fid as ev_fid
import precall as ev_pr
import aesthetic as ev_aes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# 1) 64×64 grayscale 이미지를 Tensor 로 불러오기
def load_fake_images(folder: str) -> torch.Tensor:
    tfm = transforms.Compose([
        #이미지 사이즈
        transforms.Resize((64, 64)),
        transforms.ToTensor(),           # (1,H,W)  0~1
    ])
    imgs = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, fn)
            imgs.append(tfm(Image.open(path).convert("RGB")))
    if not imgs:
        raise RuntimeError(f"No images found in {folder}")
    return torch.stack(imgs).to(device)   # (N,1,64,64)
# ──────────────────────────────────────────────────────────────


def main(fake_dir: str, real_dir: str, out_csv: str):
    fake_imgs = load_fake_images(fake_dir)

    # 각 모듈의 evaluate()는 dict 를 반환하도록 통일
    metrics: Dict[str, float] = {}
    for module in (ev_fid, ev_pr,ev_aes):
        metrics.update(module.evaluate(fake_imgs, real_dir))

    # CSV 저장
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}"])
    print(f"[INFO] saved ⇒ {out_csv}")
    for k, v in metrics.items():
        print(f"{k:12}: {v:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Run multiple evaluations at once")
    ap.add_argument("--fake_dir", required=True, help="folder with generated images")
    ap.add_argument("--real_dir", required=True, help="folder with real images")
    ap.add_argument("--out_csv", default="results.csv", help="csv output filename")
    args = ap.parse_args()

    main(args.fake_dir, args.real_dir, args.out_csv)