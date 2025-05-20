import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_flow_magnitude(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)

def extract_keyframes(motion_df, threshold):
    keyframes = []
    grouped = motion_df.groupby('camera')

    for cam, df in grouped:
        for i, row in df.iterrows():
            if row['motion_magnitude'] < threshold:
                keyframes.append({
                    'camera': cam,
                    'frame_index': int(row['frame_index']),
                    'motion_magnitude': row['motion_magnitude']
                })
    return pd.DataFrame(keyframes)

# optical flow 계산 & 저장
def process_sequence_folder(seq_path, save_csv_dir, save_plot_dir):

    os.makedirs(save_csv_dir, exist_ok=True)
    os.makedirs(save_plot_dir, exist_ok=True)

    cam_folders = sorted([d for d in os.listdir(seq_path) if d.startswith("cam")])
    all_results = []

    for cam in cam_folders:
        cam_path = os.path.join(seq_path, cam)
        frames = sorted([f for f in os.listdir(cam_path) if f.endswith(('.jpg', '.png'))])

        if len(frames) < 2:
            print(f"[Warning] {cam_path} has insufficient images (found {len(frames)}), skipping.")
            continue

        magnitudes = []
        for i in range(len(frames) - 1):
            img1 = cv2.imread(os.path.join(cam_path, frames[i]))
            img2 = cv2.imread(os.path.join(cam_path, frames[i + 1]))

            if img1 is None or img2 is None:
                print(f"[Warning] Failed to load image pair: {frames[i]}, {frames[i+1]} in {cam_path}")
                magnitudes.append(np.nan)
                continue

            try:
                mag = compute_flow_magnitude(img1, img2)
                magnitudes.append(mag)
            except Exception as e:
                print(f"[Error] Flow computation failed in {cam_path}: {e}")
                magnitudes.append(np.nan)

        df = pd.DataFrame({
            'frame_index': list(range(len(magnitudes))),
            cam: magnitudes
        })
        all_results.append(df.set_index("frame_index"))

    if not all_results:
        print(f"[Warning] No valid camera data in {seq_path}, skipping sequence.")
        return

    # 원본 저장 (normalize X)
    combined_df = pd.concat(all_results, axis=1).reset_index()
    csv_path_raw = os.path.join(save_csv_dir, f"{os.path.basename(seq_path)}_motion_norm_X.csv")
    combined_df.to_csv(csv_path_raw, index=False)

    # Min-Max 정규화 수행 (normalize O)
    all_results_norm = [df.copy() for df in all_results]
    all_values = []
    for df in all_results_norm:
        col = df.columns[0]
        all_values.extend(df[col].dropna().values.tolist())
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    for df in all_results_norm:
        col = df.columns[0]
        df[col] = (df[col] - global_min) / (global_max - global_min + 1e-6)

    combined_df_norm = pd.concat(all_results_norm, axis=1).reset_index()
    csv_path_norm = os.path.join(save_csv_dir, f"{os.path.basename(seq_path)}_motion_norm_O.csv")
    combined_df_norm.to_csv(csv_path_norm, index=False)

    # 시각화: overlay plot만 저장 (X: 원본, O: 정규화)
    save_overlay_plots(all_results, seq_name=os.path.basename(seq_path), save_plot_dir=save_plot_dir, norm_tag='norm_X')
    save_overlay_plots(all_results_norm, seq_name=os.path.basename(seq_path), save_plot_dir=save_plot_dir, norm_tag='norm_O')

# optical flow 시각화
def save_overlay_plots(all_results, seq_name, save_plot_dir, norm_tag):
    """
    norm_tag: 'norm_X' or 'norm_O'
    """
    os.makedirs(save_plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for df in all_results:
        cam_name = df.columns[0]
        plt.plot(df.index, df[cam_name], label=cam_name)

    plt.title(f"Motion Magnitude - {seq_name} ({norm_tag})")
    plt.xlabel("Frame Index")
    plt.ylabel("Motion Magnitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_plot_dir, f"{seq_name}_overlay_{norm_tag}.png"))
    plt.close()
