import os
import numpy as np
from PIL import Image
import scipy, scipy.io
from easydict import EasyDict
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch
from tqdm import tqdm
import os
import glob
import pandas as pd

def get_metadata(name):
    if name == "hanco":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 1,
                "train_images": 50000,
                "val_images": 0,
                "num_channels": 3,
            }
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return metadata


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_dataset(name, data_dir, metadata, args):
    if name == "hanco":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        if args.seq_len == 1:
            train_set = datasets.ImageFolder(
                data_dir,
                transform=transform_train,
            )
            if args.num_training_data is not None:
                train_set.samples = train_set.samples[:args.num_training_data]
                if hasattr(train_set, 'imgs'):
                    train_set.imgs = train_set.imgs[:args.num_training_data]
        else:
            train_set = MultiCamVideoDataset(
                video_root=data_dir,      # 프레임 이미지들이 들어있는 상위 디렉토리
                flow_root=args.motion_dir,        # motion csv들이 들어있는 상위 디렉토리
                seq_len=args.seq_len,                         # 중심 + 좌우 1프레임씩
                transform=transform_train,
                num_training_data=args.num_training_data,
                use_normalized_flow=args.use_normalized_flow
            )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set


def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d


# Dataset class for handling multi-camera video sequences with optical flow data
class MultiCamVideoDataset(Dataset):
    def __init__(self, video_root, flow_root, seq_len=3, transform=None, num_training_data=None, use_normalized_flow=False):
        self.video_root = video_root  # e.g., "preprocessed_v2"
        self.flow_root = flow_root    # e.g., "all_motion_csv"
        self.seq_len = seq_len        # must be odd (e.g., 3, 5, 7)
        self.half = seq_len // 2
        self.transform = transform
        self.num_training_data = num_training_data
        self.use_normalized_flow = use_normalized_flow

        assert seq_len % 2 == 1, "Sequence length must be odd (e.g., 3, 5, 7)"
        
        # Initialize storage for all frames and flow differences
        self.all_frames = []  # List to store all frame tensors
        self.all_flow_diffs = []  # List to store all flow difference tensors
        self.frame_indices = []  # List to store valid sequence indices

        # Create cache directory name based on input paths
        cache_dir = os.path.join(os.path.dirname(video_root), '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        dataset_name = f"{os.path.basename(video_root)}_{os.path.basename(flow_root)}_{seq_len}"
        if num_training_data is not None:
            dataset_name += f"_{num_training_data}"
        self.cache_path = os.path.join(cache_dir, f"{dataset_name}.pt")
        
        # Try to load cached data first
        if os.path.exists(self.cache_path):
            print(f"Loading cached dataset from {self.cache_path}")
            cached_data = torch.load(self.cache_path)
            self.all_frames = cached_data['frames']
            self.all_flow_diffs = cached_data['flow_diffs'] 
            self.frame_indices = cached_data['indices']
        else:
            # Process all data during initialization
            print("Processing dataset and creating cache...")
            self._collect_samples()
            
            # Limit to num_training_data if specified
            if self.num_training_data is not None:
                self.frame_indices = self.frame_indices[:self.num_training_data]
                
            # Cache the processed data
            torch.save({
                'frames': self.all_frames,
                'flow_diffs': self.all_flow_diffs,
                'indices': self.frame_indices
            }, self.cache_path)
            print(f"Dataset cached to {self.cache_path}")

    def _collect_samples(self):
        # Iterate through each video directory
        for video_id in sorted(os.listdir(self.video_root)):
            video_path = os.path.join(self.video_root, video_id)

            suffix = "motion_norm_O.csv" if self.use_normalized_flow else "motion_norm_X.csv"
            motion_csv_path = os.path.join(self.flow_root, video_id, f"{video_id}_{suffix}")

            # Skip if motion data not available
            if not os.path.exists(motion_csv_path):
                continue

            # Load optical flow values for this video
            motion_df = pd.read_csv(motion_csv_path)
            # Process each camera view (5 cameras total)
            for cam in [f"cam{i}" for i in [1,2,3,4,5]]:
                cam_path = os.path.join(video_path, cam)
                # Get all frame images for this camera
                frame_paths = sorted(glob.glob(os.path.join(cam_path, "*.jpg")))

                # Skip if not enough frames for a sequence
                if len(frame_paths) < self.seq_len:
                    continue

                # Get flow values for this camera view
                flow_values = motion_df[cam].values.astype("float32")
                
                # Pre-process all frames for this camera
                frames = []
                for img_path in frame_paths:
                    img = Image.open(img_path).convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    frames.append(img)
                
                # Stack all frames for this camera
                frames_tensor = torch.stack(frames)  # (T, C, H, W)
                self.all_frames.append(frames_tensor)
                # Store flow values for this camera
                flow_diffs = torch.tensor(flow_values, dtype=torch.float32)  # (T,)
                self.all_flow_diffs.append(flow_diffs)
                
                # Store valid sequence indices
                valid_indices = list(range(self.half, len(frame_paths) - self.half))
                self.frame_indices.extend([(len(self.all_frames)-1, idx) for idx in valid_indices])

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        # Get the camera index and center frame index
        cam_idx, center_idx = self.frame_indices[idx]
        
        # Get the sequence of frames
        frames = self.all_frames[cam_idx]
        start_idx = center_idx - self.half
        end_idx = center_idx + self.half + 1
        image_seq = frames[start_idx:end_idx]  # (seq_len, C, H, W)
        
        # Get the corresponding flow differences
        flow_diffs = self.all_flow_diffs[cam_idx]
        flow_seq = flow_diffs[start_idx:end_idx-1]  # (seq_len-1,)
        
        return image_seq, flow_seq
