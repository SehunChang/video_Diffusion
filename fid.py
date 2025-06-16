import os
import argparse
import shutil
import random
from cleanfid import fid
from tqdm import tqdm
import glob
import types
import zipfile
import torch
import numpy as np

num = None
def main():
    parser = argparse.ArgumentParser(description='Calculate FID scores for generated images')
    parser.add_argument('--dirs', nargs='+', required=True,
                      help='List of base directories containing generated images')
    parser.add_argument('--epochs', nargs='+', type=int, required=False,
                      help='List of epochs to process (e.g., 350 300). If not provided, process all gen* directories)')
    parser.add_argument('--datasets', type=str, required=True,
                      help='Comma-separated list of datasets to use for FID calculation. Choose from: camfilter_50k_train, train_train, train_val')
    args = parser.parse_args()


    # Convert comma-separated datasets string to list
    # Monkey patch the get_folder_features function to handle num parameter
    original_get_folder_features = fid.get_folder_features

    args.datasets = args.datasets.split(',')
    
    # Define real image paths for each dataset
    real_paths = {
        'camfilter_50k_train': "/media/NAS/USERS/juhun/diffusion+/data/preprocessed_50k_camfilter_train_",
        'train_train': "/media/NAS/USERS/juhun/diffusion+/data/preprocessed_25k_camfilter_train_1", 
        'train_val': "/media/NAS/USERS/juhun/diffusion+/data/preprocessed_25k_camfilter_train_2"
    }

    epochs = args.epochs if args.epochs else None
    for base_path in args.dirs:
        # Get all subdirectories starting with "gen"
        gen_dirs = [d for d in os.listdir(base_path) if d.startswith('gen') and os.path.isdir(os.path.join(base_path, d))]
        if epochs:
            gen_dirs = [d for d in gen_dirs if d[3:].isdigit() and int(d[3:]) in epochs]
        
        print(f"\nProcessing {base_path}")
        print("Found generation directories:", gen_dirs)
        
        # Create log file path
        log_file = os.path.join(base_path, "fid_scores.txt")
        
        # Open log file in append mode
        with open(log_file, "a") as f:
            f.write(f"\nFID Scores for {os.path.basename(base_path)}\n")
            f.write("=" * 50 + "\n")
            
            for gen_dir in gen_dirs:
                full_path = os.path.join(base_path, gen_dir)
                
                print(f"\nCalculating FID for {gen_dir}")
                
                # Calculate FID scores for selected datasets
                for dataset_name in args.datasets:
                    # Set num based on the current dataset
                    if dataset_name == "train_train" or dataset_name == "train_val":
                        num = 25000
                    else:
                        num = 50000

                    def patched_get_folder_features(fdir, model=None, num_workers=12,
                                            shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                                            custom_image_tranform=None):
                        # If num is None, use all images
                        print(f"num: {num}")
                        if num is None:
                            return original_get_folder_features(fdir, model, num_workers, None, shuffle, seed,
                                                            batch_size, device, mode, custom_fn_resize,
                                                            description, verbose, custom_image_tranform)
                        
                        # Get all image files
                        if ".zip" in fdir:
                            files = list(set(zipfile.ZipFile(fdir).namelist()))
                            files = [x for x in files if os.path.splitext(x)[1].lower()[1:] in fid.EXTENSIONS]
                        else:
                            files = sorted([file for ext in fid.EXTENSIONS
                                        for file in glob.glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)])
                        
                        if verbose:
                            print(f"Found {len(files)} images in the folder {fdir}")
                        
                        # Shuffle and take only num images
                        if shuffle:
                            random.seed(seed)
                            random.shuffle(files)
                        files = files[:num]
                        
                        # Get features for selected files
                        np_feats = fid.get_files_features(files, model, num_workers=num_workers,
                                                        batch_size=batch_size, device=device, mode=mode,
                                                        custom_fn_resize=custom_fn_resize,
                                                        custom_image_tranform=custom_image_tranform,
                                                        description=description, fdir=fdir, verbose=verbose)
                        return np_feats

                    # Apply the monkey patch
                    fid.get_folder_features = patched_get_folder_features
                    real_path = real_paths[dataset_name]
                    if not fid.test_stats_exists(dataset_name, mode="clean"):
                        print(f"FID stats for {dataset_name} not found. Calculating and storing them...")
                        fid.make_custom_stats(dataset_name, real_path, mode="clean")

                    score = fid.compute_fid(full_path, 
                                          dataset_name=dataset_name,
                                          mode="clean", 
                                          dataset_split="custom")
                    
                    # Print to console
                    print(f"FID score ({dataset_name}) for {gen_dir}: {score}")
                    
                    # Write to log file
                    f.write(f"\nResults for {gen_dir}:\n")
                    f.write(f"FID score ({dataset_name}): {score:.4f}\n")
                    f.write(f"Number of samples used: {num}\n")

if __name__ == "__main__":
    main()