import os
from cleanfid import fid

base_paths = [
#     "/media/data3/juhun/diffusion+/ckpts/unet_hanco_20250519_005719",
#     "/media/data3/juhun/diffusion+/ckpts/unet_hanco_20250524_170450", 
    "/media/data3/juhun/diffusion+/ckpts/unet_hanco_20250520_093926"
]

for base_path in base_paths:
    # Get all subdirectories starting with "gen"
    gen_dirs = [d for d in os.listdir(base_path) if d.startswith('gen') and os.path.isdir(os.path.join(base_path, d))]
    
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
            
            score_train = fid.compute_fid(full_path, 
                                        dataset_name='camfilter_50k_train',
                                        mode="clean", 
                                        dataset_split="custom")
            
        #     score_val = fid.compute_fid(full_path, 
        #                                dataset_name='camfilter_50k_val',
        #                                mode="clean", 
        #                                dataset_split="custom")
            
            # Print to console
            print(f"FID score (train) for {gen_dir}: {score_train}")
        #     print(f"FID score (val) for {gen_dir}: {score_val}")
            
            # Write to log file
            f.write(f"\nResults for {gen_dir}:\n")
            f.write(f"FID score (train): {score_train:.4f}\n")
        #     f.write(f"FID score (val): {score_val:.4f}\n")