import os
import argparse
from cleanfid import fid

def main():
    parser = argparse.ArgumentParser(description='Calculate FID scores for generated images')
    parser.add_argument('--dirs', nargs='+', required=True,
                      help='List of base directories containing generated images')
    args = parser.parse_args()

    for base_path in args.dirs:
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
                
                # Print to console
                print(f"FID score (train) for {gen_dir}: {score_train}")
                
                # Write to log file
                f.write(f"\nResults for {gen_dir}:\n")
                f.write(f"FID score (train): {score_train:.4f}\n")

if __name__ == "__main__":
    main()