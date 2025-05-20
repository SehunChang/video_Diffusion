import os
from tqdm import tqdm
from def_of import process_sequence_folder
import argparse

def main(args):
    sequences = sorted([
        d for d in os.listdir(args.input_root)
        if os.path.isdir(os.path.join(args.input_root, d))
    ])

    for seq in tqdm(sequences, desc="folder"):
        seq_path = os.path.join(args.input_root, seq)

        save_csv_path = os.path.join(args.output_csv, seq)
        save_plot_path = os.path.join(args.output_plot, seq)
        # save_keyframe_path = os.path.join(args.output_keyframe, seq)

        process_sequence_folder(
            seq_path,
            save_csv_dir=save_csv_path,
            save_plot_dir=save_plot_path,
            # keyframe_dir=save_keyframe_path,
            # threshold=args.threshold
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Motion Analysis and Keyframe Extraction")

    parser.add_argument('--input_root', type=str, required=False, default='../dataset/preprocessed_v2')
    parser.add_argument('--output_csv', type=str, default='./of_scalar')
    parser.add_argument('--output_plot', type=str, default='./of_vis')
    # parser.add_argument('--output_keyframe', type=str, default='./all_keyframes')
    # parser.add_argument('--threshold', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
