#!/bin/bash

# Default parameters
FAKE_DIR="/media/NAS/USERS/juhun/diffusion+/ckpt/unet_hanco_slerp_regress_t_weighting_moreweight/gen250"  # Replace with your fake images directory
REAL_DIR="/media/NAS/USERS/juhun/diffusion+/data/preprocessed_50k_camfilter_train_"  # Replace with your real images directory
OUTPUT_CSV="results.csv"
N_FAKE=50000  # Number of fake images to evaluate
N_REAL=50000  # Number of real images to evaluate
EVALS="aes"  # Evaluation metrics to run

# Run the evaluation script
python evaluation/run_eval.py \
    --fake_dir "$FAKE_DIR" \
    --real_dir "$REAL_DIR" \
    --out_csv "$OUTPUT_CSV" \
    --n_fake "$N_FAKE" \
    --n_real "$N_REAL" \
    --evals "$EVALS"

# Print completion message
echo "Evaluation completed. Results saved to $OUTPUT_CSV"
