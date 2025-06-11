#!/bin/bash

# Default parameters
FAKE_DIR="/media/data3/juhun/diffusion+/ckpts/unet_hanco_20250611_004629/gen250"  # Replace with your fake images directory
REAL_DIR="/media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_"  # Replace with your real images directory
OUTPUT_CSV="results.csv"
N_FAKE=50000  # Number of fake images to evaluate
N_REAL=50000  # Number of real images to evaluate
EVALS="pr,aes"  # Evaluation metrics to run

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
