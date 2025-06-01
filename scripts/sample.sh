#!/bin/bash

# First download pretrained diffusion models from https://drive.google.com/drive/folders/1BMTpNF-FSsGrWGZomcM4OS36CootbLRj?usp=sharing
# You can also find 50,000 pre-sampled synthetic images for each dataset at 
# https://drive.google.com/drive/folders/1KRWie7honV_mwPlmTgH8vrU0izQXm4UT?usp=sharing

# List of model directories and their architectures and trainers
declare -A models=(
    ["/media/data3/juhun/diffusion+/ckpts/unet_aat_hanco_20250523_012346"]="unet_aat adjacent_attention"
    # ["/media/data3/juhun/diffusion+/ckpts/unet_aat_hanco_20250523_170525"]="unet_aat causal_attention"
)

# List of epochs to sample from
epochs=(500 450 400 350 300 250 200 100)

# Common sampling arguments
base_args="--sampling-steps 100 --sampling-only --dataset hanco --batch-size 256 --num-sampled-images 50000"

# Loop through each model directory
for model_dir in "${!models[@]}"; do
    # Split the value into architecture and trainer
    read -r arch trainer <<< "${models[$model_dir]}"
    
    # Loop through each epoch
    for epoch in "${epochs[@]}"; do
        checkpoint="${model_dir}/checkpoints/checkpoint_epoch_${epoch}_ema.pt"
        # Skip if checkpoint doesn't exist
        if [ ! -f "$checkpoint" ]; then
            echo "Skipping $checkpoint - file not found"
            continue
        fi
        
        save_dir="${model_dir}/gen${epoch}"
        # Skip if save directory already exists
        if [ -d "$save_dir" ]; then
            echo "Skipping $save_dir - directory already exists"
            continue
        fi
        
        echo "Sampling from ${checkpoint} to ${save_dir}"
        
        # Add class-cond flag if architecture is unet_aat
        if [ "$arch" = "unet_aat" ]; then
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
                --nproc_per_node=8 --master_port 8103 main.py \
                --arch $arch --trainer $trainer $base_args \
                --save-dir $save_dir \
                --pretrained-ckpt $checkpoint \
                --class-cond \
                --batch-size 63
        else
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
                --nproc_per_node=8 --master_port 8103 main.py \
                --arch $arch $base_args \
                --save-dir $save_dir \
                --pretrained-ckpt $checkpoint
        fi
    done

    # Calculate FID scores after all generations for this model directory are complete
    echo "Calculating FID scores for ${model_dir}"
    python fid.py --dirs "${model_dir}"
done


# for epoch in 500 400 350; do
# sampling_args="--arch unet_small --sampling-steps 100 --sampling-only --save-dir /media/data3/juhun/diffusion+/ckpts/unet_small_hanco_20250519_141326/gen500"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
#     --dataset hanco --batch-size 256 --num-sampled-images 50000 $sampling_args \
#     --pretrained-ckpt /media/data3/juhun/diffusion+/ckpts/unet_small_hanco_20250519_141326/checkpoints/checkpoint_epoch_500_ema.pt
# done
# sampling_args="--arch unet --sampling-steps 100 --sampling-only --save-dir /media/data3/juhun/diffusion+/ckpts/unet_small_hanco_20250519_141326/gen350"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
#     --dataset hanco --batch-size 256 --num-sampled-images 50000 $sampling_args \
#     --pretrained-ckpt /media/data3/juhun/diffusion+/ckpts/unet_hanco_20250519_005719/checkpoints/checkpoint_epoch_500_ema.pt
# done
# sampling_args="--arch unet --sampling-steps 100 --sampling-only --save-dir /media/data3/juhun/diffusion+/ckpts/unet_hanco_20250519_005719/gen350"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
#     --dataset hanco --batch-size 256 --num-sampled-images 50000 $sampling_args \
#     --pretrained-ckpt /media/data3/juhun/diffusion+/ckpts/unet_hanco_20250519_005719/checkpoints/checkpoint_epoch_300_ema.pt