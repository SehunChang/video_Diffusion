#!/bin/bash

# First download pretrained diffusion models from https://drive.google.com/drive/folders/1BMTpNF-FSsGrWGZomcM4OS36CootbLRj?usp=sharing
# You can also find 50,000 pre-sampled synthetic images for each dataset at 
# https://drive.google.com/drive/folders/1KRWie7honV_mwPlmTgH8vrU0izQXm4UT?usp=sharing

# for epoch in 500 400 350; do
sampling_args="--arch unet_small --sampling-steps 100 --sampling-only --save-dir /media/data3/juhun/diffusion+/ckpts/unet_small_hanco_20250519_141326/gen500"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --dataset hanco --batch-size 256 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt /media/data3/juhun/diffusion+/ckpts/unet_small_hanco_20250519_141326/checkpoints/checkpoint_epoch_500_ema.pt
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