#!/bin/bash

# First download pretrained diffusion models from https://drive.google.com/drive/folders/1BMTpNF-FSsGrWGZomcM4OS36CootbLRj?usp=sharing
# You can also find 50,000 pre-sampled synthetic images for each dataset at 
# https://drive.google.com/drive/folders/1KRWie7honV_mwPlmTgH8vrU0izQXm4UT?usp=sharing

# for epoch in 500 400 350; do
sampling_args="--arch unet --sampling-steps 100 --sampling-only --save-dir /home/juhun/projects/video_Diffusion/sampled_images/online"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 --master_port 8103 main.py \
    --dataset hanco --batch-size 256 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt /media/data3/juhun/diffusion+/ckpts/unet_hanco_20250518_150858/checkpoints/checkpoint_epoch_250_ema.pt
# done
