#!/bin/bash

# First download pretrained diffusion models from https://drive.google.com/drive/folders/1BMTpNF-FSsGrWGZomcM4OS36CootbLRj?usp=sharing
# You can also find 50,000 pre-sampled synthetic images for each dataset at 
# https://drive.google.com/drive/folders/1KRWie7honV_mwPlmTgH8vrU0izQXm4UT?usp=sharing

for epoch in 500 400 350; do
    sampling_args="--arch UNet --sampling-steps 250 --sampling-only --save-dir ./sampled_images/epoch_${epoch}/"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8102 main.py \
        --arch UNet --dataset hanco --batch-size 256 --num-sampled-images 50000 $sampling_args \
        --pretrained-ckpt /media/NAS/USERS/juhun/diffusion+/minimal-diffusion/trained_models/UNet_hanco-epoch_${epoch}-timesteps_1000-class_condn_False_ema_0.9995.pt
done
