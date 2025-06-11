#!/bin/bash

# Change data-dir to refer to the path of training dataset on your machine
# Following datasets needs to be manually downloaded before training: melanoma, afhq, celeba, cars, flowers, gtsrb.
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
#     --arch UNet --dataset mnist --class-cond --epochs 100 --batch-size 256 --sampling-steps 100

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8102  main.py \
#     --arch UNet --dataset mnist_m --class-cond --epochs 250 --batch-size 256 --sampling-steps 100 \
#     --data-dir ~/datasets/all_mnist/mnist_m/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8104 main.py \
#     --arch UNet --dataset melanoma --class-cond --epochs 250 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/medical/melanoma/org_balanced/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105  main.py \
#     --arch UNet --dataset cifar10 --class-cond --epochs 500 --batch-size 256 --sampling-steps 100

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8106 main.py \
#     --arch UNet --dataset afhq --class-cond --epochs 500 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/misc/afhq256/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8107 main.py \
#     --arch UNet --dataset celeba --class-cond --epochs 100 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/misc/celebA_male_smile_64_balanced/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8108 main.py \
#     --arch UNet --dataset cars --class-cond --epochs 500 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/misc/stanford_cars/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8109 main.py \
#     --arch UNet --dataset flowers --class-cond --epochs 1000 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/misc/oxford_102_flowers/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8110 main.py \
#     --arch UNet --dataset gtsrb --class-cond --epochs 500 --batch-size 256 --sampling-steps 100 \
#     --data-dir ~/datasets/misc/gtsrb/GTSRB/Final_Training/Images/




#!/bin/bash

# Change data-dir to refer to the path of training dataset on your machine
# Following datasets needs to be manually downloaded before training: melanoma, afhq, celeba, cars, flowers, gtsrb.
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
#     --arch UNet --dataset mnist --class-cond --epochs 100 --batch-size 256 --sampling-steps 100

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8102  main.py \
#     --arch UNet --dataset mnist_m --class-cond --epochs 250 --batch-size 256 --sampling-steps 100 \
#     --data-dir ~/datasets/all_mnist/mnist_m/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8104 main.py \
#     --arch UNet --dataset melanoma --class-cond --epochs 250 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/medical/melanoma/org_balanced/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105  main.py \
#     --arch UNet --dataset cifar10 --class-cond --epochs 500 --batch-size 256 --sampling-steps 100

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8106 main.py \
#     --arch UNet --dataset afhq --class-cond --epochs 500 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/misc/afhq256/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8107 main.py \
#     --arch UNet --dataset celeba --class-cond --epochs 100 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/misc/celebA_male_smile_64_balanced/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8108 main.py \
#     --arch UNet --dataset cars --class-cond --epochs 500 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/misc/stanford_cars/train/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8109 main.py \
#     --arch UNet --dataset flowers --class-cond --epochs 1000 --batch-size 128 --sampling-steps 50 \
#     --data-dir ~/datasets/misc/oxford_102_flowers/

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8110 main.py \
#     --arch UNet --dataset gtsrb --class-cond --epochs 500 --batch-size 256 --sampling-steps 100 \
#     --data-dir ~/datasets/misc/gtsrb/GTSRB/Final_Training/Images/

# Wait for all GPU processes to complete
# while nvidia-smi | grep "python" > /dev/null; do
#     echo "Waiting for GPU processes to complete..."
#     sleep 60  # Check every minute
# done

# echo "GPUs are free, starting training..."

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8105 main.py \
    --arch unet \
    --dataset hanco \
    --trainer slerp_regress \
    --epochs 500 \
    --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
    --batch-size 128 \
    --sampling-steps 100 \
    --save-every 50 \
    --num-training-data 25000 \
    --seq-len 3 \
    --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
    --save-dir /media/data3/juhun/diffusion+/ckpts \
    --use_normalized_flow False \
    --trainer_anneal_start_step=1000 \
    --trainer_anneal_end_step=25000 \
    --trainer_anneal_end_weight=0.1 \
    --reg-weight=0.001 \
    --use_timestep_weighting True

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8105 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer calibrated_shared_epsilon \
#     --epochs 250 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --use_normalized_flow False \
#     --trainer_temperature=0.001 \
#     --trainer_use_flow_weighting=false \
#     --trainer_anneal_start_step=8000 \
#     --trainer_anneal_end_step=22000 \
#     --trainer_reg_weight=0.01 \
#     --trainer_anneal_end_weight=0.001 \
#     --reg_weight=0.01

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8105 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer shared_epsilon \
#     --epochs 500 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --use_normalized_flow False \
#     --trainer_anneal_start_step=22000 \
#     --trainer_anneal_end_step=70000 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8105 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer shared_epsilon \
#     --epochs 500 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --use_normalized_flow False \
#     --trainer_anneal_start_step=22000 \
#     --trainer_anneal_end_step=70000 \
#     --trainer_use_flow_weighting=false

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer pixel_tangent \
#     --epochs 250 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --trainer_use_flow_weighting=false \
    # --trainer_mtr_start_step=1000 \
    
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
#     --arch unet_aat \
#     --dataset hanco \
#     --class-cond \
#     --trainer adjacent_attention \
#     --epochs 250 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --trainer_epsilon_weight=0.1


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8101 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer shared_epsilon \
#     --epochs 500 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --trainer_use_flow_weighting=false \
#     --trainer_use_timestep_weighting=true \
#     --trainer_use_flow_weighting=false \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
#     --arch unet_small \
#     --dataset hanco \
#     --trainer shared_epsilon \
#     --use_normalized_flow \
#     --epochs 500 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --trainer_use_flow_weighting=false \



# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer standard \
#     --epochs 500 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 1 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
    # --resume /media/data3/juhun/diffusion+/ckpts/unet_hanco_20250520_093926/checkpoints/checkpoint_epoch_50.pt


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8101 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer shared_x0s \
#     --epochs 250 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --trainer_use_flow_weighting=False \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8101 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer shared_x0s \
#     --epochs 250 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 128 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \
#     --trainer_use_flow_weighting=False \
#     --trainer_reg_weight=0.01

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8101 main.py \
#     --arch unet \
#     --dataset hanco \
#     --trainer shared_epsilon \
#     --epochs 500 \
#     --data-dir /media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_train_ \
#     --batch-size 96 \
#     --sampling-steps 100 \
#     --save-every 50 \
#     --num-training-data 25000 \
#     --seq-len 3 \
#     --motion-dir /media/data3/juhun/diffusion+/data/all_motion_csv \
#     --save-dir /media/data3/juhun/diffusion+/ckpts \