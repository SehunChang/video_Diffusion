from cleanfid import fid
# fid.make_custom_stats('camfilter_50k_val', "/media/data3/juhun/diffusion+/data/preprocessed_50k_camfilter_val_" , mode="clean")
paths = [
        # "/media/data3/juhun/diffusion+/ckpts/unet_hanco_20250518_150858/gen_250/online",
        #  "/media/data3/juhun/diffusion+/ckpts/unet_hanco_20250519_005719/gen250",
        #  "/media/data3/juhun/diffusion+/ckpts/unet_hanco_20250519_005719/gen500",
         "/media/data3/juhun/diffusion+/ckpts/unet_small_hanco_20250519_141326/gen250"]

for path in paths:
    score_train = fid.compute_fid(path, dataset_name='camfilter_50k_train',
            mode="clean", dataset_split="custom")
    score_val = fid.compute_fid(path, dataset_name='camfilter_50k_val',
            mode="clean", dataset_split="custom")
    print(f"FID score (train)for {path}: {score_train}")
    print(f"FID score (val) for {path}: {score_val}")