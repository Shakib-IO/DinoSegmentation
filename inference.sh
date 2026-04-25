python inference.py \
    --image segdata/rock/test/img/image.2025.12.17.22.29.45.993.236.jpg \
    --ckpt runs/segdino_s_512_rock/ckpts/latest.pth \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dino_size s \
    --repo_dir ./dinov3 \
    --color red \
    --alpha 0.4 \
    --output_dir ./output_predictions