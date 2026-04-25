# DinoSegmentation
Apply DINOv3 for Segmentaion

![](./media/segment_img.png)
[SegDINO: An Efficient Design for Medical and Natural Image Segmentation with DINO-V3](https://arxiv.org/abs/2509.00833)

SegDINO, an efficient image segmentation framework that couples a frozen DINOv3 backbone with a lightweight MLP decoder, achieving state-of-the-art performance on both medical and natural image segmentation tasks while maintaining minimal parameter and computational cost.

### 1. Specification of dependencies
```bash
# 1. Clone this Repository
git clone https://github.com/Shakib-IO/DinoSegmentation.git
cd DinoSegmentation

# (Must Required Python 3.10+)
# 2. Create a virtual environment 
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Checkpoint
Download [DINOv3](https://github.com/facebookresearch/dinov3) pretrained weights and place them in:

```
./web_pth
```

### 3. Dataset
This dataset contains high-resolution images (approximately 3000 × 1845 pixels) of drill core samples (igneous/granitic rock) stored in industrial core boxes. The task is to perform semantic segmentation to separate the rock material from the surrounding cardboard packaging and labels. The images are provided in `.jpg` format, and the corresponding masks are in `.png`. Download the dataset from the provided [link](https://github.com/GoldspotDiscoveries/gsb-test-ml/tree/main) and place the `images` and `masks` folders inside the `data` directory.

```bash
root/
├── train/
│   ├── img/
│   └── label/
└── test/
    ├── img/
    └── label/
```

### 4. Train & Evaluation code
Example training command:
```bash
bash train.sh

OR

python train_segdino.py \
  --data_dir ./data \
  --dataset rock \
  --input_h 512 --input_w 512 \
  --dino_size s \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --repo_dir ./dinov3 \
  --img_dir_name img \
  --label_dir_name label \
  --mask_ext '.png' \
  --epochs 30 \
  --batch_size 2 \
  --lr 1e-4
```

Example testing command:
```bash
bash test.sh

OR

python test_segdino.py \
  --data_dir ./data \
  --dataset rock \
  --input_h 512 --input_w 512 \
  --dino_size s \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --ckpt ./runs/segdino_s_512_rock/ckpts/latest.pth \
  --repo_dir ./dinov3 \
  --img_dir_name img \
  --label_dir_name label \
  --mask_ext '.png'
```

Example inference command:
```bash
bash inference.py 

OR

python inference.py \
    --image data/rock/test/img/image.2025.12.17.22.29.45.993.236.jpg \
    --ckpt runs/segdino_s_512_rock/ckpts/latest.pth \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dino_size s \
    --repo_dir ./dinov3 \
    --color red \
    --alpha 0.4 \
    --output_dir ./output_predictions
```

### 5. Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yang2025segdino,
  title={SegDINO: An Efficient Design for Medical and Natural Image Segmentation with DINO-V3},
  author={Yang, Sicheng and Wang, Hongqiu and Xing, Zhaohu and Chen, Sixiang and Zhu, Lei},
  journal={arXiv preprint arXiv:2509.00833},
  year={2025},
  url={https://arxiv.org/abs/2509.00833}
}