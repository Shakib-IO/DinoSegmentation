import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. Model definition (same as in test_segdinov3.py, simplified)
# ----------------------------------------------------------------------
def load_model(ckpt_path, dino_ckpt_path, dino_size, repo_dir, device):
    # Load DINO backbone
    if dino_size == "b":
        backbone = torch.hub.load(repo_dir, 'dinov3_vitb16', source='local', weights=dino_ckpt_path)
    else:
        backbone = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=dino_ckpt_path)

    # Import DPT from your local module (adjust if needed)
    from dpt import DPT
    model = DPT(nclass=1, backbone=backbone)
    model.to(device)

    # Load segmentation checkpoint
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ----------------------------------------------------------------------
# 2. Preprocessing (same as test script: ResizeAndNormalize)
# ----------------------------------------------------------------------
def preprocess_image(image_rgb, input_size):
    """
    image_rgb: numpy array (H,W,3) in RGB order, values 0-255
    input_size: (h, w)
    Returns tensor (1,3,h,w) normalized to ImageNet stats.
    """
    h, w = input_size
    img = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    # ImageNet mean & std
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))   # CHW
    img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # add batch
    return img_tensor

# ----------------------------------------------------------------------
# 3. Prediction
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_mask(model, image_tensor, device, threshold=0.5):
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H,W)
    mask_bin = (prob > threshold).astype(np.uint8) * 255
    return mask_bin, prob

# ----------------------------------------------------------------------
# 4. Overlay creation with configurable color
# ----------------------------------------------------------------------
def create_colored_overlay(image_rgb, mask_bin, color=(255,0,0), alpha=0.45):
    """
    image_rgb: numpy array (H,W,3) RGB, values 0-255
    mask_bin:  numpy array (H,W), values 0 or 255
    color: tuple (R,G,B) each 0-255
    alpha: overlay opacity (0=transparent, 1=opaque)
    Returns overlay image in RGB.
    """
    # Create colored mask layer
    colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    # Apply color to mask region
    # for c in range(3):
    colored_mask[:, :, 0] = mask_bin # * (color[c] / 255.0)
    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, colored_mask, alpha, 0)
    return overlay


# ----------------------------------------------------------------------
# 5. Main inference on one image
# ----------------------------------------------------------------------
def process_one_image(model, image_path, device, input_size, threshold, color, alpha, output_dir):
    # Read image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Could not read {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Preprocess
    img_tensor = preprocess_image(img_rgb, input_size)

    # Predict mask (at model input size)
    mask_bin_resized, prob_map = predict_mask(model, img_tensor, device, threshold)

    # Resize mask back to original size
    mask_bin = cv2.resize(mask_bin_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Create overlay
    overlay_rgb = create_colored_overlay(img_rgb, mask_bin, color=color, alpha=alpha)

    # Save results
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_mask = os.path.join(output_dir, f"{base}_mask.png")
    out_overlay = os.path.join(output_dir, f"{base}_overlay.png")
    cv2.imwrite(out_mask, mask_bin)
    cv2.imwrite(out_overlay, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved: {out_mask} and {out_overlay}")

# ----------------------------------------------------------------------
# 6. Main entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Overlay predicted mask on images")
    parser.add_argument("--image", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Segmentation model checkpoint")
    parser.add_argument("--dino_ckpt", type=str, required=True, help="DINOv3 pretrained weights")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b","s"])
    parser.add_argument("--repo_dir", type=str, default="./dinov3", help="DINOv3 torch.hub repo")
    parser.add_argument("--input_h", type=int, default=1024)
    parser.add_argument("--input_w", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--color", type=str, default="red",
                        help="red, green, blue, yellow, magenta, cyan, or custom 'R,G,B'")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output_dir", type=str, default="./overlay_output")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Parse color
    color_map = {
        "red": (255,0,0),
        "green": (0,255,0),
        "blue": (0,0,255),
        "yellow": (255,255,0),
        "magenta": (255,0,255),
        "cyan": (0,255,255)
    }
    if args.color in color_map:
        color = color_map[args.color]
    else:
        # try custom "R,G,B"
        parts = args.color.split(",")
        if len(parts) == 3:
            color = tuple(int(c.strip()) for c in parts)
        else:
            raise ValueError(f"Unknown color: {args.color}. Use predefined or 'R,G,B'")

    # Load model
    print("Loading model...")
    model = load_model(args.ckpt, args.dino_ckpt, args.dino_size, args.repo_dir, device)

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Process single image or all images in folder
    if os.path.isfile(args.image):
        process_one_image(model, args.image, device, (args.input_h, args.input_w),
                          args.threshold, color, args.alpha, args.output_dir)
    else:
        # assume directory
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        images = [f for f in os.listdir(args.image) if f.lower().endswith(exts)]
        for img_file in tqdm(images, desc="Processing"):
            path = os.path.join(args.image, img_file)
            process_one_image(model, path, device, (args.input_h, args.input_w),
                              args.threshold, color, args.alpha, args.output_dir)

if __name__ == "__main__":
    main()