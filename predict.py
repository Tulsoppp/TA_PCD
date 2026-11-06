import os
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from model import MobilePolypNet
from evaluate import overlay_mask_on_image


def predict_single_image(model, image_path, img_size=224, threshold=0.5, device='cpu'):
    original_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = original_img.size

    # Resize and tensor conversion
    resized_img = original_img.resize((img_size, img_size), Image.BILINEAR)
    img_tensor = TF.to_tensor(resized_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Binarize
    mask_224 = (probs >= threshold).astype(np.uint8)

    # Resize predicted mask back to original resolution
    mask_orig = cv2.resize(mask_224, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Make overlay
    orig_np = np.array(original_img) / 255.0
    overlay_orig = overlay_mask_on_image(orig_np, mask_orig, color=(0, 255, 0), alpha=0.4)

    return orig_np, mask_orig, overlay_orig, probs


def main():
    parser = argparse.ArgumentParser(description="Segment colon polyp in an image using Mobile-PolypNet")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='saved_models/best_mobile_polypnet.pth')
    parser.add_argument('--output', type=str, default='results/prediction_output.png')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Trained model not found at {args.model_path}. Train the model first.")

    checkpoint = torch.load(args.model_path, map_location=args.device)
    model = MobilePolypNet(in_channels=3, num_classes=1).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Running inference on {args.image}...")
    orig_np, mask_orig, overlay_orig, _ = predict_single_image(
        model, args.image, img_size=args.img_size, threshold=args.threshold, device=args.device
    )

    # Save output visualization
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mask_visual = (mask_orig * 255).astype(np.uint8)

    # Concatenate side by side: [Original | Mask | Overlay]
    orig_bgr = cv2.cvtColor((orig_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)
    overlay_bgr = cv2.cvtColor(overlay_orig, cv2.COLOR_RGB2BGR)

    side_by_side = np.hstack([orig_bgr, mask_bgr, overlay_bgr])
    cv2.imwrite(args.output, side_by_side)
    print(f"Prediction result saved to {args.output}")


if __name__ == "__main__":
    main()
