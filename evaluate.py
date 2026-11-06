import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm

from model import MobilePolypNet
from dataset import get_dataloaders
from losses_and_metrics import calculate_metrics


def overlay_mask_on_image(image_np, mask_np, color=(255, 0, 0), alpha=0.4):
    """
    Overlays binary mask onto RGB image.
    image_np: (H, W, 3) in range [0, 1]
    mask_np: (H, W) in {0, 1}
    color: RGB tuple for overlay color
    """
    img = (image_np * 255).astype(np.uint8).copy()
    colored_mask = np.zeros_like(img)
    for c in range(3):
        colored_mask[:, :, c] = mask_np * color[c]

    mask_bool = mask_np > 0
    img[mask_bool] = cv2.addWeighted(img, 1.0 - alpha, colored_mask, alpha, 0)[mask_bool]
    return img


def evaluate_model(model_path='saved_models/best_mobile_polypnet.pth',
                   images_dir='images', masks_dir='masks', img_size=224,
                   batch_size=8, threshold=0.5, results_dir='results',
                   device='cuda' if torch.cuda.is_available() else 'cpu',
                   num_visual_samples=6):
    os.makedirs(results_dir, exist_ok=True)
    vis_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Loading model checkpoint from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first using train.py.")

    checkpoint = torch.load(model_path, map_location=device)
    model = MobilePolypNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test dataset
    _, _, test_loader = get_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=checkpoint.get('config', {}).get('seed', 42)
    )

    all_metrics = {
        'dice': [],
        'iou': [],
        'f2': [],
        'mae': [],
        'sensitivity': [],
        'specificity': [],
        'accuracy': []
    }

    test_samples_for_vis = []

    print("\nEvaluating Mobile-PolypNet on hold-out Test Set...")
    with torch.no_grad():
        for images, masks, filenames in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            for i in range(images.size(0)):
                prob_i = probs[i, 0].cpu().numpy()
                mask_i = masks[i, 0].cpu().numpy()
                img_i = images[i].permute(1, 2, 0).cpu().numpy()
                fname = filenames[i]

                metrics_i = calculate_metrics(prob_i, mask_i, threshold=threshold)
                for k in all_metrics:
                    all_metrics[k].append(metrics_i[k])

                pred_binary = (prob_i >= threshold).astype(np.uint8)

                # Save a few samples for visualization grid
                if len(test_samples_for_vis) < num_visual_samples:
                    test_samples_for_vis.append({
                        'image': img_i,
                        'true_mask': mask_i,
                        'pred_mask': pred_binary,
                        'filename': fname,
                        'dice': metrics_i['dice'],
                        'iou': metrics_i['iou']
                    })

    # Summary metrics
    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in all_metrics.items()}

    print("\n" + "=" * 65)
    print("Mobile-PolypNet Test Results on Kvasir-SEG (Hold-out Test Set)")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10} | {'Paper Reported (Kvasir)':<25}")
    print("-" * 65)
    print(f"{'DICE':<15} | {mean_metrics['dice']:<10.4f} | {std_metrics['dice']:<10.4f} | 0.9350")
    print(f"{'mIoU':<15} | {mean_metrics['iou']:<10.4f} | {std_metrics['iou']:<10.4f} | 0.8880")
    print(f"{'F2 Score':<15} | {mean_metrics['f2']:<10.4f} | {std_metrics['f2']:<10.4f} | 0.8940")
    print(f"{'MAE':<15} | {mean_metrics['mae']:<10.4f} | {std_metrics['mae']:<10.4f} | 0.0310")
    print(f"{'Sensitivity':<15} | {mean_metrics['sensitivity']:<10.4f} | {std_metrics['sensitivity']:<10.4f} | -")
    print(f"{'Specificity':<15} | {mean_metrics['specificity']:<10.4f} | {std_metrics['specificity']:<10.4f} | -")
    print(f"{'Accuracy':<15} | {mean_metrics['accuracy']:<10.4f} | {std_metrics['accuracy']:<10.4f} | -")
    print("=" * 65)

    # Save to CSV
    metrics_df = pd.DataFrame([mean_metrics])
    csv_path = os.path.join(results_dir, 'test_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nDetailed metrics saved to: {csv_path}")

    # Create visualization grid
    if test_samples_for_vis:
        fig, axes = plt.subplots(len(test_samples_for_vis), 4, figsize=(16, 4 * len(test_samples_for_vis)))
        if len(test_samples_for_vis) == 1:
            axes = np.expand_dims(axes, 0)

        for row, sample in enumerate(test_samples_for_vis):
            img = sample['image']
            gt = sample['true_mask']
            pred = sample['pred_mask']
            overlay = overlay_mask_on_image(img, pred, color=(255, 0, 0), alpha=0.45)

            # Original Image
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"Original: {sample['filename'][:15]}...", fontsize=11)
            axes[row, 0].axis('off')

            # Ground Truth
            axes[row, 1].imshow(gt, cmap='gray')
            axes[row, 1].set_title("Ground Truth Mask", fontsize=11)
            axes[row, 1].axis('off')

            # Predicted Mask
            axes[row, 2].imshow(pred, cmap='gray')
            axes[row, 2].set_title(f"Predicted (Dice: {sample['dice']:.3f}, IoU: {sample['iou']:.3f})", fontsize=11)
            axes[row, 2].axis('off')

            # Overlay
            axes[row, 3].imshow(overlay)
            axes[row, 3].set_title("Prediction Overlay", fontsize=11)
            axes[row, 3].axis('off')

        plt.tight_layout()
        vis_save_path = os.path.join(results_dir, 'test_visual_results.png')
        plt.savefig(vis_save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Visual segmentation results saved to: {vis_save_path}")

    return mean_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mobile-PolypNet")
    parser.add_argument('--model_path', type=str, default='saved_models/best_mobile_polypnet.pth')
    parser.add_argument('--images_dir', type=str, default='images')
    parser.add_argument('--masks_dir', type=str, default='masks')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--samples', type=int, default=6)

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        results_dir=args.results_dir,
        device=args.device,
        num_visual_samples=args.samples
    )
