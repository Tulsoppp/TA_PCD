import os
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import MobilePolypNet, count_parameters
from dataset import get_dataloaders
from losses_and_metrics import BCEDiceLoss, DiceLoss, calculate_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_dices = []
    all_ious = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Fast vectorized tensor dice and iou for training progress
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            intersection = (preds * masks).sum(dim=[1, 2, 3])
            total = preds.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3])
            batch_dice = ((2.0 * intersection + 1e-5) / (total + 1e-5)).mean()
            union = total - intersection
            batch_iou = ((intersection + 1e-5) / (union + 1e-5)).mean()

            all_dices.append(batch_dice.item())
            all_ious.append(batch_iou.item())

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{batch_dice.item():.4f}"
        })

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = float(sum(all_dices) / len(all_dices))
    epoch_iou = float(sum(all_ious) / len(all_ious))

    return epoch_loss, epoch_dice, epoch_iou


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_metrics = {'dice': [], 'iou': [], 'f2': [], 'mae': [], 'sensitivity': [], 'specificity': [], 'accuracy': []}

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            metrics = calculate_metrics(probs, masks)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    val_loss = running_loss / len(dataloader.dataset)
    mean_metrics = {k: float(sum(v) / len(v)) for k, v in all_metrics.items()}
    return val_loss, mean_metrics


def plot_curves(history, save_path='results/loss_dice_curve.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Dice plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_dice'], 'g-', label='Train Dice')
    plt.plot(epochs, history['val_dice'], 'm--', label='Val Dice')
    plt.title('Training & Validation Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Learning curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Mobile-PolypNet on Kvasir-SEG")
    parser.add_argument('--images_dir', type=str, default='images', help='Path to images directory')
    parser.add_argument('--masks_dir', type=str, default='masks', help='Path to masks directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image resolution (default: 224)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8 as in paper)')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience (default: 25)')
    parser.add_argument('--loss_type', type=str, default='bce_dice', choices=['bce_dice', 'dice'], help='Loss function')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--preload', action='store_true', default=True, help='Preload and resize images into RAM for fast CPU training')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit dataset size for fast testing (e.g. 50 or 100)')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print("Mobile-PolypNet Training Setup")
    print(f"Device: {args.device}")
    print(f"Image Resolution: {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Initial LR: {args.lr}")
    print(f"Max Epochs: {args.epochs} (Early Stopping Patience: {args.patience})")
    if args.max_samples:
        print(f"Quick Experiment Subset: {args.max_samples} samples")
    print("=" * 70)

    # Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        preload=args.preload,
        max_samples=args.max_samples
    )

    # Initialize model
    model = MobilePolypNet(in_channels=3, num_classes=1).to(args.device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss and Optimizer
    if args.loss_type == 'dice':
        criterion = DiceLoss().to(args.device)
    else:
        criterion = BCEDiceLoss().to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_f2': [],
        'val_mae': []
    }

    best_val_dice = -1.0
    patience_counter = 0
    best_model_path = os.path.join(args.save_dir, 'best_mobile_polypnet.pth')

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, args.device)

        scheduler.step(val_metrics['dice'])
        epoch_time = time.time() - epoch_start

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_f2'].append(val_metrics['f2'])
        history['val_mae'].append(val_metrics['mae'])

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"mIoU: {val_metrics['iou']:.4f}, F2: {val_metrics['f2']:.4f}")

        # Checkpoint: Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'val_metrics': val_metrics,
                'config': vars(args)
            }, best_model_path)
            print(f"  --> Saved new best model to {best_model_path} (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] No improvement in validation Dice score for {args.patience} epochs. Stopping.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes.")
    print(f"Best Validation Dice Score: {best_val_dice:.4f}")

    # Save history dataframe
    df_history = pd.DataFrame(history)
    csv_path = os.path.join(args.results_dir, 'training_history.csv')
    df_history.to_csv(csv_path, index=False)
    print(f"Training history saved to {csv_path}")

    # Plot curves
    plot_path = os.path.join(args.results_dir, 'loss_dice_curve.png')
    plot_curves(history, save_path=plot_path)


if __name__ == "__main__":
    main()
