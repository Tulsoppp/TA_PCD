import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    """
    Dice Loss according to Mobile-PolypNet paper equation (1):
    L_DICE = 1 - (2 * sum(p * g) + smooth) / (sum(p^2) + sum(g^2) + smooth)
    """
    def __init__(self, smooth=1e-5, squared_denom=True):
        super().__init__()
        self.smooth = smooth
        self.squared_denom = squared_denom

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs_flat = probs.contiguous().view(probs.size(0), -1)
        targets_flat = targets.contiguous().view(targets.size(0), -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        if self.squared_denom:
            cardinality = torch.sum(probs_flat ** 2, dim=1) + torch.sum(targets_flat ** 2, dim=1)
        else:
            cardinality = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice)


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    Standard in biomedical segmentation for stability and precision.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(y_pred_probs, y_true_binary, threshold=0.5):
    """
    Calculate full evaluation metrics reported in Table 1 & Table 4 of Mobile-PolypNet:
    - DICE (F1-score)
    - mIoU (Jaccard)
    - F2-score (beta=2)
    - MAE (Mean Absolute Error)
    - Sensitivity (Recall)
    - Specificity
    - Accuracy

    Args:
        y_pred_probs: numpy array or torch tensor of predicted probabilities [0, 1]
        y_true_binary: numpy array or torch tensor of ground truth masks {0, 1}
        threshold: binarization threshold (default: 0.5)
    Returns:
        Dictionary of metrics
    """
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()
    if isinstance(y_true_binary, torch.Tensor):
        y_true_binary = y_true_binary.detach().cpu().numpy()

    # Calculate MAE before thresholding
    mae = float(np.mean(np.abs(y_pred_probs - y_true_binary)))

    # Binarize predictions
    y_pred = (y_pred_probs >= threshold).astype(np.uint8)
    y_true = (y_true_binary >= 0.5).astype(np.uint8)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    smooth = 1e-7

    # Precision and Recall (Sensitivity)
    precision = (tp + smooth) / (tp + fp + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth) # Recall
    specificity = (tn + smooth) / (tn + fp + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)

    # Dice (F1)
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)

    # IoU (Jaccard)
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    # F2 Score (beta=2, emphasizing recall)
    f2 = (5.0 * precision * sensitivity) / (4.0 * precision + sensitivity + smooth)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'f2': float(f2),
        'mae': float(mae),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'accuracy': float(accuracy)
    }


if __name__ == "__main__":
    criterion = BCEDiceLoss()
    logits = torch.randn(4, 1, 224, 224)
    targets = torch.randint(0, 2, (4, 1, 224, 224)).float()
    loss = criterion(logits, targets)
    print(f"Sample loss: {loss.item():.4f}")
    
    probs = torch.sigmoid(logits)
    metrics = calculate_metrics(probs, targets)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
