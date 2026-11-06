import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidualBlock(nn.Module):
    """
    MobileNetV2-style Inverted Residual Block as described in Mobile-PolypNet:
    1. 1x1 Conv (expansion) + BatchNorm + ReLU6
    2. 3x3 Depthwise Conv (filtering) + BatchNorm + ReLU6
    3. 1x1 Conv (contraction/projection) + BatchNorm + Linear
    Residual skip connection is added when stride == 1 and in_channels == out_channels.
    """
    def __init__(self, in_channels: int, expand_channels: int, out_channels: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.stride = stride
        self.use_residual = use_residual and (stride == 1) and (in_channels == out_channels)

        # 1x1 Expansion pointwise conv
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU6(inplace=True)
        )

        # 3x3 Depthwise conv
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=stride, padding=1, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU6(inplace=True)
        )

        # 1x1 Projection / contraction conv (linear activation)
        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.project_conv(out)
        if self.use_residual:
            return x + out
        return out


class MobilePolypNet(nn.Module):
    """
    Mobile-PolypNet: Lightweight Colon Polyp Segmentation Network
    Reference: Karmakar & Nooshabadi, Journal of Imaging 2022
    
    Encoder: 5 resolution levels using bottleneck inverted residual blocks.
    Decoder: Upsampling + skip connections + bottleneck inverted residual blocks.
    Head: 2 conv layers (32 -> 16 -> 1) with Sigmoid activation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # --- 3.1.1 Input Layer (Stem) ---
        # Traditional conv 32 filters -> depthwise 3x3 -> pointwise 32 filters
        self.stem_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.stem_dw = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.stem_pw = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32)
            # Linear activation
        )

        # --- 3.1.2 Encoder ---
        # Level 1: 3 blocks, expand=48, contract=8, stride=2 on 1st block (224 -> 112)
        self.enc1_1 = InvertedResidualBlock(32, 48, 8, stride=2, use_residual=False)
        self.enc1_2 = InvertedResidualBlock(8, 48, 8, stride=1, use_residual=True)
        self.enc1_3 = InvertedResidualBlock(8, 48, 8, stride=1, use_residual=True)

        # Level 2: 3 blocks, expand=96, contract=16, stride=2 on 1st block (112 -> 56)
        self.enc2_1 = InvertedResidualBlock(8, 96, 16, stride=2, use_residual=False)
        self.enc2_2 = InvertedResidualBlock(16, 96, 16, stride=1, use_residual=True)
        self.enc2_3 = InvertedResidualBlock(16, 96, 16, stride=1, use_residual=True)

        # Level 3: 3 blocks, expand=144, contract=32, stride=2 on 1st block (56 -> 28)
        self.enc3_1 = InvertedResidualBlock(16, 144, 32, stride=2, use_residual=False)
        self.enc3_2 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)
        self.enc3_3 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)

        # Level 4: 3 blocks, expand=144, contract=32, stride=2 on 1st block (28 -> 14)
        self.enc4_1 = InvertedResidualBlock(32, 144, 32, stride=2, use_residual=False)
        self.enc4_2 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)
        self.enc4_3 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)

        # Level 5 (Bottleneck): stride=2 on 1st block, total 6 blocks (14 -> 7)
        self.enc5_1 = InvertedResidualBlock(32, 144, 32, stride=2, use_residual=False)
        self.enc5_2 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)
        self.enc5_3 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)
        self.enc5_4 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)
        self.enc5_5 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)
        self.enc5_6 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)

        # --- 3.1.3 Decoder ---
        # Up 4: 7x7 -> 14x14
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        # Concat: up4 (16) + enc4 (32) = 48
        self.dec4_1 = InvertedResidualBlock(48, 144, 32, stride=1, use_residual=False)
        self.dec4_2 = InvertedResidualBlock(32, 144, 32, stride=1, use_residual=True)

        # Up 3: 14x14 -> 28x28
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        # Concat: up3 (16) + enc3 (32) = 48
        self.dec3_1 = InvertedResidualBlock(48, 96, 16, stride=1, use_residual=False)
        self.dec3_2 = InvertedResidualBlock(16, 96, 16, stride=1, use_residual=True)

        # Up 2: 28x28 -> 56x56
        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        # Concat: up2 (16) + enc2 (16) = 32
        self.dec2_1 = InvertedResidualBlock(32, 96, 16, stride=1, use_residual=False)
        self.dec2_2 = InvertedResidualBlock(16, 96, 16, stride=1, use_residual=True)

        # Up 1: 56x56 -> 112x112
        self.up1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        # Concat: up1 (8) + enc1 (8) = 16
        self.dec1_1 = InvertedResidualBlock(16, 48, 8, stride=1, use_residual=False)
        self.dec1_2 = InvertedResidualBlock(8, 48, 8, stride=1, use_residual=True)

        # Up 0: 112x112 -> 224x224
        self.up0 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        # Concat: up0 (8) + stem (32) = 40
        self.dec0_1 = InvertedResidualBlock(40, 48, 8, stride=1, use_residual=False)
        self.dec0_2 = InvertedResidualBlock(8, 48, 8, stride=1, use_residual=True)

        # --- 3.1.4 Output Layer ---
        self.head = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
            # Sigmoid is applied externally or in inference
        )

    def forward(self, x):
        # Input layer
        x0 = self.stem_conv1(x)
        x0 = self.stem_dw(x0)
        x0 = self.stem_pw(x0) # (B, 32, 224, 224)

        # Encoder Level 1
        c1 = self.enc1_1(x0)
        c1 = self.enc1_2(c1)
        c1 = self.enc1_3(c1) # (B, 8, 112, 112)

        # Encoder Level 2
        c2 = self.enc2_1(c1)
        c2 = self.enc2_2(c2)
        c2 = self.enc2_3(c2) # (B, 16, 56, 56)

        # Encoder Level 3
        c3 = self.enc3_1(c2)
        c3 = self.enc3_2(c3)
        c3 = self.enc3_3(c3) # (B, 32, 28, 28)

        # Encoder Level 4
        c4 = self.enc4_1(c3)
        c4 = self.enc4_2(c4)
        c4 = self.enc4_3(c4) # (B, 32, 14, 14)

        # Encoder Level 5 (Bottleneck)
        c5 = self.enc5_1(c4)
        c5 = self.enc5_2(c5)
        c5 = self.enc5_3(c5)
        c5 = self.enc5_4(c5)
        c5 = self.enc5_5(c5)
        c5 = self.enc5_6(c5) # (B, 32, 7, 7)

        # Decoder Level 4
        e4 = self.up4(c5)
        e4 = torch.cat([e4, c4], dim=1) # (B, 48, 14, 14)
        e4 = self.dec4_1(e4)
        e4 = self.dec4_2(e4) # (B, 32, 14, 14)

        # Decoder Level 3
        e3 = self.up3(e4)
        e3 = torch.cat([e3, c3], dim=1) # (B, 48, 28, 28)
        e3 = self.dec3_1(e3)
        e3 = self.dec3_2(e3) # (B, 16, 28, 28)

        # Decoder Level 2
        e2 = self.up2(e3)
        e2 = torch.cat([e2, c2], dim=1) # (B, 32, 56, 56)
        e2 = self.dec2_1(e2)
        e2 = self.dec2_2(e2) # (B, 16, 56, 56)

        # Decoder Level 1
        e1 = self.up1(e2)
        e1 = torch.cat([e1, c1], dim=1) # (B, 16, 112, 112)
        e1 = self.dec1_1(e1)
        e1 = self.dec1_2(e1) # (B, 8, 112, 112)

        # Decoder Level 0
        e0 = self.up0(e1)
        e0 = torch.cat([e0, x0], dim=1) # (B, 40, 224, 224)
        e0 = self.dec0_1(e0)
        e0 = self.dec0_2(e0) # (B, 8, 224, 224)

        # Output head
        out = self.head(e0) # (B, 1, 224, 224)
        return out


def count_parameters(model: nn.Module):
    """Counts total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    net = MobilePolypNet(in_channels=3, num_classes=1)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = net(dummy_input)
    total, trainable = count_parameters(net)
    print(f"Output shape: {output.shape}")
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model size in MB: {total * 4 / (1024 * 1024):.2f} MB")
