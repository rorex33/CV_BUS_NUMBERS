import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=5, ocr_vocab_size=36 + 1):  # +1 for CTC blank
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 1024),
            ConvBlock(1024, 2048)
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 4),
            nn.Sigmoid()  # bbox coords are relative (0-1)
        )

        # OCR head
        self.ocr_pool = nn.AdaptiveAvgPool2d((1, None))  # keep width
        self.ocr_feat_proj = nn.Linear(2048, 128)
        self.ocr_lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.ocr_fc = nn.Linear(128, ocr_vocab_size)  # 64*2=128

    def forward(self, x):
        features = self.backbone(x)  # B x 2048 x H x W

        cls_out = self.cls_head(features)           # B x num_classes
        bbox_out = self.bbox_head(features)         # B x 4

        ocr_feat = self.ocr_pool(features)          # B x 2048 x 1 x W
        ocr_feat = ocr_feat.squeeze(2).permute(0, 2, 1)  # B x W x 2048
        ocr_feat = self.ocr_feat_proj(ocr_feat)     # B x W x 128
        ocr_feat, _ = self.ocr_lstm(ocr_feat)       # B x W x 128
        ocr_out = self.ocr_fc(ocr_feat)             # B x W x vocab_size

        return cls_out, bbox_out, ocr_out