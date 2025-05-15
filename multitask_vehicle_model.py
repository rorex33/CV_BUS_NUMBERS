import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=5, ocr_vocab_size=36 + 1):  # +1 for CTC blank
        super().__init__()
        self.num_classes = num_classes
        self.ocr_vocab_size = ocr_vocab_size

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
            nn.Sigmoid()
        )

        # OCR head
        self.ocr_pool = nn.AdaptiveAvgPool2d((1, None))
        self.ocr_feat_proj = nn.Linear(2048, 128)
        self.ocr_lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.ocr_fc = nn.Linear(128, ocr_vocab_size)

    def forward(self, x):
        features = self.backbone(x)
        cls_out = self.cls_head(features)
        bbox_out = self.bbox_head(features)

        ocr_feat = self.ocr_pool(features)
        ocr_feat = ocr_feat.squeeze(2).permute(0, 2, 1)  # B x W x 2048
        ocr_feat = self.ocr_feat_proj(ocr_feat)
        ocr_feat, _ = self.ocr_lstm(ocr_feat)
        ocr_out = self.ocr_fc(ocr_feat)  # B x W x vocab_size

        return cls_out, bbox_out, ocr_out

    def compute_losses(self, cls_out, cls_target, bbox_out, bbox_target, ocr_out, ocr_target, ocr_lengths):
        loss_cls = self.focal_loss(cls_out, cls_target)
        loss_bbox = self.smooth_l1_loss(bbox_out, bbox_target)
        loss_ocr = self.ocr_ctc_loss(ocr_out, ocr_target, ocr_lengths)
        return loss_cls + loss_bbox + loss_ocr, loss_cls, loss_bbox, loss_ocr

    def focal_loss(self, inputs, targets, alpha=1, gamma=2, reduction='mean'):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss

        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def smooth_l1_loss(self, inputs, targets, reduction='mean'):
        loss = F.smooth_l1_loss(inputs, targets, reduction=reduction)
        return loss

    def ocr_ctc_loss(self, ocr_out, ocr_target, target_lengths):
        # ocr_out: B x W x V => W x B x V
        ocr_out = ocr_out.permute(1, 0, 2)
        input_lengths = torch.full(size=(ocr_out.size(1),), fill_value=ocr_out.size(0), dtype=torch.long)
        return F.ctc_loss(ocr_out, ocr_target, input_lengths, target_lengths, blank=self.ocr_vocab_size - 1, zero_infinity=True)

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