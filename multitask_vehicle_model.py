import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.features(x)


class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=5, vocab_size=36):
        super().__init__()
        self.backbone = Backbone()

        self.vehicle_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        self.plate_bbox = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

        self.ocr_cnn = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.ocr_lstm = nn.LSTM(128, 64, bidirectional=True, batch_first=True, num_layers=2)
        self.ocr_fc = nn.Linear(64 * 2, vocab_size)

    def forward(self, x):
        features = self.backbone(x)

        cls_out = self.vehicle_cls(features)
        bbox_out = self.plate_bbox(features)

        ocr_feat = self.ocr_cnn(features)
        ocr_feat = ocr_feat.permute(0, 3, 2, 1).flatten(2)  # [B, W, H*C]
        ocr_feat, _ = self.ocr_lstm(ocr_feat)
        ocr_out = self.ocr_fc(ocr_feat)  # [B, W, vocab_size]

        return cls_out, bbox_out, ocr_out


# Focal Loss implementation
def focal_loss(logits, targets, alpha=1.0, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()


# Smooth L1 for bbox
smooth_l1_loss = nn.SmoothL1Loss()


# CTC loss for OCR
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

def ocr_ctc_loss(log_probs, targets, input_lengths, target_lengths):
    log_probs = log_probs.log_softmax(2)  # [B, W, V] -> log_probs
    log_probs = log_probs.permute(1, 0, 2)  # [W, B, V] for CTCLoss
    return ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)


# Example use
if __name__ == '__main__':
    model = MultiTaskModel(num_classes=5, vocab_size=36)
    dummy_input = torch.randn(2, 3, 512, 512)
    cls_out, bbox_out, ocr_out = model(dummy_input)
    print(cls_out.shape, bbox_out.shape, ocr_out.shape)  # [2,5] [2,4] [2,W,36]
