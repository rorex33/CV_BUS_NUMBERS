import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Добавлен пулинг для уменьшения размерности
        )

    def forward(self, x):
        return self.block(x)

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=5, ocr_vocab_size=36 + 1):  # +1 для CTC blank
        super().__init__()
        self.num_classes = num_classes
        self.ocr_vocab_size = ocr_vocab_size

        # Упрощенный бэкбон (оптимизирован под RTX 3060 12GB)
        self.backbone = nn.Sequential(
            ConvBlock(3, 64),    # 512x512 -> 256x256
            ConvBlock(64, 128),  # 256x256 -> 128x128
            ConvBlock(128, 256), # 128x128 -> 64x64
            ConvBlock(256, 512)  # 64x64 -> 32x32
        )

        # Классификация транспорта
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        # Детекция номера (улучшенная локальными фичами)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=1),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 4),  # Для 512x512 вход -> 32x32 после бэкбона
            nn.Sigmoid()
        )

        # OCR с механизмом внимания
        self.ocr_pool = nn.AdaptiveAvgPool2d((1, None))  # Сохраняет ширину (W)
        self.ocr_feat_proj = nn.Linear(512, 64)  # Уменьшено с 2048->64
        self.ocr_lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=False)  # Оптимизировано для CPU
        self.ocr_attention = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        self.ocr_fc = nn.Linear(32, ocr_vocab_size)

    def forward(self, x):
        # Общие фичи
        features = self.backbone(x)  # B x 512 x 32 x 32

        # Классификация транспорта
        cls_out = self.cls_head(features)  # B x 5

        # Детекция номера
        bbox_out = self.bbox_head(features)  # B x 4 (нормализованные координаты)

        # OCR
        ocr_feat = self.ocr_pool(features)  # B x 512 x 1 x W
        ocr_feat = ocr_feat.squeeze(2).permute(0, 2, 1)  # B x W x 512
        ocr_feat = self.ocr_feat_proj(ocr_feat)  # B x W x 64
        ocr_feat, _ = self.ocr_lstm(ocr_feat)  # B x W x 32
        
        # Механизм внимания
        attention_weights = self.ocr_attention(ocr_feat)  # B x W x 1
        attended_features = (ocr_feat * attention_weights).sum(dim=1)  # B x 32
        ocr_out = self.ocr_fc(attended_features).unsqueeze(1)  # B x 1 x vocab_size

        return cls_out, bbox_out, ocr_out

    def compute_losses(self, cls_out, cls_target, bbox_out, bbox_target, ocr_out, ocr_target, ocr_lengths):
        # Focal Loss для классификации
        loss_cls = self.focal_loss(cls_out, cls_target)
        
        # Smooth L1 Loss для bbox
        loss_bbox = self.smooth_l1_loss(bbox_out, bbox_target)
        
        # CTC Loss для OCR (модифицирован под attention)
        loss_ocr = self.ocr_ctc_loss(ocr_out, ocr_target, ocr_lengths)
        
        return loss_cls + loss_bbox + loss_ocr, loss_cls, loss_bbox, loss_ocr

    def focal_loss(self, inputs, targets, alpha=0.8, gamma=2, reduction='mean'):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def smooth_l1_loss(self, inputs, targets, beta=0.1, reduction='mean'):
        diff = torch.abs(inputs - targets)
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss.mean()

    def ocr_ctc_loss(self, ocr_out, ocr_target, target_lengths):
        # ocr_out: B x 1 x V -> 1 x B x V
        ocr_out = ocr_out.permute(1, 0, 2)
        input_lengths = torch.full(ocr_out.size(1), 1, dtype=torch.long)  # Все последовательности длины 1
        return F.ctc_loss(
            ocr_out, ocr_target, input_lengths, target_lengths,
            blank=self.ocr_vocab_size - 1, zero_infinity=True
        )

    def predict(self, x, cls_threshold=0.6, ocr_confidence_threshold=0.4):
        """Инференс с постобработкой"""
        cls_out, bbox_out, ocr_out = self.forward(x)
        
        # Классификация транспорта
        cls_probs = F.softmax(cls_out, dim=1)
        cls_pred = torch.argmax(cls_probs, dim=1)
        cls_pred[cls_probs[torch.arange(len(cls_probs)), cls_pred] < cls_threshold] = 4  # "Неизвестный"
        
        # Детекция номера (фильтрация по уверенности)
        plate_bbox = bbox_out if (bbox_out[:, 2] * bbox_out[:, 3] > 0.01).all() else None  # Отсеиваем слишком маленькие bbox
        
        # Распознавание текста
        ocr_probs = F.softmax(ocr_out, dim=2)
        top_char = torch.argmax(ocr_probs, dim=2)
        top_conf = torch.max(ocr_probs, dim=2).values
        if top_conf.item() < ocr_confidence_threshold:
            plate_text = "Не распознано"
        else:
            plate_text = self.decode_plate(top_char[0])
        
        return cls_pred, plate_bbox, plate_text

    def decode_plate(self, char_indices):
        """Перевод индексов в символы (замените на ваш словарь)"""
        chars = "0123456789АВЕКМНОРСТУХ"  # Пример словаря
        return ''.join([chars[i] for i in char_indices if i < len(chars)])