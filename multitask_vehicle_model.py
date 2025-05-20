"""
Многозадачная модель для:
1. Классификации типа транспорта (5 классов)
2. Детекции bounding box номерного знака
3. Распознавания текста (OCR) с механизмом внимания

Архитектура:
- Общий CNN бэкбон для извлечения признаков
- Три специализированные "головы" для каждой задачи
- Механизм внимания для OCR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class ConvBlock(nn.Module):
    """
    Базовый блок свертки с пулингом
    Состоит из:
    - Conv2d -> BatchNorm -> ReLU -> MaxPool2d
    Уменьшает размерность в 2 раза
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),  # Нормализация активаций
            nn.ReLU(inplace=True),        # Активация с оптимизацией памяти
            nn.MaxPool2d(2)               # Уменьшение размерности
        )

    def forward(self, x):
        return self.block(x)

class MultiTaskModel(nn.Module):
    """
    Многозадачная модель с тремя головками:
    1. Классификация транспорта
    2. Регрессия bounding box
    3. Распознавание текста (OCR)
    """
    def __init__(self, num_classes=5, ocr_vocab_size=36 + 1):  # +1 для CTC blank
        super().__init__()
        self.num_classes = num_classes
        self.ocr_vocab_size = ocr_vocab_size

        # ------------------------------------------
        # 1. Общий бэкбон для извлечения признаков
        #    (оптимизирован под RTX 3060 12GB)
        # ------------------------------------------
        self.backbone = nn.Sequential(
            ConvBlock(3, 64),    # 512x512 -> 256x256
            ConvBlock(64, 128),   # 256x256 -> 128x128
            ConvBlock(128, 256),  # 128x128 -> 64x64
            ConvBlock(256, 512)   # 64x64 -> 32x32
        )

        # ------------------------------------------
        # 2. Головка классификации транспорта
        # ------------------------------------------
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Глобальный пулинг
            nn.Flatten(),                  # Вытягивание в вектор
            nn.Linear(512, num_classes)    # Финальная классификация
        )

        # ------------------------------------------
        # 3. Головка детекции bounding box
        #    (предсказывает [x_center, y_center, width, height])
        # ------------------------------------------
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=1),  # Локальные признаки
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 4),        # Регрессия координат
            nn.Sigmoid()                       # Нормализация в [0, 1]
        )

        # ------------------------------------------
        # 4. Головка OCR с механизмом внимания
        # ------------------------------------------
        self.ocr_pool = nn.AdaptiveAvgPool2d((1, None))  # Сохраняет ширину
        self.ocr_feat_proj = nn.Linear(512, 64)          # Проекция признаков
        self.ocr_lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=False)
        
        # Механизм внимания
        self.ocr_attention = nn.Sequential(
            nn.Linear(32, 1),  # Оценка важности каждого элемента последовательности
            nn.Softmax(dim=1)   # Нормализация весов
        )
        
        self.ocr_fc = nn.Linear(32, ocr_vocab_size)  # Финальная классификация символов

    def forward(self, x):
        """
        Прямой проход через модель
        Возвращает кортеж:
        - cls_out: логиты классов транспорта [B, num_classes]
        - bbox_out: координаты bbox [B, 4]
        - ocr_out: логиты символов [B, 1, vocab_size]
        """
        # 1. Извлечение общих признаков
        features = self.backbone(x)  # [B, 512, 32, 32]

        # 2. Классификация транспорта
        cls_out = self.cls_head(features)  # [B, 5]

        # 3. Детекция номера
        bbox_out = self.bbox_head(features)  # [B, 4]

        # 4. Распознавание текста
        # Подготовка признаков
        ocr_feat = self.ocr_pool(features)  # [B, 512, 1, W]
        ocr_feat = ocr_feat.squeeze(2).permute(0, 2, 1)  # [B, W, 512]
        ocr_feat = self.ocr_feat_proj(ocr_feat)  # [B, W, 64]
        
        # Обработка LSTM
        ocr_feat, _ = self.ocr_lstm(ocr_feat)  # [B, W, 32]
        
        # Механизм внимания
        attention_weights = self.ocr_attention(ocr_feat)  # [B, W, 1]
        attended_features = (ocr_feat * attention_weights).sum(dim=1)  # [B, 32]
        
        # Финальная классификация
        ocr_out = self.ocr_fc(attended_features).unsqueeze(1)  # [B, 1, vocab_size]

        return cls_out, bbox_out, ocr_out

    def compute_losses(self, cls_out, bbox_out, ocr_out, cls_target, bbox_target, ocr_target, ocr_lengths):
        """
        Вычисление всех функций потерь:
        1. Focal Loss для классификации
        2. Smooth L1 Loss для регрессии bbox
        3. CTC Loss для OCR
        """

        # Явное приведение типов для совместимости
        cls_out = cls_out.float()
        bbox_out = bbox_out.float()
        ocr_out = ocr_out.float()

        # Проверка размерностей
        if cls_out.dim() == 1:
            cls_out = cls_out.unsqueeze(0)
        
        cls_target = cls_target.long().view(-1)  # Приведение к [batch_size]
        
        # Проверка совпадения размеров
        if cls_out.shape[0] != cls_target.shape[0]:
            raise ValueError(f"Shape mismatch: cls_out {cls_out.shape}, cls_target {cls_target.shape}")
        
        # Вычисление потерь
        loss_cls = F.cross_entropy(cls_out, cls_target)  # Классификация
        loss_bbox = F.smooth_l1_loss(bbox_out, bbox_target)  # Регрессия
        loss_ocr = self.ocr_ctc_loss(ocr_out, ocr_target, ocr_lengths)  # OCR
    
        return loss_cls + loss_bbox + loss_ocr, loss_cls, loss_bbox, loss_ocr

    def focal_loss(self, inputs, targets, alpha=0.8, gamma=2, reduction='mean'):
        """
        Focal Loss для борьбы с дисбалансом классов
        Параметры:
        - alpha: вес для сложных примеров
        - gamma: степень фокусировки
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Вероятность правильного класса
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def smooth_l1_loss(self, inputs, targets, beta=0.1, reduction='mean'):
        """
        Smooth L1 Loss (менее чувствителен к выбросам, чем MSE)
        Параметр beta определяет переход между L1 и L2
        """
        diff = torch.abs(inputs - targets)
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss.mean()

    def ocr_ctc_loss(self, ocr_out, ocr_target, target_lengths):
        """
        Connectionist Temporal Classification (CTC) Loss
        с явным приведением к float32 для совместимости
        
        Args:
            ocr_out: Выход модели [B, 1, vocab_size]
            ocr_target: Целевые последовательности [B, max_len]
            target_lengths: Длины целевых последовательностей [B]
        """
        
        # Приведение к float32 для CTC loss
        ocr_out = ocr_out.float()  # Явное преобразование
        target_lengths = target_lengths.to(ocr_out.device)

        # Проверка размерностей
        if ocr_out.dim() != 3:
            raise ValueError(f"ocr_out должен быть 3D тензором, получено {ocr_out.shape}")
        if target_lengths.dim() != 1:
            raise ValueError(f"target_lengths должен быть 1D тензором, получено {target_lengths.shape}")

        log_probs = ocr_out.log_softmax(dim=2)  # [B, 1, vocab_size]
        log_probs = log_probs.permute(1, 0, 2)  # [1, B, vocab_size]

        # Убедимся, что target_lengths имеет правильный размер
        if len(target_lengths) != log_probs.size(1):
            raise ValueError(
                f"Несоответствие размеров: target_lengths {len(target_lengths)} != batch_size {log_probs.size(1)}"
            )
        
        input_lengths = torch.full(
            size=(log_probs.size(1),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=log_probs.device
        )
        
        return F.ctc_loss(
            log_probs=log_probs,
            targets=ocr_target,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.ocr_vocab_size - 1,
            zero_infinity=True
        )

        '''
        # Подготовка входов
        ocr_out = ocr_out.permute(1, 0, 2)  # [1, B, vocab_size]
        input_lengths = torch.full(
            size=(ocr_out.size(1),),
            fill_value=ocr_out.size(0),
            dtype=torch.long,
            device=ocr_out.device
        )
        
        return F.ctc_loss(
            log_probs=ocr_out.log_softmax(dim=2),  # CTC требует log_softmax
            targets=ocr_target,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.ocr_vocab_size - 1,
            zero_infinity=True
        )
        '''

    def predict(self, x, cls_threshold=0.6, ocr_confidence_threshold=0.4):
        """
        Инференс с постобработкой
        Возвращает:
        - pred_class: предсказанный класс транспорта
        - pred_bbox: координаты bounding box [x, y, w, h]
        - pred_text: распознанный текст
        """
        # Прямой проход
        cls_out, bbox_out, ocr_out = self.forward(x)
        
        # 1. Обработка классификации
        cls_probs = F.softmax(cls_out, dim=1)
        cls_pred = torch.argmax(cls_probs, dim=1)
        # Помечаем как "неизвестный" при низкой уверенности
        cls_pred[cls_probs[torch.arange(len(cls_probs)), cls_pred] < cls_threshold] = 4
        
        # 2. Фильтрация bbox
        plate_bbox = bbox_out if (bbox_out[:, 2] * bbox_out[:, 3] > 0.01).all() else None
        
        # 3. Декодирование OCR
        ocr_probs = F.softmax(ocr_out, dim=2)
        top_char = torch.argmax(ocr_probs, dim=2)
        top_conf = torch.max(ocr_probs, dim=2).values
        if top_conf.item() < ocr_confidence_threshold:
            plate_text = "Не распознано"
        else:
            plate_text = self.decode_plate(top_char[0])
        
        return cls_pred, plate_bbox, plate_text

    def decode_plate(self, char_indices):
        """Преобразование индексов в строку с фильтрацией blank символов"""
        chars = "0123456789АВЕКМНОРСТУХ"  # Словарь символов
        return ''.join([chars[i] for i in char_indices if i < len(chars)])