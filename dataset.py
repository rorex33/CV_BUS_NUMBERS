import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
from typing import List, Tuple


class VehicleDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        vocab: str,
        image_size: int = 512,
        augment: bool = False,
        debug: bool = False
    ):
        """
        Args:
            root_dir: Путь к директории с данными
            vocab: Строка с допустимыми символами (например, "0123456789АВЕКМНОРСТУХ")
            image_size: Размер изображения на входе модели
            augment: Включить аугментации
            debug: Режим отладки (проверка данных)
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.vocab = vocab
        self.augment = augment
        self.debug = debug
        self.char_to_idx = {c: i+1 for i, c in enumerate(vocab)}  # 0 - blank для CTC
        self.data = []
        self.class_counts = Counter()
        self._load_data()

    def _load_data(self) -> None:
        """Загрузка и валидация данных"""
        label_path = os.path.join(self.root_dir, "labels.txt")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Labels file not found: {label_path}")

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 7:
                    if self.debug:
                        print(f"Skipping malformed line: {line}")
                    continue

                img_name, cls, x, y, w, h, text = parts
                img_path = os.path.join(self.root_dir, "images", img_name)
                
                # Валидация класса
                try:
                    cls = int(cls)
                    if cls < 0 or cls > 4:  # 0-4 классы (4 - неизвестный)
                        raise ValueError
                except ValueError:
                    if self.debug:
                        print(f"Invalid class {cls} in {img_name}")
                    continue

                # Валидация bbox
                try:
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    if w <= 0 or h <= 0:
                        raise ValueError
                except ValueError:
                    if self.debug:
                        print(f"Invalid bbox {x},{y},{w},{h} in {img_name}")
                    continue

                # Проверка символов
                unknown_chars = set(c for c in text if c not in self.char_to_idx)
                if unknown_chars and self.debug:
                    print(f"Unknown chars {unknown_chars} in {img_name}")

                self.data.append((img_name, cls, x, y, w, h, text))
                self.class_counts[cls] += 1

        if self.debug:
            print(f"Loaded {len(self.data)} samples")
            print("Class distribution:", self.class_counts)

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Аугментации изображения"""
        # Цветовые искажения
        if random.random() > 0.5:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:,:,1] = np.clip(image[:,:,1] * random.uniform(0.8, 1.2), 0, 255)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # Гауссов шум
        if random.random() > 0.5:
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)

        return image

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, List[int], int]:
        try:
            img_name, cls, x, y, w, h, text = self.data[idx]
            img_path = os.path.join(self.root_dir, "images", img_name)
            
            # Загрузка изображения
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Error loading image {img_path}")

            # Нормализация координат bbox
            h_orig, w_orig = image.shape[:2]
            x, y, w, h = x/w_orig, y/h_orig, w/w_orig, h/h_orig

            # Аугментации
            if self.augment:
                image = self._augment_image(image)

            # Ресайз и нормализация
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)

            # Кодирование текста
            label_seq = [self.char_to_idx.get(c, 0) for c in text]
            
            return image, cls, torch.tensor([x, y, w, h], dtype=torch.float32), label_seq, len(label_seq)

        except Exception as e:
            if self.debug:
                print(f"Error processing {img_name}: {e}")
            # Возвращаем случайный валидный пример при ошибке
            return self[random.randint(0, len(self)-1)]


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Обработка батча с паддингом для OCR"""
    images, cls_targets, bbox_targets, ocr_targets, ocr_lengths = zip(*batch)
    
    images = torch.stack(images)
    cls_targets = torch.tensor(cls_targets, dtype=torch.long)
    bbox_targets = torch.stack(bbox_targets)
    
    # Паддинг для OCR (0 = blank для CTC)
    max_len = max(ocr_lengths)
    padded_ocr = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(ocr_targets):
        padded_ocr[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    
    return images, cls_targets, bbox_targets, padded_ocr, torch.tensor(ocr_lengths, dtype=torch.long)