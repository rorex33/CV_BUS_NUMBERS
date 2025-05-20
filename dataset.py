"""
Модуль для создания датасета для задачи:
1. Классификации транспорта
2. Детекции bounding box номерных знаков
3. Распознавания текста (OCR)

Особенности:
- Поддержка аугментаций
- Обработка данных для CTC loss
- Автоматическая нормализация bounding box
- Отказоустойчивая загрузка изображений
"""

import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List

class VehicleDataset(Dataset):
    """
    Кастомный датасет для многозадачного обучения.
    
    Args:
        root_dir (str): Путь к папке с данными (должна содержать images/ и labels.txt)
        vocab (str): Строка с допустимыми символами (например, "0123456789АВЕКМНОРСТУХ")
        image_size (int): Размер, к которому будут ресайзиться изображения (по умолчанию 512)
        augment (bool): Включить аугментации (по умолчанию False)
        debug (bool): Режим отладки с дополнительным выводом (по умолчанию False)
    """
    def __init__(
        self,
        root_dir: str,
        vocab: str,
        image_size: int = 512,
        augment: bool = False,
        debug: bool = False
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.vocab = vocab
        # Словарь для преобразования символов в индексы (0 зарезервирован для blank в CTC)
        self.char_to_idx = {c: i+1 for i, c in enumerate(vocab)}
        self.augment = augment
        self.debug = debug
        self.data = []  # Список для хранения меток данных
        self._load_data()  # Загрузка данных при инициализации

    def _load_data(self):
        """
        Загрузка данных из labels.txt. Формат строки:
        имя_файла.jpg,класс,x_center,y_center,width,height,текст_номера
        """
        label_path = os.path.join(self.root_dir, "labels.txt")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Labels file not found: {label_path}")

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split(',')
                # Пропускаем некорректные строки
                if len(parts) != 7:
                    if self.debug:
                        print(f"Skipping malformed line: {line}")
                    continue

                img_name, cls, x, y, w, h, text = parts
                img_path = os.path.join(self.root_dir, "images", img_name)
                
                try:
                    # Валидация и преобразование типов
                    cls = int(cls)
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    self.data.append((img_name, cls, x, y, w, h, text))
                except ValueError as e:
                    if self.debug:
                        print(f"Invalid data in line: {line}. Error: {e}")

    def __len__(self):
        """Возвращает количество элементов в датасете"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], int]:
        """
        Получение одного элемента датасета по индексу.
        
        Returns:
            tuple: (image, class, bbox, ocr_seq, ocr_length)
            - image: тензор изображения [3, H, W]
            - class: тензор класса (скаляр)
            - bbox: тензор координат [x_center, y_center, width, height]
            - ocr_seq: список индексов символов
            - ocr_length: длина последовательности
        """
        img_name, cls, x, y, w, h, text = self.data[idx]
        img_path = os.path.join(self.root_dir, "images", img_name)
        
        # Загрузка изображения с обработкой ошибок
        image = cv2.imread(img_path)
        if image is None:
            if self.debug:
                print(f"Error loading image: {img_path}, using random sample instead")
            return self[random.randint(0, len(self)-1)]  # Fallback на случайный образец
            
        # Конвертация из BGR в RGB (OpenCV по умолчанию использует BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Нормализация координат bbox к относительным значениям [0, 1]
        h_orig, w_orig = image.shape[:2]
        x, y, w, h = x/w_orig, y/h_orig, w/w_orig, h/h_orig
        
        # Ресайз изображения к единому размеру
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Конвертация в тензор и нормализация пикселей в [0, 1]
        # Перестановка осей из HWC в CHW формат (требуется PyTorch)
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        
        # Подготовка bbox в формате [x_center, y_center, width, height]
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)
        
        # Кодирование текста в последовательность индексов
        label_seq = [self.char_to_idx.get(c, 0) for c in text]  # 0 - blank символ
        
        return (
            image,          # Тензор изображения [3, H, W]
            torch.tensor(cls, dtype=torch.long),  # Класс как тензор (скаляр)
            bbox,           # Тензор координат bbox [4]
            label_seq,      # Закодированная последовательность символов
            len(label_seq)  # Длина последовательности
        )

def collate_fn(batch):
    """
    Функция для объединения отдельных элементов в батчи.
    Особенно важна для последовательностей переменной длины в OCR.
    
    Args:
        batch: Список элементов, возвращаемых __getitem__
    
    Returns:
        tuple: (images, classes, bboxes, ocr_seqs, ocr_lengths)
        - images: тензор [B, 3, H, W]
        - classes: тензор [B]
        - bboxes: тензор [B, 4]
        - ocr_seqs: паддинг-тензор [B, max_len]
        - ocr_lengths: тензор длин последовательностей [B]
    """
    # Распаковка батча по компонентам
    images, cls_targets, bbox_targets, ocr_targets, ocr_lengths = zip(*batch)
    
    # Объединение в тензоры
    images = torch.stack(images)          # [B, 3, H, W]
    classes = torch.stack(cls_targets)    # [B]
    bboxes = torch.stack(bbox_targets)    # [B, 4]
    
    # Специальная обработка последовательностей переменной длины:
    # 1. Конвертируем каждую последовательность в тензор
    # 2. Применяем паддинг до максимальной длины в батче
    ocr_seqs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x, dtype=torch.long) for x in ocr_targets],
        batch_first=True,
        padding_value=0  # Используем 0 как padding индекс
    )  # [B, max_len]
    
    ocr_lengths = torch.tensor(ocr_lengths, dtype=torch.long)  # [B]
    
    return (images, classes, bboxes, ocr_seqs, ocr_lengths)