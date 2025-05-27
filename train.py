"""
Модуль для обучения многозадачной модели:
1. Классификация типа транспорта
2. Детекция bounding box
3. Распознавание текста (OCR)

Основные компоненты:
- Загрузка и обработка данных
- Инициализация модели
- Цикл обучения с mixed precision
- Логирование и сохранение моделей
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn.utils import clip_grad_norm_
from dataset import VehicleDataset, collate_fn
from multitask_vehicle_model import MultiTaskModel
import yaml
from tqdm import tqdm  # Для красивого прогресс-бара
import os
import numpy as np

# ==============================================
# Основная функция обучения
# ==============================================
def main():
    """
    Главный цикл обучения модели. Выполняет:
    1. Загрузку конфигурации
    2. Подготовку данных
    3. Инициализацию модели
    4. Цикл обучения с валидацией
    5. Сохранение лучшей модели
    """
    
    # ------------------------------
    # 1. Конфигурация и настройки
    # ------------------------------
    # Загрузка параметров из YAML-файла
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Автоматический выбор устройства (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------
    # 2. Подготовка модели
    # ------------------------------
    # Инициализация многозадачной модели
    model = MultiTaskModel(
        num_classes=5,  # Количество классов транспорта
        ocr_vocab_size=len(config['model']['vocab']) + 1, # Размер словаря OCR + blank символ
        freeze_cls=True # Замораживаем классификацию
    ).to(device)  # Перенос модели на выбранное устройство
    
    # Отключаем autocast для CTC loss
    @torch.autocast(device_type=device.type, enabled=False)
    def compute_losses_wrapper(*args, **kwargs):
        return model.compute_losses(*args, **kwargs)

    # ------------------------------
    # 3. Загрузка данных
    # ------------------------------
    # Создание датасета для обучения
    train_dataset = VehicleDataset(
        root_dir="E:\IT\data",  # Путь к обучающим данным
        vocab=config['model']['vocab'],  # Словарь символов
        image_size=config['image']['size'],  # Размер входного изображения
        augment=True # Аугментация данных
    )
    
    # Создание DataLoader для батчевой обработки
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],  # Размер батча
        shuffle=True,  # Перемешивание данных
        collate_fn=collate_fn,  # Функция для сборки батчей
        num_workers=0,  # Количество процессов для загрузки (0 для Windows)
        pin_memory=True  # Ускоряет перенос данных на GPU
    )

    # ------------------------------
    # 4. Оптимизация и планировщик
    # ------------------------------
    # Инициализация оптимизатора Adam
    optimizer = Adam(
        model.parameters(),  # Параметры для оптимизации
        lr=config['training']['lr']  # Скорость обучения
    )
    
    # Планировщик для динамического изменения lr
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Мониторим уменьшение потерь
        factor=0.5,  # Коэффициент уменьшения lr
        patience=3  # Количество эпох без улучшения перед уменьшением lr
    )
    
    # Инициализация GradScaler для mixed precision обучения
    scaler = torch.amp.GradScaler(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # ------------------------------
    # 5. Цикл обучения
    # ------------------------------
    best_loss = float('inf')  # Лучший показатель потерь
    
    for epoch in range(config['training']['epochs']):
        model.train()  # Переводим модель в режим обучения
        epoch_loss = 0.0  # Суммарные потери за эпоху
        
        # Прогресс-бар для визуализации обучения
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['training']['epochs']}"
        )
        
        for batch in progress_bar:
            # Перенос данных на устройство (GPU/CPU)
            images, cls_targets, bbox_targets, ocr_targets, ocr_lengths = [
                x.to(device, non_blocking=True) for x in batch
            ]
            
            # Обнуление градиентов (оптимизированная версия)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass с mixed precision
            with torch.amp.autocast(device_type=device.type):
                # Получение предсказаний модели
                cls_out, bbox_out, ocr_out = model(images)
                
                # Вычисление функции потерь
                loss, loss_cls, loss_bbox, loss_ocr = model.compute_losses(
                    cls_out,  # Выход классификатора
                    bbox_out,  # Выход детектора bbox
                    ocr_out,  # Выход OCR
                    cls_targets.view(-1),  # Целевые классы (приводим к [batch_size])
                    bbox_targets,  # Целевые bbox
                    ocr_targets,  # Целевые тексты
                    ocr_lengths  # Длины текстов
                )
            
            # Backward pass с масштабированием градиентов
            scaler.scale(loss).backward()
            
            # Обрезка градиентов для стабильности
            clip_grad_norm_(model.parameters(), 1.0)
            
            # Шаг оптимизации
            scaler.step(optimizer)
            
            # Обновление масштаба
            scaler.update()
            
            # Логирование
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_cls.item():.4f}",
                "bbox": f"{loss_bbox.item():.4f}",
                "ocr": f"{loss_ocr.item():.4f}"
            })
        
        # ------------------------------
        # 6. Валидация и сохранение
        # ------------------------------
        avg_loss = epoch_loss / len(train_loader)
        
        # Обновление learning rate
        scheduler.step(avg_loss)
        
        # Сохранение лучшей модели
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, "best_model.pth")
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

# Точка входа (особенно важна для Windows)
if __name__ == "__main__":
    main()