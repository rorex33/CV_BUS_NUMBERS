import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn.utils import clip_grad_norm_
from dataset import VehicleDataset, collate_fn
from multitask_vehicle_model import MultiTaskModel
import yaml
from tqdm import tqdm
import os
import numpy as np

# Загрузка конфигурации
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Инициализация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = config['model']['vocab']
model = MultiTaskModel(num_classes=5, ocr_vocab_size=len(vocab) + 1).to(device)

# Датасет и DataLoader (с правильным collate_fn)
def collate_fn(batch):
    """Обрабатывает батч с учетом паддинга для OCR"""
    images, cls_targets, bbox_targets, ocr_targets = zip(*batch)
    
    # Стек изображений и bbox
    images = torch.stack(images)
    cls_targets = torch.tensor(cls_targets)
    bbox_targets = torch.stack(bbox_targets)
    
    # Паддинг для OCR (0 = blank для CTC)
    max_len = max(len(t) for t in ocr_targets)
    padded_ocr = torch.zeros((len(batch), max_len), dtype=torch.long)
    ocr_lengths = torch.tensor([len(t) for t in ocr_targets], dtype=torch.long)
    for i, t in enumerate(ocr_targets):
        padded_ocr[i, :len(t)] = torch.tensor(t)
    
    return images, cls_targets, bbox_targets, padded_ocr, ocr_lengths

dataset = VehicleDataset(root_dir="data", vocab=vocab, image_size=config['image']['size'])
dataloader = DataLoader(dataset, 
                       batch_size=config['training']['batch_size'],
                       shuffle=True, 
                       collate_fn=collate_fn,
                       num_workers=4)

# Оптимизатор и шедулер
optimizer = Adam(model.parameters(), lr=config['training']['lr'])
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
scaler = torch.cuda.amp.GradScaler()

# Функция валидации
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, cls_targets, bbox_targets, ocr_targets, ocr_lengths in val_loader:
            images = images.to(device)
            cls_targets = cls_targets.to(device)
            bbox_targets = bbox_targets.to(device)
            ocr_targets = ocr_targets.to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                cls_out, bbox_out, ocr_out = model(images)
                loss, _, _, _ = model.compute_losses(
                    cls_out, cls_targets,
                    bbox_out, bbox_targets,
                    ocr_out, ocr_targets,
                    ocr_lengths
                )
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Тренировка
best_val_loss = np.inf
for epoch in range(config['training']['epochs']):
    model.train()
    epoch_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}")
    for images, cls_targets, bbox_targets, ocr_targets, ocr_lengths in progress_bar:
        images = images.to(device)
        cls_targets = cls_targets.to(device)
        bbox_targets = bbox_targets.to(device)
        ocr_targets = ocr_targets.to(device)
        
        # Обнуление градиентов
        optimizer.zero_grad()
        
        # Forward pass с mixed precision
        with torch.amp.autocast(device_type='cuda'):
            cls_out, bbox_out, ocr_out = model(images)
            loss, loss_cls, loss_bbox, loss_ocr = model.compute_losses(
                cls_out, cls_targets,
                bbox_out, bbox_targets,
                ocr_out, ocr_targets,
                ocr_lengths  # Уже содержит фактические длины
            )
        
        # Backward pass
        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Логирование
        epoch_loss += loss.item()
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "CLS": f"{loss_cls.item():.4f}",
            "BBOX": f"{loss_bbox.item():.4f}",
            "OCR": f"{loss_ocr.item():.4f}"
        })
    
    # Валидация и сохранение модели
    val_loss = validate(model, dataloader, device)
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print(f"New best model saved (Val Loss: {val_loss:.4f})")
    
    print(f"Epoch {epoch + 1} | Train Loss: {epoch_loss / len(dataloader):.4f} | Val Loss: {val_loss:.4f}")