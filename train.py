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

def main():
    # Загрузка конфигурации
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация модели
    model = MultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(config['model']['vocab']) + 1
    ).to(device)
    
    # Датасеты и DataLoader (с уменьшенным num_workers для Windows)
    train_dataset = VehicleDataset(
        root_dir="data/train",
        vocab=config['model']['vocab'],
        image_size=config['image']['size']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Для Windows рекомендуется 0
        pin_memory=True
    )

    # Оптимизатор и шедулер
    optimizer = Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Исправленный инициализатор GradScaler
    scaler = torch.amp.GradScaler(device_type='cuda' if torch.cuda.is_available() else 'cpu')

    # Тренировочный цикл
    best_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch in progress_bar:
            images, cls_targets, bbox_targets, ocr_targets, ocr_lengths = [
                x.to(device, non_blocking=True) for x in batch
            ]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss, loss_cls, loss_bbox, loss_ocr = model.compute_losses(
                    *outputs,
                    cls_targets,
                    bbox_targets,
                    ocr_targets,
                    ocr_lengths
                )
            
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_cls.item():.4f}",
                "bbox": f"{loss_bbox.item():.4f}",
                "ocr": f"{loss_ocr.item():.4f}"
            })
        
        # Валидация и сохранение модели
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, "best_model.pth")
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    # Для Windows важно использовать __name__ == "__main__"
    main()