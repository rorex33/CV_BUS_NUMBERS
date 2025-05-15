import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from dataset import VehicleDataset, collate_fn
from multitask_vehicle_model import MultiTaskModel
import yaml
from tqdm import tqdm
import os


# Загрузка конфигурации
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = config['model']['vocab']
model = MultiTaskModel(num_classes=5, ocr_vocab_size=len(vocab) + 1).to(device)

# Датасет и DataLoader
dataset = VehicleDataset(root_dir="data", vocab=vocab, image_size=config['image']['size'])
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)

# Оптимизатор и AMP
optimizer = Adam(model.parameters(), lr=config['training']['lr'])
scaler = torch.amp.GradScaler('cuda')

# Тренировка
for epoch in range(config['training']['epochs']):
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_bbox = 0.0
    running_ocr = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}", leave=False)

    for images, cls_targets, bbox_targets, ocr_targets, ocr_lengths in progress_bar:
        images = images.to(device)
        cls_targets = cls_targets.to(device)
        bbox_targets = bbox_targets.to(device)
        ocr_targets = ocr_targets.to(device)
        input_lengths = torch.full((images.size(0),), ocr_targets.size(1), dtype=torch.long).to(device)

        with torch.amp.autocast('cuda'):
            cls_out, bbox_out, ocr_out = model(images)
            loss, loss_cls, loss_bbox, loss_ocr = model.compute_losses(
                cls_out, cls_targets,
                bbox_out, bbox_targets,
                ocr_out, ocr_targets,
                ocr_lengths
            )

        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Сбор статистики
        running_loss += loss.item()
        running_cls += loss_cls.item()
        running_bbox += loss_bbox.item()
        running_ocr += loss_ocr.item()

        # Обновление прогресс-бара
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "CLS": f"{loss_cls.item():.4f}",
            "BBOX": f"{loss_bbox.item():.4f}",
            "OCR": f"{loss_ocr.item():.4f}"
        })

    print(f"[Epoch {epoch + 1}] Total: {running_loss:.4f} | CLS: {running_cls:.4f} | BBOX: {running_bbox:.4f} | OCR: {running_ocr:.4f}")