import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from dataset import VehicleDataset, collate_fn
from multitask_vehicle_model import MultiTaskModel
import yaml
import os


with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = config['model']['vocab']
model = MultiTaskModel(num_classes=5, ocr_vocab_size=len(vocab) + 1).to(device)

dataset = VehicleDataset(root_dir="data", vocab=vocab, image_size=config['image']['size'])
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)

optimizer = Adam(model.parameters(), lr=config['training']['lr'])
scaler = torch.amp.GradScaler('cuda')

for epoch in range(config['training']['epochs']):
    model.train()
    for images, cls_targets, bbox_targets, ocr_targets, ocr_lengths in dataloader:
        images = images.to(device)
        cls_targets = cls_targets.to(device)
        bbox_targets = bbox_targets.to(device)
        ocr_targets = ocr_targets.to(device)
        input_lengths = torch.full((images.size(0),), ocr_targets.size(1), dtype=torch.long).to(device)

        with torch.amp.autocast('cuda'):
            cls_out, bbox_out, ocr_out = model(images)
            loss, loss_cls, loss_bbox, loss_ocr = model.compute_losses(cls_out, cls_targets, bbox_out, bbox_targets, ocr_out, ocr_targets, ocr_lengths)


        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Total Loss: {loss.item():.4f}, CLS: {loss_cls.item():.4f}, BBOX: {loss_bbox.item():.4f}, OCR: {loss_ocr.item():.4f}")
