import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def encode_label(text, vocab):
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}
    return [char2idx.get(c, 0) for c in text]


def collate_fn(batch):
    images, cls_targets, bbox_targets, ocr_targets, ocr_lengths = zip(*batch)
    images = torch.stack(images)
    cls_targets = torch.tensor(cls_targets, dtype=torch.long)
    bbox_targets = torch.stack(bbox_targets)
    max_len = max(ocr_lengths)

    padded_ocr = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(ocr_targets):
        padded_ocr[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return images, cls_targets, bbox_targets, padded_ocr, torch.tensor(ocr_lengths, dtype=torch.long)


class VehicleDataset(Dataset):
    def __init__(self, root_dir, vocab, image_size=512):
        self.root_dir = root_dir
        self.image_size = image_size
        self.vocab = vocab
        self.data = []
        self._load_data()

    def _load_data(self):
        for line in open(os.path.join(self.root_dir, "labels.txt")):
            img_name, cls, x, y, w, h, text = line.strip().split(',')
            self.data.append((img_name, int(cls), float(x), float(y), float(w), float(h), text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, cls, x, y, w, h, text = self.data[idx]
        img_path = os.path.join(self.root_dir, "images", img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)

        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)
        label_seq = encode_label(text, self.vocab)

        return image, cls, bbox, label_seq, len(label_seq)
