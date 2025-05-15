import cv2
import torch
import numpy as np
from multitask_vehicle_model import MultiTaskModel
from dataset import encode_label
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

vocab = config['model']['vocab']
idx2char = {i + 1: c for i, c in enumerate(vocab)}
model = MultiTaskModel(num_classes=5, vocab_size=len(vocab) + 1)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

def decode_ctc(pred):
    pred = pred.argmax(dim=2).squeeze().tolist()
    result = []
    prev = -1
    for p in pred:
        if p != 0 and p != prev:
            result.append(idx2char.get(p, ''))
        prev = p
    return ''.join(result)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (512, 512))
    img_t = torch.tensor(img.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        cls_out, bbox_out, ocr_out = model(img_t)

    cls = cls_out.softmax(dim=1).squeeze()
    label = torch.argmax(cls).item()
    prob = cls[label].item()

    if label == 4 and prob > config['thresholds']['unknown_cls']:
        print("Неизвестный транспорт")
    else:
        print(f"Класс: {label}, уверенность: {prob:.2f}")

    bbox = bbox_out.squeeze().numpy()
    h, w = frame.shape[:2]
    cx, cy, bw, bh = bbox * [w, h, w, h]
    x1, y1, x2, y2 = map(int, [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    ocr_text = decode_ctc(ocr_out)
    print("Номер:", ocr_text)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
