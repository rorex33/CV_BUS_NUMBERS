import cv2
import torch
import numpy as np
from multitask_vehicle_model import MultiTaskModel
import yaml

def decode_ctc(pred, idx2char, blank=0):
    """Улучшенное декодирование CTC с фильтрацией повторов и blank-символов"""
    pred = pred.argmax(dim=-1).squeeze()  # [T]
    result = []
    prev = blank
    for p in pred:
        if p != blank and p != prev:
            result.append(idx2char.get(p.item(), ''))
        prev = p
    return ''.join(result) or "Не распознано"

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    vocab = config['model']['vocab']
    idx2char = {i + 1: c for i, c in enumerate(vocab)}
    idx2char[0] = ''  # blank symbol для CTC

    model = MultiTaskModel(num_classes=5, ocr_vocab_size=len(vocab) + 1)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Препроцессинг изображения
        img = cv2.resize(frame, (config['image']['size'], config['image']['size']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        img_t = (img_t / 255.0).sub_(0.5).div_(0.5)  # Нормализация [-1, 1]

        with torch.no_grad():
            cls_out, bbox_out, ocr_out = model(img_t)

        # Обработка классификации
        cls_probs = cls_out.softmax(dim=1).squeeze()
        label = torch.argmax(cls_probs).item()
        prob = cls_probs[label].item()
        label_text = f"{config['classes'][label]}: {prob:.2f}" if label != 4 else "Неизвестный транспорт"

        # Обработка bbox
        bbox = bbox_out.squeeze().numpy()
        h, w = frame.shape[:2]
        cx, cy, bw, bh = bbox * [w, h, w, h]
        x1, y1, x2, y2 = map(int, [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])

        # Декодирование OCR
        ocr_conf = ocr_out.softmax(dim=-1).max().item()
        if ocr_conf < config['thresholds']['ocr_confidence']:
            ocr_text = "Низкая уверенность"
        else:
            ocr_text = decode_ctc(ocr_out, idx2char)

        # Визуализация
        color = (0, 255, 0) if label != 4 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, ocr_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()