import cv2
import torch
import torch.nn.functional as F
from multitask_vehicle_model import MultiTaskModel
import yaml

def decode_ctc(pred, vocab, blank_index=36):
    """Декодирование выхода OCR с учетом blank символа"""
    if pred.dim() == 3:  # Если размерность [B, T, C]
        pred = pred.squeeze(0)  # Удаляем batch dimension [T, C]
    pred = pred.argmax(dim=-1)  # Получаем индексы символов
    
    # Обрабатываем случай, когда pred - скаляр (0-d tensor)
    if pred.dim() == 0:
        pred = pred.unsqueeze(0)  # Преобразуем в 1-d tensor
    
    result = []
    prev = blank_index
    for p in pred:
        p = p.item()
        if p != blank_index and p != prev:
            result.append(vocab[p])
        prev = p
    return ''.join(result) if result else "Не распознано"

def main():
    # Загрузка конфигурации
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Параметры из конфига
    vocab = config['model']['vocab']
    img_size = config['image']['size']
    
    # Соответствие классов и их названий
    class_mapping = {
        0: "автобус",
        1: "троллейбус", 
        2: "трамвай",
        3: "маршрутка",
        4: "неизвестный"
    }

    # Инициализация модели
    model = MultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(vocab)+1
    )
    
    # Загрузка весов
    try:
        state_dict = torch.load("best_model.pth", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    model.eval()
    
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Препроцессинг
        img = cv2.resize(frame, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_t = (img_t - 0.5) / 0.5
        img_t = img_t.unsqueeze(0)

        # Инференс
        with torch.no_grad():
            cls_out, bbox_out, ocr_out = model(img_t)

        # Обработка классификации
        cls_probs = F.softmax(cls_out, dim=1).squeeze()
        label = torch.argmax(cls_probs).item()
        conf = cls_probs[label].item()
        
        # Применение порога уверенности
        if conf < 0.5:
            label = 4
        
        label_text = f"{class_mapping[label]}: {conf:.2f}"

        # Обработка bbox
        bbox = bbox_out.squeeze().cpu().numpy()
        h, w = frame.shape[:2]
        cx, cy, bw, bh = bbox * [w, h, w, h]
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)
        
        # Проверка валидности bbox
        bbox_valid = (0 <= x1 < x2 <= w) and (0 <= y1 < y2 <= h)
        
        # Обработка OCR
        try:
            ocr_text = decode_ctc(ocr_out, vocab) if bbox_valid else "Низкая уверенность"
        except Exception as e:
            print(f"Ошибка декодирования OCR: {e}")
            ocr_text = "Ошибка OCR"

        # Визуализация
        color = (0, 255, 0) if label != 4 else (0, 0, 255)
        
        if bbox_valid:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, ocr_text, (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, "Неверный bbox", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        ###
        print("\n--- Raw Model Output ---")
        print(f"Class probs: {cls_probs.detach().cpu().numpy()}")
        print(f"Predicted class: {label} ({class_mapping[label]})")
        print(f"BBox coords: {bbox}")
        print(f"OCR output shape: {ocr_out.shape}")

        debug_text = [
        f"Model debug:",
        f"Class: {label} ({class_mapping[label]}) Conf: {conf:.2f}",
        f"BBox: {x1},{y1}-{x2},{y2}",
        f"OCR: {ocr_text}"
        ]

        for i, text in enumerate(debug_text):
            cv2.putText(frame, text, (10, 30 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
        
        # Временно замените видеопоток на тестовое изображение
        test_img = cv2.imread("./data/train/images/89981.jpg")  # положите в папку с примером транспорта
        if test_img is not None:
            frame = cv2.resize(test_img, (img_size, img_size))
            print(">>> Using test image instead of camera feed!")
        ###
        
        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()