import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multitask_vehicle_model import MultiTaskModel
import torch.nn.functional as F

def test_model(model, device, config):
    """Тестирование всех компонентов модели"""
    # 1. Создаем тестовое изображение
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    cv2.putText(img, "AB1234", (150, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    cv2.rectangle(img, (100, 200), (400, 300), (0,255,0), 2)
    
    # 2. Преобразуем в тензор
    img_tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 3. Переводим модель в режим оценки
    model.eval()
    
    # 4. Полный тест модели
    with torch.no_grad():
        # Прямой проход
        cls_out, bbox_out, ocr_out = model(img_tensor)
        
        # Проверка классификации
        cls_probs = F.softmax(cls_out, dim=1)
        print("\n=== Классификация транспорта ===")
        print(f"Вероятности классов: {cls_probs.cpu().numpy()[0]}")
        print(f"Предсказанный класс: {torch.argmax(cls_probs).item()}")
        
        # Проверка BBox
        print("\n=== Детекция номера ===")
        bbox_coords = bbox_out.cpu().numpy()[0]
        print(f"Координаты bbox (нормализованные): {bbox_coords}")
        
        # Проверка на валидность bbox
        if np.any(bbox_coords < 0) or np.any(bbox_coords > 1):
            print("Ошибка: Координаты bbox вне диапазона [0, 1]!")
        
        # Визуализация BBox
        h, w = img.shape[:2]
        x, y, bw, bh = bbox_coords
        x1, y1 = int((x - bw/2)*w), int((y - bh/2)*h)
        x2, y2 = int((x + bw/2)*w), int((y + bh/2)*h)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
        
        # Проверка OCR
        print("\n=== Распознавание текста ===")
        print(f"Размер выхода OCR: {ocr_out.shape}")
        
        # Анализ выходов OCR
        probs = F.softmax(ocr_out, dim=2)
        top_probs, top_chars = torch.topk(probs, 3, dim=2)
        
        print("\nТоп-3 предсказания для первых 5 позиций:")
        for i in range(5):
            chars = [config['model']['vocab'][idx] if idx < len(config['model']['vocab']) else '#' 
                    for idx in top_chars[0,i]]
            print(f"Позиция {i}: {chars} с вероятностями {top_probs[0,i].cpu().numpy()}")
        
        # Декодирование с разными порогами
        for threshold in [0.1, 0.3, 0.5]:
            text = decode_ctc(ocr_out, config['model']['vocab'], threshold)
            print(f"Порог {threshold}: '{text}'")
    
    # 5. Визуализация результатов
    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Результаты теста")
    plt.axis('off')
    plt.show()

def decode_ctc(pred, vocab, threshold=0.3):
    """Улучшенный декодер с порогом уверенности"""
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    
    probs = F.softmax(pred, dim=-1)
    top_probs, top_chars = torch.max(probs, dim=-1)
    
    result = []
    for i in range(len(top_chars)):
        if top_probs[i] > threshold and top_chars[i] < len(vocab):
            result.append(vocab[top_chars[i]])
    
    return ''.join(result) if result else "Не распознано"

def load_and_test(config_path="config.yaml", model_path="best_model.pth"):
    """Загрузка модели и запуск тестов"""
    # Загрузка конфига
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Инициализация устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Создание модели
    model = MultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(config['model']['vocab']) + 1
    ).to(device)
    
    # Загрузка весов
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из {model_path}")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return
    
    # Запуск тестов
    test_model(model, device, config)
    
    # Дополнительные проверки
    print("\n=== Проверка архитектуры ===")
    print(f"OCR слой: {model.ocr_fc.weight.shape}")  # Должно быть [23, 32]
    
    # Проверка на случайных данных
    dummy = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        cls, bbox, ocr = model(dummy)
        print("\nТест на случайных данных:")
        print(f"Классификация: {cls.shape}, BBox: {bbox.shape}, OCR: {ocr.shape}")

if __name__ == "__main__":
    import yaml
    load_and_test()