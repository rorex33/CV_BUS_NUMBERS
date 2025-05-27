import torch
from multitask_vehicle_model import MultiTaskModel
from dataset import VehicleDataset  # Импортируем для доступа к vocab

# 1. Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModel(
    num_classes=5,  # Должно соответствовать обученной модели
    ocr_vocab_size=len("0123456789АВЕКМНОРСТУХ") + 1  # +1 для blank символа
).to(device)

# 2. Загрузка весов (если есть)
try:
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки весов: {e}")
    print("Будет использована модель со случайными весами")

# 3. Подготовка словаря OCR
vocab = "0123456789АВЕКМНОРСТУХ"  # Должен совпадать с обучающим

# 4. Функция декодирования OCR (аналогичная вашей)
def decode_ctc(pred, vocab, blank_index=36):
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    pred = pred.argmax(dim=-1)
    
    result = []
    prev = blank_index
    for p in pred:
        p = p.item()
        if p != blank_index and p != prev:
            result.append(vocab[p])
        prev = p
    return ''.join(result) if result else "Не распознано"

# 5. Тестовый прогон
model.eval()  # Переводим модель в режим оценки

# Создаем тестовый тензор (1 изображение, 3 канала, 512x512)
dummy_input = torch.randn(1, 3, 512, 512).to(device)

with torch.no_grad():
    cls_out, bbox_out, ocr_out = model(dummy_input)
    
    # Анализ результатов
    print("\n=== Результаты теста ===")
    print(f"Входной тензор shape: {dummy_input.shape}")
    print(f"Входные значения диапазон: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    
    # Классификация
    cls_probs = torch.softmax(cls_out, dim=1)
    print(f"\nКлассификация (shape: {cls_out.shape}):")
    print(f"Сырые выходы: {cls_out.cpu().numpy()}")
    print(f"Вероятности: {cls_probs.cpu().numpy()}")
    print(f"Предсказанный класс: {cls_out.argmax().item()}")
    
    # BBox
    print(f"\nBBox (shape: {bbox_out.shape}):")
    print(f"Координаты: {bbox_out.squeeze().cpu().numpy()}")
    print("Проверка диапазона (должен быть [0, 1]):", 
          torch.all((bbox_out >= 0) & (bbox_out <= 1)).item())
    
    # OCR
    print(f"\nOCR (shape: {ocr_out.shape}):")
    print(f"Пример выхода: {ocr_out[0, :3, :5].cpu().numpy()}")  # Первые 3 символа
    print("Декодированный текст:", decode_ctc(ocr_out, vocab))

# 6. Дополнительные проверки
print("\n=== Проверка архитектуры ===")
print("Модель на устройстве:", next(model.parameters()).device)
print("Количество параметров:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))