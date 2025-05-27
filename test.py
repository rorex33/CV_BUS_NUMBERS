import torch
from multitask_vehicle_model import MultiTaskModel
import torch.nn as nn

# 1. Конфигурация (должна совпадать с обучением)
VOCAB = "0123456789ABEKMHOPCTYX"  # 22 символа
OCR_VOCAB_SIZE = len(VOCAB) + 1  # 23 (22 символа + blank)
BLANK_INDEX = OCR_VOCAB_SIZE - 1  # blank = 22

# 2. Инициализация модели (с проверкой)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModel(
    num_classes=5,
    ocr_vocab_size=OCR_VOCAB_SIZE  # Важно! Должно быть 23
).to(device)

# 3. Улучшенная загрузка весов
try:
    checkpoint = torch.load("best_model.pth", map_location=device)
    
    # Проверка совместимости
    if checkpoint.get('config', {}).get('model', {}).get('vocab', '') != VOCAB:
        print("Предупреждение: Словарь в чекпоинте не совпадает с текущим!")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Модель загружена | Словарь: {VOCAB} | Размер: {OCR_VOCAB_SIZE}")
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    print("Инициализация новых весов...")
    # Инициализация весов для OCR слоя
    nn.init.xavier_uniform_(model.ocr_fc.weight)
    nn.init.zeros_(model.ocr_fc.bias)

# 4. Исправленная функция декодирования
def decode_ctc(pred, vocab, blank_index=BLANK_INDEX):  # Используем глобальный blank_index
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    pred = pred.argmax(dim=-1)
    
    result = []
    prev = blank_index
    for p in pred:
        p = p.item()
        if p != blank_index and p != prev:
            if p < len(vocab):  # Защита от выхода за границы
                result.append(vocab[p])
        prev = p
    return ''.join(result) if result else "Не распознано"

# 5. Проверочный прогон
model.eval()
dummy_input = torch.randn(1, 3, 512, 512).to(device)

print("\n=== Тест модели ===")
print(f"Ожидаемый размер словаря: {OCR_VOCAB_SIZE}")
print(f"Фактический ocr_fc.weight: {model.ocr_fc.weight.shape}")  # Должно быть [23, 32]

with torch.no_grad():
    cls, bbox, ocr = model(dummy_input)
    print("\nВыходы модели:")
    print(f"OCR shape: {ocr.shape}")  # Должно быть [1, W, 23]
    
    # Проверка диапазона выходов
    print("\nПроверка выходов OCR:")
    print(f"Мин: {ocr.min().item():.3f} Макс: {ocr.max().item():.3f}")
    print(f"Blank индекс: {BLANK_INDEX}")
    
    # Декодирование
    print("\nДекодированный текст:", decode_ctc(ocr, VOCAB))

# 6. Валидация архитектуры
assert model.ocr_fc.out_features == OCR_VOCAB_SIZE, \
    f"Несоответствие! Модель ожидает {model.ocr_fc.out_features}, а должно быть {OCR_VOCAB_SIZE}"