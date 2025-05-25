import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from multitask_vehicle_model import MultiTaskModel
import yaml
import numpy as np
import os

def export_model_for_lite(
    config_path: str = "config.yaml",
    model_path: str = "best_model.pth",
    output_path: str = "model.ptl",  # Изменили расширение на .ptl для lite
    image_size: int = None,
    quantize: bool = True  # Добавили флаг квантования
):
    """Экспорт модели для PyTorch Lite с оптимизацией
    
    Args:
        config_path: Путь к конфигурационному файлу
        model_path: Путь к сохраненной модели
        output_path: Куда сохранить экспортированную модель
        image_size: Опциональное указание размера изображения
        quantize: Флаг для включения квантования
    """
    # Проверка версии PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Preparing model for PyTorch Lite...")
    
    # Загрузка конфигурации
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    vocab = config['model']['vocab']
    img_size = image_size or config['image']['size']
    
    # Инициализация модели
    model = MultiTaskModel(num_classes=5, ocr_vocab_size=len(vocab)+1)
    
    # Загрузка весов
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Подготовка примера ввода
    example_input = torch.randn(1, 3, img_size, img_size)
    
    try:
        # 1. Проверка работы модели
        print("\n=== Проверка модели ===")
        with torch.no_grad():
            cls_out, bbox_out, ocr_out = model(example_input)
            print("Входные размеры:", example_input.shape)
            print("Выходные размеры:")
            print(f"  Классы: {cls_out.shape}")
            print(f"  BBox: {bbox_out.shape}")
            print(f"  OCR: {ocr_out.shape}")
        
        # 2. Трассировка модели
        print("\n=== Трассировка модели ===")
        traced_model = torch.jit.trace(model, example_input, check_trace=True)
        
        # 3. Оптимизация для мобильных устройств
        print("\n=== Оптимизация для мобильных устройств ===")
        optimized_model = optimize_for_mobile(
            traced_model,
            optimization_blocklist={
                torch.nn.Linear,  # Блокируем оптимизацию Linear слоев
                torch.nn.LSTM     # Блокируем оптимизацию LSTM
            }
        )
        
        # 4. Квантование (если включено)
        if quantize:
            print("\n=== Квантование модели ===")
            quantized_model = torch.quantization.quantize_dynamic(
                optimized_model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            model_to_save = quantized_model
        else:
            model_to_save = optimized_model
        
        # 5. Проверка оптимизированной модели
        print("\n=== Проверка оптимизированной модели ===")
        with torch.no_grad():
            test_output = model_to_save(example_input)
            assert len(test_output) == 3, "Модель должна возвращать 3 выхода"
            print("Проверка пройдена успешно!")
        
        # 6. Сохранение модели
        print("\n=== Сохранение модели ===")
        model_to_save.save(output_path)
        
        # Проверка размера файла
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Модель сохранена в {output_path}")
        print(f"Размер модели: {model_size:.2f} MB")
        
        # 7. Проверка загрузки
        print("\n=== Тест загрузки модели ===")
        loaded_model = torch.jit.load(output_path)
        print("Модель успешно загружена!")
        
        return True
        
    except Exception as e:
        print(f"\n!!! Ошибка экспорта: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Настройки экспорта
    export_config = {
        "config_path": "config.yaml",
        "model_path": "best_model.pth",
        "output_path": "model.ptl",  # Используем .ptl для lite-моделей
        "quantize": True             # Включаем квантование
    }
    
    success = export_model_for_lite(**export_config)
    
    if success:
        print("\nЭкспорт успешно завершен! Модель готова для использования в Android.")
    else:
        print("\nЭкспорт не удался. Проверьте ошибки выше.")