import torch
from multitask_vehicle_model import MultiTaskModel
import yaml
import numpy as np

def export_model(
    config_path: str = "config.yaml",
    model_path: str = "model.pth",
    output_path: str = "multitask_model.pt",
    image_size: int = None
):
    """Экспорт модели с валидацией
    
    Args:
        config_path: Путь к конфигурационному файлу
        model_path: Путь к сохраненной модели
        output_path: Куда сохранить экспортированную модель
        image_size: Опциональное указание размера изображения
    """
    # Загрузка конфигурации
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    vocab = config['model']['vocab']
    img_size = image_size or config['image']['size']
    
    # Инициализация модели
    model = MultiTaskModel(num_classes=5, ocr_vocab_size=len(vocab)+1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Подготовка примера ввода
    example_input = torch.randn(1, 3, img_size, img_size)
    
    try:
        # Валидация модели перед экспортом
        with torch.no_grad():
            cls_out, bbox_out, ocr_out = model(example_input)
            print("Проверка прямого прохода:")
            print(f"Классы: {cls_out.shape}, BBox: {bbox_out.shape}, OCR: {ocr_out.shape}")
        
        # Трассировка
        traced_model = torch.jit.trace(model, example_input, check_trace=True)
        
        # Дополнительная валидация
        test_output = traced_model(example_input)
        assert np.allclose(cls_out.numpy(), test_output[0].numpy(), atol=1e-5)
        
        # Сохранение
        traced_model.save(output_path)
        print(f"✅ Модель успешно экспортирована в {output_path}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка экспорта: {str(e)}")
        return False

if __name__ == "__main__":
    export_model()