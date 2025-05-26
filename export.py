import torch
from multitask_vehicle_model import MultiTaskModel
import yaml

def export_model():
    """
    Экспорт multitask-модели в формат TorchScript.
    
    Процесс включает:
    1. Загрузку конфигурации модели из YAML-файла
    2. Инициализацию модели и загрузку обученных весов
    3. Компиляцию модели с использованием torch.jit.script
    4. Сохранение скомпилированной модели в файл
    
    Сохраняет:
    - model.pt: TorchScript-совместимая версия модели
    
    Raises:
        FileNotFoundError: Если отсутствуют конфигурационный файл или файл весов
        RuntimeError: При проблемах с загрузкой весов или компиляцией модели
    """
    # 1. Загрузка конфигурации
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # 2. Инициализация и загрузка весов
    model = MultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(config['model']['vocab']) + 1  # +1 для CTC blank
    )
    checkpoint = torch.load("best_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Компиляция в TorchScript (используем scripting вместо tracing!)
    scripted_model = torch.jit.script(model)

    # 4. Сохранение модели
    scripted_model.save("model.pt")
    print("TorchScript-модель успешно сохранена в model.pt")

if __name__ == "__main__":
    export_model()