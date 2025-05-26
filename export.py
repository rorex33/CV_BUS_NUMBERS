import torch
from multitask_vehicle_model import MultiTaskModel
import yaml

def simple_export(
    config_path: str = "config.yaml",
    model_path: str = "best_model.pth",
    output_path: str = "model.pt",
    img_size: int = 224
):
    """Минималистичный экспорт в TorchScript"""
    print("=== Простой экспорт модели ===")
    
    # 1. Загрузка модели
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = MultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(config['model']['vocab']) + 1
    )
    
    model.load_state_dict(torch.load(model_path, map_location="cpu")['model_state_dict'])
    model.eval()

    # 2. Трассировка без лишних проверок
    example_input = torch.randn(1, 3, img_size, img_size)
    traced_model = torch.jit.trace(model, example_input)
    
    # 3. Сохранение
    traced_model.save(output_path)
    print(f"Модель сохранена в {output_path}")

if __name__ == "__main__":
    simple_export()