import torch
from multitask_vehicle_model import MultiTaskModel
import yaml
import os
import traceback
from typing import List, Tuple

def export_model(
    config_path: str = "config.yaml",
    model_path: str = "best_model.pth",
    output_path: str = "mobile_model.pt",
    img_size: int = 224,
    quantize: bool = True
):
    """Исправленный экспорт модели с обработкой adaptive pooling"""
    print(f"=== TorchScript Export (PyTorch {torch.__version__}) ===")
    
    # 1. Загрузка конфигурации
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. Инициализация модели с фиксированным output_size
    class FixedMultiTaskModel(MultiTaskModel):
        def __init__(self, num_classes: int, ocr_vocab_size: int):
            super().__init__(num_classes, ocr_vocab_size)
            # Заменяем adaptive pooling на fixed-size
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Фиксированный размер выхода
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return super().forward(x)
    
    model = FixedMultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(config['model']['vocab']) + 1
    )
    
    # 3. Загрузка весов
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    # 4. Подготовка примера ввода
    example_input = torch.randn(1, 3, img_size, img_size)
    
    try:
        # 5. Конвертация через scripting с аннотациями типов
        print("\n=== TorchScript Conversion ===")
        
        # Аннотируем все используемые типы
        @torch.jit.script
        def script_wrapper(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return model(x)
        
        # 6. Проверка модели
        print("\n=== Model Validation ===")
        with torch.no_grad():
            orig_out = model(example_input)
            script_out = script_wrapper(example_input)
            
            for i, (o, s) in enumerate(zip(orig_out, script_out)):
                if not torch.allclose(o, s, atol=1e-4):
                    raise ValueError(f"Output {i} mismatch!")
            print("Validation passed!")

        # 7. Квантование
        if quantize:
            print("\n=== Applying Quantization ===")
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            # Повторная трассировка после квантования
            traced_model = torch.jit.trace(quantized_model, example_input)
            model_to_save = traced_model
        else:
            model_to_save = torch.jit.trace(model, example_input)

        # 8. Сохранение модели
        print("\n=== Saving Model ===")
        model_to_save.save(output_path)
        
        # 9. Проверка размера
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model saved ({size_mb:.2f} MB)")
        
        return True

    except Exception as e:
        print(f"\n!!! Export failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if export_model():
        print("\nЭкспорт успешен! Используйте mobile_model.pt в Android")
    else:
        print("\nЭкспорт не удался. Проверьте ошибки выше.")