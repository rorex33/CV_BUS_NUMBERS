import torch
from multitask_vehicle_model import MultiTaskModel
import yaml
import os
import traceback
from typing import Tuple

def export_with_proper_scripting(
    config_path: str = "config.yaml",
    model_path: str = "best_model.pth",
    output_path: str = "script_model.pt",
    img_size: int = 224,
    quantize: bool = True
):
    """Экспорт модели с правильным использованием torch.jit.script"""
    print(f"=== Scripting Export (PyTorch {torch.__version__}) ===")
    
    # 1. Загрузка конфигурации
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. Модифицируем модель для совместимости
    class ScriptableModel(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Явно преобразуем выход в кортеж
            outputs = self.model(x)
            if isinstance(outputs, torch.Tensor):
                return outputs, torch.zeros_like(outputs), torch.zeros_like(outputs)
            elif len(outputs) == 3:
                return tuple(outputs)
            else:
                raise ValueError("Model must return 1 or 3 tensors")

    # 3. Инициализация модели
    original_model = MultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(config['model']['vocab']) + 1
    )
    original_model.load_state_dict(torch.load(model_path, map_location="cpu")['model_state_dict'])
    original_model.eval()
    
    model = ScriptableModel(original_model)
    model.eval()

    # 4. Пример ввода
    example_input = torch.randn(1, 3, img_size, img_size)
    
    try:
        # 5. Проверка работы модели
        print("\n=== Model Test ===")
        with torch.no_grad():
            out1, out2, out3 = model(example_input)
            print(f"Output shapes: {out1.shape}, {out2.shape}, {out3.shape}")

        # 6. Scripting с явными аннотациями
        print("\n=== Applying TorchScript ===")
        scripted_model = torch.jit.script(model)
        
        # 7. Проверка scripted модели
        print("\n=== Script Validation ===")
        with torch.no_grad():
            script_out = scripted_model(example_input)
            if not all(torch.allclose(o1, o2) for o1, o2 in zip(model(example_input), script_out)):
                raise ValueError("Scripted model outputs don't match!")
            print("Validation passed!")

        # 8. Оптимизация для мобильных
        print("\n=== Mobile Optimization ===")
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
        
        # 9. Квантование
        if quantize:
            print("\n=== Quantization ===")
            quantized_model = torch.quantization.quantize_dynamic(
                optimized_model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            model_to_save = quantized_model
        else:
            model_to_save = optimized_model

        # 10. Сохранение
        print("\n=== Saving Model ===")
        model_to_save.save(output_path)
        print(f"Model saved to {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")
        
        return True

    except Exception as e:
        print(f"\n!!! Export failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if export_with_proper_scripting():
        print("\n=== Успешный экспорт! ===")
        print("Инструкции по использованию в Android:")
        print("1. Поместите script_model.pt в папку assets")
        print("2. Используйте следующий код для загрузки:")
        print("""
        try {
            // Загрузка модели
            Module model = LiteModuleLoader.loadModuleFromAsset(assets, "script_model.pt");
            
            // Подготовка входа
            float[] mean = {0.485f, 0.456f, 0.406f};
            float[] std = {0.229f, 0.224f, 0.225f};
            Tensor input = TensorImageUtils.bitmapToFloat32Tensor(bitmap, mean, std);
            
            // Получение выходов
            IValue outputs = model.forward(IValue.from(input));
            Tuple<Tensor, Tensor, Tensor> result = outputs.toTuple();
            
            // Обработка результатов
            Tensor classes = result.get(0).toTensor();
            Tensor boxes = result.get(1).toTensor();
            Tensor text = result.get(2).toTensor();
        } catch (Exception e) {
            Log.e("Model", "Error", e);
        }
        """)
    else:
        print("\nЭкспорт не удался. Возможные решения:")
        print("1. Упростите архитектуру модели")
        print("2. Проверьте типы возвращаемых значений")
        print("3. Попробуйте torch.jit.trace вместо script")