import torch
import torch.nn as nn
from multitask_vehicle_model import MultiTaskModel
import yaml
import os
import traceback

def export_with_scripting(
    config_path: str = "config.yaml",
    model_path: str = "best_model.pth",
    output_path: str = "script_model.pt",  # Рекомендуемое расширение для script
    img_size: int = 224,
    quantize: bool = True
):
    """Экспорт модели через torch.jit.script с полной проверкой"""
    print(f"=== TorchScript Scripting Export (PyTorch {torch.__version__}) ===")
    
    # 1. Загрузка конфигурации
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. Инициализация модели
    model = MultiTaskModel(
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
        # 5. Конвертация через scripting
        print("\n=== TorchScript Scripting ===")
        scripted_model = torch.jit.script(model)
        
        # 6. Проверка модели
        print("\n=== Model Validation ===")
        with torch.no_grad():
            # Проверка оригинальной модели
            orig_output = model(example_input)
            print("Original model outputs:")
            for i, out in enumerate(orig_output):
                print(f"Output {i}: shape={out.shape}")
            
            # Проверка scripted модели
            script_output = scripted_model(example_input)
            print("\nScripted model outputs:")
            for i, out in enumerate(script_output):
                print(f"Output {i}: shape={out.shape}")
            
            # Сравнение выходов
            for orig, script in zip(orig_output, script_output):
                if not torch.allclose(orig, script, atol=1e-4):
                    raise ValueError("Scripted model output doesn't match original!")
        
        print("\nValidation passed! Outputs match.")

        # 7. Оптимизация для мобильных устройств
        print("\n=== Mobile Optimization ===")
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
        
        # 8. Квантование (опционально)
        if quantize:
            print("\n=== Applying Quantization ===")
            quantized_model = torch.quantization.quantize_dynamic(
                optimized_model,
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM},
                dtype=torch.qint8
            )
            model_to_save = quantized_model
        else:
            model_to_save = optimized_model

        # 9. Сохранение модели
        print("\n=== Saving Model ===")
        model_to_save.save(output_path)
        
        # Проверка размера файла
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model saved to: {output_path}")
        print(f"Model size: {model_size:.2f} MB")
        
        # 10. Финальная проверка
        print("\n=== Final Verification ===")
        loaded_model = torch.jit.load(output_path)
        test_output = loaded_model(example_input)
        print("Load test successful!")
        
        return True

    except Exception as e:
        print(f"\n!!! Export failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Конфигурация экспорта
    export_config = {
        "config_path": "config.yaml",
        "model_path": "best_model.pth",
        "output_path": "script_model.pt",
        "img_size": 224,
        "quantize": True
    }
    
    if export_with_scripting(**export_config):
        print("\n=== Экспорт успешно завершен! ===")
        print("Инструкции:")
        print("1. Поместите файл script_model.pt в папку assets Android-проекта")
        print("2. Используйте следующий код для загрузки:")
        print("""
        try {
            model = LiteModuleLoader.loadModuleFromAsset(assets, "script_model.pt")
        } catch (e: Exception) {
            Log.e("Model", "Load error", e)
        }
        """)
    else:
        print("\n!!! Экспорт не удался !!!")
        print("Рекомендации:")
        print("1. Проверьте, содержит ли модель неподдерживаемые операторы")
        print("2. Упростите архитектуру модели при необходимости")
        print("3. Попробуйте использовать torch.jit.trace вместо scripting")