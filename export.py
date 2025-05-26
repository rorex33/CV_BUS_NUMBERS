import torch
from multitask_vehicle_model import MultiTaskModel
import yaml
import os
import traceback

def export_pure_torchscript(
    config_path: str = "config.yaml",
    model_path: str = "best_model.pth",
    output_path: str = "model.pt",
    img_size: int = 224
):
    """Чистый экспорт в TorchScript без дополнительных оптимизаций"""
    print(f"=== Pure TorchScript Export (PyTorch {torch.__version__}) ===")
    
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
        # 5. Проверка прямого прохода
        print("\n=== Forward Pass Test ===")
        with torch.no_grad():
            outputs = model(example_input)
            if isinstance(outputs, torch.Tensor):
                print(f"Model returns single tensor: {outputs.shape}")
            else:
                print(f"Model returns {len(outputs)} outputs:")
                for i, out in enumerate(outputs):
                    print(f"Output {i}: {out.shape}")

        # 6. Создаем адаптер для гарантированного формата вывода
        class ModelWrapper(torch.nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model
            
            def forward(self, x):
                result = self.model(x)
                # Гарантируем возврат 3 тензоров
                if isinstance(result, torch.Tensor):
                    return (result, torch.tensor(0.0), torch.tensor(0.0))
                elif len(result) == 2:
                    return (*result, torch.tensor(0.0))
                return result

        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()

        # 7. Трассировка модели
        print("\n=== Tracing Model ===")
        traced_model = torch.jit.trace(wrapped_model, example_input)
        
        # 8. Проверка трассированной модели
        print("\n=== Trace Validation ===")
        with torch.no_grad():
            traced_outputs = traced_model(example_input)
            print("Traced model outputs:")
            for i, out in enumerate(traced_outputs):
                print(f"Output {i}: {out.shape}")

        # 9. Сохранение модели
        print("\n=== Saving Model ===")
        traced_model.save(output_path)
        print(f"Model saved to {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")
        
        return True

    except Exception as e:
        print(f"\n!!! Export failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if export_pure_torchscript():
        print("\n=== Export Successful ===")
        print("Next steps:")
        print("1. Copy model.pt to Android app's assets folder")
        print("2. Use LiteModuleLoader for loading:")
        print("""
        try {
            Module model = LiteModuleLoader.loadModuleFromAsset(assets, "model.pt");
            // Input tensor must be ${img_size}x${img_size}
            IValue outputs = model.forward(IValue.from(inputTensor));
            // Process outputs...
        } catch (Exception e) {
            Log.e("Model", "Error", e);
        }
        """)
    else:
        print("\n!!! Export Failed !!!")
        print("Debugging tips:")
        print("1. Check model architecture for dynamic control flow")
        print("2. Verify input tensor shape matches model expectations")
        print("3. Try simpler model version first")