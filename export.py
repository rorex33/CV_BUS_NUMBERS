import torch
import torch.nn as nn
from multitask_vehicle_model import MultiTaskModel
import yaml

def export_model():
    """Финальный исправленный экспорт модели"""
    # 1. Загрузка конфигурации и модели
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    model = MultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(config['model']['vocab']) + 1
    )
    
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu")['model_state_dict'])
    model.eval()

    # 2. Адаптер для фиксации размеров bbox_head
    class FixedBBoxHead(nn.Module):
        def __init__(self, original_head):
            super().__init__()
            self.conv = original_head[0]
            self.flatten = original_head[1]
            
            # Автоматический расчет размера
            with torch.no_grad():
                test_input = torch.randn(1, 512, 32, 32)
                features = self.conv(test_input)
                flat_features = self.flatten(features)
                in_features = flat_features.shape[1]
            
            self.linear = nn.Linear(in_features, 4)
            self.sigmoid = original_head[3]
            
        def forward(self, x):
            x = self.conv(x)
            x = self.flatten(x)
            x = self.linear(x)
            return self.sigmoid(x)

    # 3. Замена bbox_head
    model.bbox_head = FixedBBoxHead(model.bbox_head)

    # 4. Трассировка модели
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    
    # 5. Сохранение модели
    traced_model.save("model.pt")
    print("Модель успешно экспортирована в model.pt")

if __name__ == "__main__":
    export_model()