import torch
from multitask_vehicle_model import MultiTaskModel
import yaml

def export_model():
    """Исправленный экспорт модели с учетом архитектуры"""
    # 1. Загрузка модели
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    model = MultiTaskModel(
        num_classes=5,
        ocr_vocab_size=len(config['model']['vocab']) + 1
    )
    
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu")['model_state_dict'])
    model.eval()

    # 2. Фиксируем размеры для bbox_head
    class FixedBBoxHead(nn.Module):
        def __init__(self, original_head):
            super().__init__()
            self.conv = original_head[0]  # Conv2d
            self.flatten = original_head[1]  # Flatten
            # Пересчитываем размер для Linear
            with torch.no_grad():
                test_input = torch.randn(1, 512, 32, 32)
                features = self.conv(test_input)  # [1, 64, 32, 32]
                flat_features = self.flatten(features)  # [1, 64*32*32]
            self.linear = nn.Linear(flat_features.shape[1], 4)
            self.sigmoid = original_head[3]  # Sigmoid
            
        def forward(self, x):
            x = self.conv(x)
            x = self.flatten(x)
            x = self.linear(x)
            return self.sigmoid(x)

    # Заменяем bbox_head
    model.bbox_head = FixedBBoxHead(model.bbox_head)

    # 3. Трассировка модели
    example_input = torch.randn(1, 3, 224, 224)  # Фиксированный размер
    traced_model = torch.jit.trace(model, example_input)
    
    # 4. Сохранение
    traced_model.save("model.pt")
    print("Модель успешно экспортирована в model.pt")

if __name__ == "__main__":
    export_model()