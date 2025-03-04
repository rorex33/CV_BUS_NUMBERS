import torch
import torch.nn as nn

class BusNumberRecognizer(nn.Module):
    def __init__(self, num_classes=10):
        super(BusNumberRecognizer, self).__init__()
        # Свёрточные слои для извлечения признаков
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Полносвязные слои для классификации
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),  # Размер выхода CNN: (128, 28, 28)
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Применяем CNN
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Выравниваем для полносвязного слоя
        # Применяем полносвязные слои
        x = self.fc(x)
        return x

# Проверка модели
if __name__ == "__main__":
    model = BusNumberRecognizer(num_classes=10)
    print(model)