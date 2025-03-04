import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BusNumberRecognizer  # Импортируем модель
from dataset import BusNumberDataset  # Импортируем датасет
from torchvision import transforms

# Настройки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка датасета
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]  # Замените на свои
labels = [0, 1, ...]  # Метки классов (номера автобусов)
dataset = BusNumberDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Инициализация модели
model = BusNumberRecognizer(num_classes=10).to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Для сохранения лучшей модели
best_loss = float("inf")
best_model_path = "best_bus_number_recognizer.pt"
patience = 5  # Сколько эпох ждать улучшения Loss
no_improvement_count = 0  # Счётчик эпох без улучшения

# Обучение
for epoch in range(num_epochs):
    model.train()  # Переводим модель в режим обучения
    running_loss = 0.0
    for images, labels in dataloader:
        # Перемещаем данные на GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Средние потери за эпоху
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Сохраняем модель, если потери улучшились
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
        no_improvement_count = 0  # Сбрасываем счётчик
        print(f"Новая лучшая модель сохранена с loss: {best_loss:.4f}")
    else:
        no_improvement_count += 1  # Увеличиваем счётчик
    
    # Ранняя остановка, если Loss не улучшается в течение patience эпох
    if no_improvement_count >= patience:
        print(f"Обучение остановлено, так как Loss не улучшался в течение {patience} эпох.")
        break

# Сохранение финальной модели в формате TorchScript для Android
model.load_state_dict(torch.load(best_model_path))  # Загружаем лучшую модель
example_input = torch.rand(1, 3, 224, 224).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("bus_number_recognizer.pt")
print("Лучшая модель сохранена в формате TorchScript.")