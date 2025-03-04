from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BusNumberDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Пример использования
if __name__ == "__main__":
    # Пример данных (замените на свои)
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
    labels = [0, 1, ...]  # Метки классов (номера автобусов)

    # Преобразования для изображений
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Создание датасета
    dataset = BusNumberDataset(image_paths, labels, transform=transform)
    print(f"Размер датасета: {len(dataset)}")