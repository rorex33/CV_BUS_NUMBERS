import torch
from multitask_vehicle_model import MultiTaskModel
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

vocab = config['model']['vocab']
model = MultiTaskModel(num_classes=5, vocab_size=len(vocab) + 1)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

example_input = torch.rand(1, 3, config['image']['size'], config['image']['size'])
traced_model = torch.jit.trace(model, example_input)
traced_model.save("multitask_model_ts.pt")

print("✅ TorchScript модель сохранена как multitask_model_ts.pt")
