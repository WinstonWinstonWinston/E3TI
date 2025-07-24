import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from models.simple_cnn import SimpleCNN
from data.dataloader import get_dataloaders
import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def evaluate(cfg: DictConfig):
    print("🧪 Starting evaluation...")
    device = torch.device(cfg.experiments.train.device)

    _, test_loader = get_dataloaders(cfg.experiments.train.batch_size)
    test_dataset = test_loader.dataset
    logger.info(f"Number of test samples: {len(test_dataset)}")
    
    # Load model
    model = SimpleCNN(
        hidden_channels=cfg.models.model.hidden_channels,
        num_classes=cfg.models.model.num_classes
    ).to(device)

    model_path = os.path.join(os.getcwd(), "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    logger.info(f"✅ Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate()
