import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf

from models.simple_cnn import SimpleCNN
from data.dataloader import get_dataloaders
import os
import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Access like cfg.model.hidden_channels, cfg.experiment.batch_size, etc.
    
    device = torch.device(cfg.experiments.train.device)
    train_loader, test_loader = get_dataloaders(cfg.experiments.train.batch_size)
    train_dataset = train_loader.dataset
    logger.info(f"Number of training samples: {len(train_dataset)}")
    test_dataset = test_loader.dataset
    logger.info(f"Number of test samples: {len(test_dataset)}")

    model = SimpleCNN(
        hidden_channels=cfg.models.model.hidden_channels,
        num_classes=cfg.models.model.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.experiments.train.learning_rate)

    for epoch in range(cfg.experiments.train.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{cfg.experiments.train.epochs} - Loss: {loss.item():.4f}")
        logger.info(f"Epoch {epoch+1}/{cfg.experiments.train.epochs} - Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "model.pt"))
    print("✅ Model saved to model.pt")

if __name__ == "__main__":
    train()
