import torch
import hydra
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os

from models.simple_cnn import SimpleCNN
from data.dataloader import get_dataloaders
import logging
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.experiments.train.device)

    _, test_loader = get_dataloaders(batch_size=cfg.experiments.train.batch_size)

    # Load model
    model = SimpleCNN(
        hidden_channels=cfg.models.model.hidden_channels,
        num_classes=cfg.models.model.num_classes
    ).to(device)

    model_path = os.path.join(os.getcwd(), "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred, confidences = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            confidences.extend(conf.cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)

    if cfg.evaluation.analyze.misclassified_only:
        mask = y_true != y_pred
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        confidences = confidences[mask]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Ensure Hydra runtime config is available
    output_dir = HydraConfig.get().runtime.output_dir
    save_path = os.path.join(output_dir, "confusion_matrix.png")

    plt.savefig(save_path)
    print(f"✅ Saved confusion matrix to {save_path}")

    if cfg.evaluation.analyze.show_confidences:
        print("🔍 Sample prediction confidences:")
        logger.info("🔍 Sample prediction confidences:")

        for i in range(min(10, len(confidences))):
            print(f"True: {y_true[i]} | Pred: {y_pred[i]} | Confidence: {confidences[i]:.2f}")
            logger.info(f"True: {y_true[i]} | Pred: {y_pred[i]} | Confidence: {confidences[i]:.2f}")


if __name__ == "__main__":
    main()
