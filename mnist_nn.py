"""Simple fully connected neural network for MNIST classification."""

import torch
from torch import nn


class SimpleMNISTClassifier(nn.Module):
    """Two-layer fully connected network for MNIST digits."""

    def __init__(self, hidden_units: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


def predict(logits: torch.Tensor) -> torch.Tensor:
    """Return the predicted labels from logits."""
    return logits.argmax(dim=1)


def calc_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the classification accuracy for predictions and targets."""
    return (predictions == targets).float().mean()


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
