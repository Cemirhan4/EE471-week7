import torch
import torch.nn as nn

# CIFAR-10 10 Object Classes
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# -----------------------------------------------------------------------------
# 1. BASELINE MODEL: Multi-Layer Perceptron (MLP) adapted from Quickstart Tutorial
# -----------------------------------------------------------------------------
class CIFARBaselineMLP(nn.Module):
    """
    Standard PyTorch Tutorial MLP pattern adapted for 3-channel CIFAR-10:
    Input: 3 x 32 x 32 RGB image flattened to 3072 features.
    Architecture: Linear(3072, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    Trained with: SGD (lr=0.001)
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# -----------------------------------------------------------------------------
# 2. OPTIMIZED MODEL: Convolutional Neural Network (CNN) with Hyperparameter Tuning
# -----------------------------------------------------------------------------
class CIFAROptimizedCNN(nn.Module):
    """
    Hyperparameter-Optimized Architecture for CIFAR-10:
    - 4 Convolutional layers capturing low-level and high-level visual features
    - Batch Normalization on every conv block to stabilize training and combat covariate shift
    - Max-Pooling (2x2) for spatial reduction
    - Spatial and linear Dropout (0.25 and 0.5) to regularize and eliminate overfitting
    - Adam optimizer with tuned learning rate (0.001)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 -> 64 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 64 x 16 x 16
            nn.Dropout2d(0.25),

            # Block 2: 64 -> 128 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 128 x 8 x 8
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
