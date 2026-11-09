import torch
import torch.nn as nn

# Fashion-MNIST 10 Garment Classes
FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# -----------------------------------------------------------------------------
# 1. BASELINE MODEL: PyTorch Quickstart Tutorial Architecture (MLP)
# -----------------------------------------------------------------------------
class FashionBaselineMLP(nn.Module):
    """
    Standard PyTorch Quickstart Tutorial architecture:
    Multi-layer Perceptron (MLP) with 2 hidden layers (512 units each) and ReLU.
    Input: 1 x 28 x 28 grayscale image flattened to 784 features.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
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
class FashionOptimizedCNN(nn.Module):
    """
    Hyperparameter-Optimized Architecture:
    - 2D Convolutions with Batch Normalization (stabilizes and speeds up convergence)
    - Max-Pooling layers for spatial downsampling and translation invariance
    - Dropout layers (0.25 and 0.5) to prevent overfitting
    - Adam optimizer with tuned learning rate (0.001) instead of vanilla SGD
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels (28x28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 64 x 14 x 14
            nn.Dropout2d(0.25),

            # Block 2: 64 -> 128 channels (14x14)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 128 x 7 x 7
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
