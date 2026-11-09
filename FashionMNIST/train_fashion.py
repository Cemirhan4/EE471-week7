import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from model import FASHION_CLASSES, FashionBaselineMLP, FashionOptimizedCNN

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Device configuration (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== EE471 Week 7: Fashion-MNIST Training ===")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# 1. LOAD FASHION-MNIST DATASETS & DATALOADERS
# -----------------------------------------------------------------------------
transform_baseline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("\nDownloading / Loading FashionMNIST dataset...")
training_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform_baseline
)

test_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform_baseline
)

batch_size = 64
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Save sample images from test set for easy demonstration in Postman & Web UI
print("Exporting sample test images for demo...")
exported_classes = set()
for img_tensor, label in test_data:
    class_name = FASHION_CLASSES[label]
    if class_name not in exported_classes:
        # Convert tensor back to PIL (un-normalize)
        unnorm = (img_tensor * 0.5 + 0.5) * 255.0
        img_pil = Image.fromarray(unnorm.squeeze().numpy().astype('uint8'), mode='L')
        safe_name = class_name.replace("/", "_").replace(" ", "_").lower()
        img_path = os.path.join(TEST_IMAGES_DIR, f"{safe_name}.png")
        img_pil.save(img_path)
        exported_classes.add(class_name)
    if len(exported_classes) == 10:
        break
print(f"Saved {len(exported_classes)} sample garment images to: {TEST_IMAGES_DIR}")


# -----------------------------------------------------------------------------
# 2. TRAINING & EVALUATION HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    running_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    epoch_loss = running_loss / size
    epoch_acc = (correct / size) * 100
    return epoch_loss, epoch_acc


def test_epoch(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in enumerate(dataloader):
            # X is index, y is (batch_x, batch_y)
            batch_x, batch_y = y[0].to(device), y[1].to(device)
            pred = model(batch_x)
            test_loss += loss_fn(pred, batch_y).item()
            correct += (pred.argmax(1) == batch_y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_acc = (correct / size) * 100
    return test_loss, test_acc


# -----------------------------------------------------------------------------
# 3. TRAIN BASELINE MODEL (PyTorch Quickstart Tutorial Specification)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("PHASE 1: Training Fashion-MNIST Baseline Model (MLP with SGD, lr=1e-3)")
print("="*60)

baseline_model = FashionBaselineMLP().to(device)
loss_fn = nn.CrossEntropyLoss()
baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=1e-3)
baseline_epochs = 5

baseline_history = []
t0 = time.time()
for epoch in range(baseline_epochs):
    train_loss, train_acc = train_epoch(train_loader, baseline_model, loss_fn, baseline_optimizer)
    val_loss, val_acc = test_epoch(test_loader, baseline_model, loss_fn)
    baseline_history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
    print(f"Epoch {epoch+1}/{baseline_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f} | Test Acc: {val_acc:.2f}%")

baseline_time = time.time() - t0
baseline_path = os.path.join(MODELS_DIR, "fashion_baseline.pth")
torch.save(baseline_model.state_dict(), baseline_path)
print(f"Saved Baseline Model to: {baseline_path} (Training time: {baseline_time:.1f}s)")


# -----------------------------------------------------------------------------
# 4. TRAIN OPTIMIZED MODEL (Hyperparameter Tuned CNN with Adam, lr=1e-3)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("PHASE 2: Training Fashion-MNIST Optimized Model (CNN + Adam + BatchNorm + Dropout)")
print("="*60)

optimized_model = FashionOptimizedCNN().to(device)
optimized_optimizer = torch.optim.Adam(optimized_model.parameters(), lr=0.001, weight_decay=1e-4)
optimized_epochs = 8

optimized_history = []
t0 = time.time()
for epoch in range(optimized_epochs):
    train_loss, train_acc = train_epoch(train_loader, optimized_model, loss_fn, optimized_optimizer)
    val_loss, val_acc = test_epoch(test_loader, optimized_model, loss_fn)
    optimized_history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
    print(f"Epoch {epoch+1}/{optimized_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f} | Test Acc: {val_acc:.2f}%")

optimized_time = time.time() - t0
optimized_path = os.path.join(MODELS_DIR, "fashion_optimized.pth")
torch.save(optimized_model.state_dict(), optimized_path)
print(f"Saved Optimized Model to: {optimized_path} (Training time: {optimized_time:.1f}s)")


# -----------------------------------------------------------------------------
# 5. SAVE PERFORMANCE COMPARISON METRICS (For Report & Web UI)
# -----------------------------------------------------------------------------
final_metrics = {
    "dataset": "Fashion-MNIST",
    "classes": FASHION_CLASSES,
    "baseline": {
        "architecture": "3-Layer MLP (Linear 784->512->512->10)",
        "optimizer": "SGD",
        "learning_rate": 0.001,
        "epochs": baseline_epochs,
        "batch_size": batch_size,
        "final_test_loss": round(baseline_history[-1]["val_loss"], 4),
        "final_test_accuracy": round(baseline_history[-1]["val_acc"], 2),
        "training_time_seconds": round(baseline_time, 1),
        "history": baseline_history
    },
    "optimized": {
        "architecture": "Deep CNN (Conv2D(32->64->128) + BatchNorm + MaxPool + Dropout + Dense(256->10))",
        "optimizer": "Adam (weight_decay=1e-4)",
        "learning_rate": 0.001,
        "epochs": optimized_epochs,
        "batch_size": batch_size,
        "final_test_loss": round(optimized_history[-1]["val_loss"], 4),
        "final_test_accuracy": round(optimized_history[-1]["val_acc"], 2),
        "training_time_seconds": round(optimized_time, 1),
        "history": optimized_history
    },
    "accuracy_gain_percent": round(optimized_history[-1]["val_acc"] - baseline_history[-1]["val_acc"], 2)
}

metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(final_metrics, f, indent=2)

print("\n" + "="*60)
print(f"🏆 FASHION-MNIST RESULTS SUMMARY:")
print(f"Baseline Accuracy  : {final_metrics['baseline']['final_test_accuracy']}%")
print(f"Optimized Accuracy : {final_metrics['optimized']['final_test_accuracy']}%")
print(f"Performance Gain   : +{final_metrics['accuracy_gain_percent']}% improvement!")
print(f"Metrics saved to   : {metrics_path}")
print("="*60)
