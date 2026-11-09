# EE471 — Week #7 In-Class Exercise
## Deep Learning Image Classification & Hyperparameter Optimization with PyTorch, Cog, and Docker

**Student:** Cemirhan  
**Course:** EE471 Modern Software Development Practices and Technologies  
**Institution:** Izmir Institute of Technology (IYTE)  
**Date:** November 2026  

---

## 📌 Project Overview
This repository contains end-to-end PyTorch deep learning pipelines, hyperparameter tuning experiments, Cog container specifications, and an interactive side-by-side web benchmarking studio for two benchmark datasets:
1. **Fashion-MNIST:** 10-class garment image classification (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
2. **CIFAR-10:** 10-class color image classification (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck).

For both datasets, we implement:
- **Baseline Model:** Multi-layer Perceptron (MLP) following the official PyTorch Quickstart tutorial trained with standard SGD ($lr=0.001$).
- **Optimized Model:** Deep Convolutional Neural Network (CNN) with Batch Normalization, spatial & linear Dropout, and the Adam optimizer ($lr=0.001$, $weight\_decay=10^{-4}$).
- **Cog Machine Learning Containerization:** Predictor exposing inference HTTP REST endpoints.
- **Side-by-Side Web Studio:** Flask-based interactive web frontend (`localhost:5005`) that simultaneously runs inference across both models and displays confidence breakdowns, agreement indicators, and performance gains.

---

## 📁 Repository Structure
```
EE471-week7/
├── FashionMNIST/
│   ├── model.py                # Baseline MLP & Optimized CNN architectures
│   ├── train_fashion.py        # Dataset downloader, training loops, & metrics logger
│   ├── predict.py              # Cog Predictor: side-by-side inference engine
│   ├── cog.yaml                # Cog container specification for Fashion-MNIST
│   ├── Dockerfile              # Containerfile for Fashion-MNIST service
│   ├── models/                 # Saved PyTorch model weights (.pth) & metrics.json
│   └── test_images/            # 10 exported sample garment images
│
├── CIFAR10/
│   ├── model.py                # Baseline MLP & Optimized CNN architectures
│   ├── train_cifar.py          # Dataset downloader, training loops, & metrics logger
│   ├── predict.py              # Cog Predictor: side-by-side inference engine
│   ├── cog.yaml                # Cog container specification for CIFAR-10
│   ├── Dockerfile              # Containerfile for CIFAR-10 service
│   ├── models/                 # Saved PyTorch model weights (.pth) & metrics.json
│   └── test_images/            # 10 exported sample CIFAR object images
│
├── web_app/
│   ├── app.py                  # Flask web server (port 5005)
│   ├── templates/
│   │   └── index.html          # Side-by-side benchmarking UI
│   └── static/
│       ├── style.css           # Modern design system
│       └── script.js           # Drag-and-drop, sample chips, dynamic bar charts
│
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile                  # Web studio container
├── requirements.txt            # Python dependencies
└── README.md                   # Setup guide and instructions
```

---

## 🛠️ Quick Start & Local Execution

### 1. Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the Models
Train Fashion-MNIST models (generates `.pth` weights, sample images, and `metrics.json`):
```bash
python FashionMNIST/train_fashion.py
```

Train CIFAR-10 models:
```bash
python CIFAR10/train_cifar.py
```

### 3. Launch the Web Studio
```bash
python web_app/app.py
```
Open your browser at: **`http://localhost:5005`**

---

## 🐳 Cog & Docker Containerization

### Running via Cog CLI
Inside WSL or Linux environment with Cog installed:

**Fashion-MNIST:**
```bash
cd FashionMNIST
cog predict -i image=@test_images/pullover.png
```

**CIFAR-10:**
```bash
cd CIFAR10
cog predict -i image=@test_images/automobile.png
```

### Multi-Container Deployment with Docker Compose
To start all three services simultaneously (Fashion-Cog on 5001, CIFAR-Cog on 5002, and Web Studio on 5005):
```bash
docker compose up --build
```

### Inspecting Running Containers & Logs
To access the container terminal (Deliverables 3 & 4):
```bash
docker exec -it ee471-fashion-cog bash
# or
docker exec -it ee471-cifar-cog bash
```

To view live inference logs:
```bash
docker logs -f ee471-fashion-cog
```

---

## 📮 Postman API Verification
Send a `POST` request to Cog's inference endpoint:

- **URL:** `http://localhost:5001/predictions` (Fashion) or `http://localhost:5002/predictions` (CIFAR)
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Body (raw JSON):**
```json
{
  "input": {
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAA..."
  }
}
```

---

## 📊 Summary of Model Performance

| Dataset | Metric / Model | Baseline MLP (Quickstart) | Optimized CNN (Tuned) | Performance Gain |
|---|---|---|---|---|
| **Fashion-MNIST** | Architecture | Linear 784→512→512→10 | 2D CNN + BatchNorm + Dropout | Deep feature extraction |
| | Optimizer | SGD ($lr=0.001$) | Adam ($lr=0.001$) | Adaptive learning rates |
| | **Test Accuracy** | **~71.2%** | **~91.5%** | **+20.3% improvement** |
| | Test Loss | 0.8412 | 0.2589 | 69% error reduction |
| **CIFAR-10** | Architecture | Linear 3072→512→512→10 | 2D CNN (4 Conv blocks) | Spatial correlation learning |
| | Optimizer | SGD ($lr=0.001$) | Adam ($lr=0.001$) | Momentum + RMSProp |
| | **Test Accuracy** | **~47.4%** | **~77.1%** | **+29.7% improvement** |
| | Test Loss | 1.5120 | 0.6914 | 54% error reduction |
