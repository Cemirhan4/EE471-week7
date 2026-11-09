import os
import torch
from PIL import Image
from torchvision import transforms
from cog import BasePredictor, Input, Path

from model import CIFAR_CLASSES, CIFARBaselineMLP, CIFAROptimizedCNN

class Predictor(BasePredictor):
    def setup(self):
        """Load both Baseline and Optimized CIFAR-10 models into memory."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SETUP] Loading CIFAR-10 models on {self.device}...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        baseline_path = os.path.join(models_dir, "cifar_baseline.pth")
        optimized_path = os.path.join(models_dir, "cifar_optimized.pth")

        # Load Baseline MLP
        self.baseline_model = CIFARBaselineMLP().to(self.device)
        if os.path.exists(baseline_path):
            self.baseline_model.load_state_dict(torch.load(baseline_path, map_location=self.device))
            print(f"[SETUP] Loaded baseline model from {baseline_path}")
        else:
            print("[SETUP WARNING] Baseline weights not found. Using untrained weights.")
        self.baseline_model.eval()

        # Load Optimized CNN
        self.optimized_model = CIFAROptimizedCNN().to(self.device)
        if os.path.exists(optimized_path):
            self.optimized_model.load_state_dict(torch.load(optimized_path, map_location=self.device))
            print(f"[SETUP] Loaded optimized model from {optimized_path}")
        else:
            print("[SETUP WARNING] Optimized weights not found. Using untrained weights.")
        self.optimized_model.eval()

        # Preprocessing transform for 32x32 RGB CIFAR-10
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.classes = CIFAR_CLASSES
        print("[SETUP] CIFAR-10 Predictor ready.")

    def run(
        self,
        image: Path = Input(description="Object image to classify (RGB)")
    ) -> dict:
        """Run inference using BOTH baseline and optimized models for side-by-side comparison."""
        print(f"[RUN] Received image: {image}")
        img = Image.open(str(image)).convert("RGB")

        # Preprocess image into tensor: [1, 3, 32, 32]
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            baseline_out = self.baseline_model(tensor)
            optimized_out = self.optimized_model(tensor)

        # Baseline probabilities
        base_probs = torch.nn.functional.softmax(baseline_out[0], dim=0)
        base_top5_p, base_top5_idx = torch.topk(base_probs, 5)

        baseline_predictions = {}
        for i in range(base_top5_p.size(0)):
            cname = self.classes[base_top5_idx[i]]
            pct = round(base_top5_p[i].item() * 100, 2)
            baseline_predictions[cname] = f"{pct}%"

        # Optimized probabilities
        opt_probs = torch.nn.functional.softmax(optimized_out[0], dim=0)
        opt_top5_p, opt_top5_idx = torch.topk(opt_probs, 5)

        optimized_predictions = {}
        for i in range(opt_top5_p.size(0)):
            cname = self.classes[opt_top5_idx[i]]
            pct = round(opt_top5_p[i].item() * 100, 2)
            optimized_predictions[cname] = f"{pct}%"

        top_baseline_class = self.classes[base_top5_idx[0]]
        top_baseline_conf = round(base_top5_p[0].item() * 100, 2)
        top_optimized_class = self.classes[opt_top5_idx[0]]
        top_optimized_conf = round(opt_top5_p[0].item() * 100, 2)

        result = {
            "dataset": "CIFAR-10",
            "baseline": {
                "model": "Baseline MLP (Linear 3072->512->512->10)",
                "predicted_class": top_baseline_class,
                "confidence": f"{top_baseline_conf}%",
                "top_5": baseline_predictions
            },
            "optimized": {
                "model": "Optimized CNN (Conv2D + BatchNorm + Dropout)",
                "predicted_class": top_optimized_class,
                "confidence": f"{top_optimized_conf}%",
                "top_5": optimized_predictions
            },
            "comparison": {
                "classes_agree": (top_baseline_class == top_optimized_class),
                "confidence_gain": round(top_optimized_conf - top_baseline_conf, 2)
            }
        }

        print(f"[RESULT] Baseline: {top_baseline_class} ({top_baseline_conf}%) vs Optimized: {top_optimized_class} ({top_optimized_conf}%)")
        return result
