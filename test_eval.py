import os
import torch
from PIL import Image
from torchvision import transforms
import importlib.util

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

cifar_mod = load_mod("cifar_m", os.path.join(ROOT_DIR, "CIFAR10", "model.py"))
classes = cifar_mod.CIFAR_CLASSES

device = torch.device("cpu")
base = cifar_mod.CIFARBaselineMLP()
base.load_state_dict(torch.load(os.path.join(ROOT_DIR, "CIFAR10", "models", "cifar_baseline.pth"), map_location=device))
base.eval()

opt = cifar_mod.CIFAROptimizedCNN()
opt.load_state_dict(torch.load(os.path.join(ROOT_DIR, "CIFAR10", "models", "cifar_optimized.pth"), map_location=device))
opt.eval()

tf = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dir = os.path.join(ROOT_DIR, "CIFAR10", "test_images")
results = {}
for fname in sorted(os.listdir(test_dir)):
    if fname.endswith(".png"):
        img_p = os.path.join(test_dir, fname)
        img = Image.open(img_p).convert("RGB")
        tensor = tf(img).unsqueeze(0)
        with torch.no_grad():
            b_out = base(tensor)
            o_out = opt(tensor)
        b_pred = classes[b_out.argmax(1).item()]
        o_pred = classes[o_out.argmax(1).item()]
        true_label = fname.replace(".png", "")
        results[true_label] = {"baseline": b_pred, "optimized": o_pred}

print("=== CIFAR TEST SAMPLES EVALUATION ===")
for label, preds in results.items():
    status = "OK" if preds["optimized"] == label else "DIFF"
    print(f"True: {label:12s} | Baseline: {preds['baseline']:12s} | Optimized: {preds['optimized']:12s} | [{status}]")
