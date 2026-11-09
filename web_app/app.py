import os
import io
import json
import base64
import requests
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms

# Import models for direct fallback inference using explicit file paths
import sys
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fashion_mod = load_module_from_path("fashion_model", os.path.join(ROOT_DIR, "FashionMNIST", "model.py"))
FASHION_CLASSES = fashion_mod.FASHION_CLASSES
FashionBaselineMLP = fashion_mod.FashionBaselineMLP
FashionOptimizedCNN = fashion_mod.FashionOptimizedCNN

cifar_mod = load_module_from_path("cifar_model", os.path.join(ROOT_DIR, "CIFAR10", "model.py"))
CIFAR_CLASSES = cifar_mod.CIFAR_CLASSES
CIFARBaselineMLP = cifar_mod.CIFARBaselineMLP
CIFAROptimizedCNN = cifar_mod.CIFAROptimizedCNN

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Cog container inference endpoints (override via env if in docker network)
FASHION_COG_URL = os.getenv("FASHION_COG_URL", "http://localhost:5001/predictions")
CIFAR_COG_URL = os.getenv("CIFAR_COG_URL", "http://localhost:5002/predictions")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preload local models for direct standalone inference
fashion_baseline = FashionBaselineMLP().to(device)
fashion_optimized = FashionOptimizedCNN().to(device)
f_base_p = os.path.join(ROOT_DIR, "FashionMNIST", "models", "fashion_baseline.pth")
f_opt_p = os.path.join(ROOT_DIR, "FashionMNIST", "models", "fashion_optimized.pth")
if os.path.exists(f_base_p):
    fashion_baseline.load_state_dict(torch.load(f_base_p, map_location=device))
if os.path.exists(f_opt_p):
    fashion_optimized.load_state_dict(torch.load(f_opt_p, map_location=device))
fashion_baseline.eval()
fashion_optimized.eval()

cifar_baseline = CIFARBaselineMLP().to(device)
cifar_optimized = CIFAROptimizedCNN().to(device)
c_base_p = os.path.join(ROOT_DIR, "CIFAR10", "models", "cifar_baseline.pth")
c_opt_p = os.path.join(ROOT_DIR, "CIFAR10", "models", "cifar_optimized.pth")
if os.path.exists(c_base_p):
    cifar_baseline.load_state_dict(torch.load(c_base_p, map_location=device))
if os.path.exists(c_opt_p):
    cifar_optimized.load_state_dict(torch.load(c_opt_p, map_location=device))
cifar_baseline.eval()
cifar_optimized.eval()

# Transforms
fashion_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    dataset = request.args.get('dataset', 'fashion').lower()
    metrics_file = (
        os.path.join(ROOT_DIR, "FashionMNIST", "models", "metrics.json")
        if dataset == 'fashion'
        else os.path.join(ROOT_DIR, "CIFAR10", "models", "metrics.json")
    )
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found. Please train models first."}), 404


@app.route('/api/samples', methods=['GET'])
def get_samples():
    dataset = request.args.get('dataset', 'fashion').lower()
    samples_dir = (
        os.path.join(ROOT_DIR, "FashionMNIST", "test_images")
        if dataset == 'fashion'
        else os.path.join(ROOT_DIR, "CIFAR10", "test_images")
    )
    if not os.path.exists(samples_dir):
        return jsonify({"samples": []})
    samples = [
        {"name": f.replace('.png', '').replace('_', ' ').title(), "filename": f}
        for f in os.listdir(samples_dir) if f.endswith('.png')
    ]
    return jsonify({"samples": samples})


def local_infer(image_path, dataset):
    img = Image.open(image_path)
    if dataset == 'fashion':
        tensor = fashion_transform(img).unsqueeze(0).to(device)
        classes = FASHION_CLASSES
        with torch.no_grad():
            b_out = fashion_baseline(tensor)
            o_out = fashion_optimized(tensor)
        ds_name = "Fashion-MNIST"
        b_name = "Baseline MLP (PyTorch Quickstart)"
        o_name = "Optimized CNN (Hyperparameter Tuned)"
    else:
        tensor = cifar_transform(img.convert("RGB")).unsqueeze(0).to(device)
        classes = CIFAR_CLASSES
        with torch.no_grad():
            b_out = cifar_baseline(tensor)
            o_out = cifar_optimized(tensor)
        ds_name = "CIFAR-10"
        b_name = "Baseline MLP (Linear 3072->512->512->10)"
        o_name = "Optimized CNN (Conv2D + BatchNorm + Dropout)"

    # Softmax probabilities
    b_p = torch.nn.functional.softmax(b_out[0], dim=0)
    b_top5_p, b_top5_i = torch.topk(b_p, 5)
    b_preds = {classes[b_top5_i[i]]: f"{round(b_top5_p[i].item()*100, 2)}%" for i in range(5)}

    o_p = torch.nn.functional.softmax(o_out[0], dim=0)
    o_top5_p, o_top5_i = torch.topk(o_p, 5)
    o_preds = {classes[o_top5_i[i]]: f"{round(o_top5_p[i].item()*100, 2)}%" for i in range(5)}

    b_top_class = classes[b_top5_i[0]]
    b_top_conf = round(b_top5_p[0].item() * 100, 2)
    o_top_class = classes[o_top5_i[0]]
    o_top_conf = round(o_top5_p[0].item() * 100, 2)

    return {
        "dataset": ds_name,
        "source": "Local Fallback Inference Engine",
        "baseline": {
            "model": b_name,
            "predicted_class": b_top_class,
            "confidence": f"{b_top_conf}%",
            "top_5": b_preds
        },
        "optimized": {
            "model": o_name,
            "predicted_class": o_top_class,
            "confidence": f"{o_top_conf}%",
            "top_5": o_preds
        },
        "comparison": {
            "classes_agree": (b_top_class == o_top_class),
            "confidence_gain": round(o_top_conf - b_top_conf, 2)
        }
    }


@app.route('/api/classify', methods=['POST'])
def classify_image():
    dataset = request.form.get('dataset', 'fashion').lower()
    sample_name = request.form.get('sample_name')

    filepath = None
    if sample_name:
        sample_folder = (
            os.path.join(ROOT_DIR, "FashionMNIST", "test_images")
            if dataset == 'fashion'
            else os.path.join(ROOT_DIR, "CIFAR10", "test_images")
        )
        sample_path = os.path.join(sample_folder, secure_filename(sample_name))
        if os.path.exists(sample_path):
            filepath = sample_path

    if not filepath:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded.'}), 400
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Allowed image formats: PNG, JPG, JPEG, WEBP.'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    cog_url = FASHION_COG_URL if dataset == 'fashion' else CIFAR_COG_URL

    # Try Cog Container Inference first
    try:
        with open(filepath, "rb") as img_f:
            b64_data = base64.b64encode(img_f.read()).decode("utf-8")
        mime = "image/png"
        if filepath.lower().endswith(('.jpg', '.jpeg')):
            mime = "image/jpeg"
        data_uri = f"data:{mime};base64,{b64_data}"

        payload = {"input": {"image": data_uri}}
        resp = requests.post(cog_url, json=payload, timeout=5)
        if resp.status_code == 200:
            cog_result = resp.json().get("output", resp.json())
            cog_result["source"] = f"Cog Container ({cog_url})"
            return jsonify(cog_result)
    except Exception as e:
        print(f"[COG BRIDGE WARNING] Cog container at {cog_url} not reachable ({e}). Using local inference.")

    # Local fallback
    result = local_infer(filepath, dataset)
    return jsonify(result)


if __name__ == '__main__':
    print(f"Starting EE471 Week 7 Web App on http://localhost:5005 ...")
    app.run(host='0.0.0.0', port=5005, debug=True)
