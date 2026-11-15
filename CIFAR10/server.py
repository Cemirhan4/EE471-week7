import os
import sys
import io
import base64
import tempfile
import requests
from flask import Flask, request, jsonify
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import Predictor

app = Flask(__name__)
print("=== Initializing CIFAR-10 Cog Server ===")
predictor = Predictor()
predictor.setup()

@app.route('/predictions', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True) or {}
        img_input = data.get("input", {}).get("image", "")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_path = temp_file.name
        temp_file.close()

        if img_input.startswith("http://") or img_input.startswith("https://"):
            resp = requests.get(img_input, timeout=10)
            with open(temp_path, "wb") as f:
                f.write(resp.content)
        elif "base64," in img_input:
            b64_str = img_input.split("base64,")[1]
            with open(temp_path, "wb") as f:
                f.write(base64.b64decode(b64_str))
        elif os.path.exists(img_input):
            temp_path = img_input
        else:
            default_sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images", "automobile.png")
            temp_path = default_sample

        result = predictor.run(image=temp_path)
        return jsonify({"output": result, "status": "succeeded"})
    except Exception as e:
        print(f"[ERROR in Predict]: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5002))
    print(f"CIFAR-10 Cog Endpoint ready: http://localhost:{port}/predictions")
    app.run(host='0.0.0.0', port=port)
