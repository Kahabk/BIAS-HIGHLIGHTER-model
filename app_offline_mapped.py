#!/usr/bin/env python3
# app_offline_mapped.py
# Usage: python app_offline_mapped.py --model_dir our_models/bias_premsa --map label_map.json --port 8000

import argparse, os, json, time
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='our_models/bias_premsa')
parser.add_argument('--map', default='label_map.json')
parser.add_argument('--port', type=int, default=8000)
args = parser.parse_args()

if not os.path.exists(args.model_dir):
    raise SystemExit("Model dir not found: " + args.model_dir)
if not os.path.exists(args.map):
    raise SystemExit("Mapping file not found: " + args.map + ". Run probe_and_map.py first.")

print("Loading mapping from", args.map)
with open(args.map,'r',encoding='utf-8') as f:
    mapping = json.load(f)

print("Loading tokenizer & model from", args.model_dir, "-- this may take a few seconds")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

LABELS = [mapping.get(str(i), 'unknown') for i in range(model.config.num_labels)]
print("Using LABELS:", LABELS)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests (ngrok + browser)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "bias-detection-offline",
        "labels": LABELS,
        "note": "POST /predict JSON {'text':'...'}"
    })

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({'error': 'Invalid JSON payload'}), 400
    if 'text' in payload and payload['text']:
        text = payload['text']
    else:
        text = (payload.get('headline','') or '') + '\n' + (payload.get('body','') or '')
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        pred_idx = int(logits.argmax(dim=-1).cpu().numpy()[0])

    resp = {
        'label': LABELS[pred_idx] if pred_idx < len(LABELS) else str(pred_idx),
        'probs': { LABELS[i] if i < len(LABELS) else str(i): float(probs[i]) for i in range(len(probs)) },
        'latency_sec': round(time.time() - start, 3)
    }
    return jsonify(resp)

if __name__ == '__main__':
    print("Starting Flask app on port", args.port)
    app.run(host='0.0.0.0', port=args.port)
