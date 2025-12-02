#!/usr/bin/env python3
# run_local_infer_mapped.py
# Usage:
#  python run_local_infer_mapped.py --model_dir our_models/bias_premsa --map label_map.json --text "The text..."

import argparse, json, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='our_models/bias_premsa')
parser.add_argument('--map', default='label_map.json', help='Numeric->human mapping JSON')
parser.add_argument('--text', default='', help='Text to classify (headline\\nbody or body)')
args = parser.parse_args()

if not os.path.exists(args.model_dir):
    raise SystemExit("Model directory not found: " + args.model_dir)
if not os.path.exists(args.map):
    raise SystemExit("Mapping file not found: " + args.map + ". Run probe_and_map.py first.")

with open(args.map,'r',encoding='utf-8') as f:
    mapping = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.eval()

LABELS = [mapping.get(str(i), 'unknown') for i in range(model.config.num_labels)]

def predict(text):
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        pred_idx = int(logits.argmax(dim=-1).cpu().numpy())
    return {'label': LABELS[pred_idx] if pred_idx < len(LABELS) else str(pred_idx),
            'probs': {LABELS[i] if i < len(LABELS) else str(i): float(probs[i]) for i in range(len(probs))}}

if __name__ == '__main__':
    if not args.text:
        args.text = input("Enter text to classify: ")
    out = predict(args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
