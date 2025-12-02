#!/usr/bin/env python3
# probe_and_map.py
# Usage:
#   python probe_and_map.py --model_dir our_models/bias_premsa --out label_map.json
#
# This script loads the local model, runs three short canonical texts (left/center/right)
# and uses the highest-scoring predicted index for each probe to assign numeric -> human labels.
# It writes a JSON file like {"0":"left","1":"center","2":"right"}.

import argparse, json, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='our_models/bias_premsa', help='Local model folder')
parser.add_argument('--out', default='label_map.json', help='Output JSON mapping file')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

if not os.path.exists(args.model_dir):
    raise SystemExit(f"Model dir not found: {args.model_dir}")

print("Loading model from", args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.eval()

# canonical probe texts (short). Tweak if you want different probes.
left_text = "The government should expand social welfare programs to help low-income families."
center_text = "Experts discussed balanced approaches combining fiscal responsibility with social programs."
right_text = "Lower taxes and deregulation are necessary to boost business and economic growth."

probes = {
    'left': left_text,
    'center': center_text,
    'right': right_text
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

print("Running probes to infer label mapping...")
pred_for_probe = {}
probs_for_probe = {}
for name, text in probes.items():
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        pred = int(logits.argmax(dim=-1).cpu().numpy())
    pred_for_probe[name] = pred
    probs_for_probe[name] = probs
    print(f"Probe '{name}' -> predicted index: {pred}, probs: {probs}")

# Build mapping: prefer one-to-one mapping
mapping = {}
used = set()
for human_label in ['left','center','right']:
    pred = pred_for_probe[human_label]
    if pred in used:
        # collision — try to disambiguate by highest probability among unused labels
        # choose index with highest average confidence for this human_label among indices not used
        cand = None
        bestp = -1.0
        for i, p in enumerate(probs_for_probe[human_label]):
            if i in used: continue
            if p > bestp:
                bestp = p; cand = i
        if cand is None:
            # fallback: find any unused index
            for i in range(len(probs_for_probe[human_label])):
                if i not in used:
                    cand = i; break
        mapping[str(cand)] = human_label
        used.add(cand)
    else:
        mapping[str(pred)] = human_label
        used.add(pred)

# If there are any remaining indices (e.g. label count !=3), map them to 'unknown'
num_labels = model.config.num_labels if hasattr(model.config,'num_labels') else len(next(iter(probs_for_probe.values())))
for i in range(num_labels):
    if str(i) not in mapping:
        mapping[str(i)] = 'unknown' if 'unknown' not in mapping.values() else 'other'

print("Inferred mapping (numeric -> human):")
print(json.dumps(mapping, indent=2))

# Save mapping
if os.path.exists(args.out) and not args.overwrite:
    print(f"Warning: {args.out} exists. Use --overwrite to replace. Current file not overwritten.")
else:
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("Saved mapping to", args.out)

# Also print the raw model id2label if present for manual checking
cfg = getattr(model, 'config', None)
if cfg and getattr(cfg, 'id2label', None):
    print("Model id2label:", cfg.id2label)
else:
    print("Model config has no id2label entries.")
