#!/usr/bin/env python3
# download_model_offline.py
# Usage: python download_model_offline.py --model premsa/political-bias-prediction-allsides-DeBERTa --out_dir our_models/bias_premsa

import argparse, os, sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='premsa/political-bias-prediction-allsides-DeBERTa',
                    help='Hugging Face model id to download (e.g. premsa/political-bias-prediction-allsides-DeBERTa)')
parser.add_argument('--out_dir', required=False, default='our_models/bias_premsa',
                    help='Local directory to save tokenizer + model')
parser.add_argument('--force', action='store_true', help='Overwrite if out_dir exists')
args = parser.parse_args()

if os.path.exists(args.out_dir) and not args.force:
    print(f"Directory {args.out_dir} already exists. Use --force to overwrite or remove it first.")
    sys.exit(0)

os.makedirs(args.out_dir, exist_ok=True)
print(f"Downloading model '{args.model}' to local directory: {args.out_dir}")

try:
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.out_dir)
    print("Tokenizer saved to", args.out_dir)
except Exception as e:
    print("Failed to download or save tokenizer:", e)
    raise

try:
    print("Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.save_pretrained(args.out_dir)
    print("Model saved to", args.out_dir)
except Exception as e:
    print("Failed to download or save model:", e)
    raise

print("Done. You can now use the model offline by loading from:", args.out_dir)
