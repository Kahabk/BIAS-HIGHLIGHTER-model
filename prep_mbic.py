# prep_mbic.py
import pandas as pd, json, argparse, uuid
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='data/mbic/mbic.csv', help='Path to MBIC CSV')
parser.add_argument('--out_prefix', default='data/mbic_processed', help='Prefix for train/val/test jsonl')
args = parser.parse_args()

df = pd.read_csv(args.csv)
# Inspect columns
print('Columns:', df.columns.tolist())

# adjust these if your CSV column names differ
text_col = 'Text' if 'Text' in df.columns else df.columns[0]
label_col = 'Label' if 'Label' in df.columns else df.columns[1]

def clean_label(x):
    s = str(x).lower()
    if 'left' in s: return 'left'
    if 'right' in s: return 'right'
    return 'center'

df['political_bias'] = df[label_col].apply(clean_label)
df = df[[text_col, 'political_bias']].dropna()
df = df.rename(columns={text_col: 'body'}).reset_index(drop=True)

# optional: downsample or clean extremely short rows
df['length'] = df['body'].str.len()
df = df[df['length'] > 20]

train, test = train_test_split(df, test_size=0.1, stratify=df['political_bias'], random_state=42)
train, val  = train_test_split(train, test_size=0.1111, stratify=train['political_bias'], random_state=42)  # 0.1111 * 0.9 ≈ 0.1

def write_jsonl(df, path):
    with open(path, 'w', encoding='utf-8') as f:
        for _, r in df.iterrows():
            obj = {
                'id': str(uuid.uuid4()),
                'headline': '',
                'body': r['body'],
                'political_bias': r['political_bias']
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

import os
os.makedirs(args.out_prefix, exist_ok=True)
write_jsonl(train, f"{args.out_prefix}/train.jsonl")
write_jsonl(val,   f"{args.out_prefix}/val.jsonl")
write_jsonl(test,  f"{args.out_prefix}/test.jsonl")

print('Wrote:', f"{args.out_prefix}/train.jsonl", f"{args.out_prefix}/val.jsonl", f"{args.out_prefix}/test.jsonl")
