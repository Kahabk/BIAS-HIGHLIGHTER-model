# preprocess_allsides_local.py
# Reads allsides.csv / allsides.json (if available) and produces train.jsonl / val.jsonl / test.jsonl
import pandas as pd, json, argparse, uuid, os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--allsides_csv', default='allsides.csv')
parser.add_argument('--allsides_json', default='allsides.json')
parser.add_argument('--out_dir', default='data/allsides_processed')
parser.add_argument('--test_size', type=float, default=0.10)
parser.add_argument('--val_size', type=float, default=0.10)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

def read_csv(path):
    try:
        return pd.read_csv(path, dtype=str).fillna('')
    except Exception as e:
        print("CSV read failed:", e)
        return None

def read_json(path):
    try:
        with open(path,'r',encoding='utf-8') as f:
            return [json.loads(l) for l in f] 
    except Exception as e:
        print("JSON read failed:", e)
        return None

df = read_csv(args.allsides_csv)
json_docs = read_json(args.allsides_json) if os.path.exists(args.allsides_json) else None

# Heuristic: find outlet->bias mapping in CSV
# expected columns: outlet, bias (or source, rating, lean)
outlet_col = None
bias_col = None
if df is not None:
    for c in df.columns:
        if c.lower() in ('outlet','source','media','publication','publisher'):
            outlet_col = c; break
    for c in df.columns:
        if c.lower() in ('bias','leans','rating','bias_label','allsides_rating','political_leaning'):
            bias_col = c; break

outlet_map = {}
if outlet_col and bias_col:
    for _, r in df.iterrows():
        name = str(r[outlet_col]).strip()
        lbl = str(r[bias_col]).strip().lower()
        if 'left' in lbl: lab='left'
        elif 'right' in lbl: lab='right'
        elif 'center' in lbl or 'lean' in lab or 'mixed' in lbl: lab='center'
        else: lab='unknown'
        outlet_map[name.lower()] = lab

print(f"Loaded {len(outlet_map)} outlet labels")

# Next, build article list.
# If allsides.csv also contains article rows (title/text) use them; otherwise use allsides.json if it contains articles.
articles = []

# Try CSV rows for text
if df is not None:
    # check for text-like columns
    text_col = None
    title_col = None
    for c in df.columns:
        if c.lower() in ('text','body','article','content'):
            text_col = c; break
    for c in df.columns:
        if c.lower() in ('title','headline'):
            title_col = c; break

    if text_col or title_col:
        for _, r in df.iterrows():
            headline = str(r[title_col]) if title_col else ''
            body = str(r[text_col]) if text_col else ''
            source = ''
            for c in (outlet_col, 'source','publication','publisher'):
                if c and c in df.columns:
                    source = str(r[c]); break
            label = outlet_map.get(source.lower(), 'unknown') if source else 'unknown'
            if not body and not headline:
                continue
            articles.append({
                'id': str(uuid.uuid4()),
                'headline': headline,
                'body': body if body else headline,
                'source': source,
                'political_bias': label
            })

# fallback: try allsides.json if no article rows in CSV
if not articles and json_docs:
    for item in json_docs:
        # try common keys
        title = item.get('title') or item.get('headline') or ''
        body = item.get('text') or item.get('body') or item.get('article') or ''
        source = item.get('source') or item.get('outlet') or item.get('publisher') or ''
        label = outlet_map.get(str(source).lower(), 'unknown')
        if not (title or body):
            continue
        articles.append({
            'id': str(uuid.uuid4()),
            'headline': title,
            'body': body if body else title,
            'source': source,
            'political_bias': label
        })

print("Collected", len(articles), "articles")

# Clean and filter very short
articles = [a for a in articles if len((a['headline'] or '') + (a['body'] or '')) > 50]

# Split into train/val/test stratified by political_bias where possible
labels = [a['political_bias'] if a['political_bias'] else 'unknown' for a in articles]
if len(set(labels)) > 1:
    train, rest = train_test_split(articles, test_size=(args.test_size + args.val_size), random_state=42, stratify=labels)
    val, test = train_test_split(rest, test_size=(args.test_size / (args.test_size + args.val_size)), random_state=42, stratify=[a['political_bias'] for a in rest])
else:
    # no label diversity — simple split
    n = len(articles)
    train = articles[:int(0.8*n)]
    val = articles[int(0.8*n):int(0.9*n)]
    test = articles[int(0.9*n):]

def write_jsonl(arr, path):
    with open(path,'w',encoding='utf-8') as f:
        for a in arr:
            out = {
                'id': a.get('id'),
                'headline': a.get('headline',''),
                'body': a.get('body',''),
                'political_bias': a.get('political_bias','unknown')
            }
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

write_jsonl(train, os.path.join(args.out_dir,'train.jsonl'))
write_jsonl(val,   os.path.join(args.out_dir,'val.jsonl'))
write_jsonl(test,  os.path.join(args.out_dir,'test.jsonl'))

print("Wrote splits to", args.out_dir)
