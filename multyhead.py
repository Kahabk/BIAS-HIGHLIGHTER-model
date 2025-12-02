import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

LABELS = {
    'political_bias': ['left','center','right','unknown'],
    'tone': ['positive','neutral','negative'],
    'framing': ['sensational','factual','neutral']
}

class MultiHeadModel(nn.Module):
    def __init__(self, encoder_name, hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.pool = lambda x: x.last_hidden_state[:,0,:]  # CLS pooling
        self.head_p = nn.Linear(hidden_size, len(LABELS['political_bias']))
        self.head_t = nn.Linear(hidden_size, len(LABELS['tone']))
        self.head_f = nn.Linear(hidden_size, len(LABELS['framing']))
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out)
        return self.head_p(pooled), self.head_t(pooled), self.head_f(pooled)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='xlm-roberta-base')
parser.add_argument('--train', default='train.jsonl')
parser.add_argument('--val', default='val.jsonl')
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

print('Loading datasets...')
raw = load_dataset('json', data_files={'train': args.train, 'validation': args.val})

label2id = {k: {l:i for i,l in enumerate(LABELS[k])} for k in LABELS}

def map_labels(ex):
    ex['p_label'] = label2id['political_bias'].get(ex.get('political_bias','unknown'), 3)
    ex['t_label'] = label2id['tone'].get(ex.get('tone','neutral'), 1)
    ex['f_label'] = label2id['framing'].get(ex.get('framing','neutral'), 2)
    return ex

raw = raw.map(map_labels)

print('Tokenizer...')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

def tokenize(batch):
    texts = [(h or '') + '\n' + (b or '') for h,b in zip(batch['headline'], batch['body'])]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=args.max_len)

raw = raw.map(tokenize, batched=True)
cols = ['input_ids','attention_mask','p_label','t_label','f_label']
for s in ['train','validation']:
    raw[s].set_format(type='torch', columns=cols)

train_loader = DataLoader(raw['train'], batch_size=args.batch, shuffle=True)
val_loader = DataLoader(raw['validation'], batch_size=args.batch)

print('Building model...')
device = args.device
model = MultiHeadModel(args.model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        p_labels = batch['p_label'].to(device)
        t_labels = batch['t_label'].to(device)
        f_labels = batch['f_label'].to(device)

        logits_p, logits_t, logits_f = model(input_ids, attention_mask)
        loss_p = criterion(logits_p, p_labels)
        loss_t = criterion(logits_t, t_labels)
        loss_f = criterion(logits_f, f_labels)
        loss = loss_p + loss_t + loss_f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': loss.item()})

    # validation
    model.eval()
    all_p_preds, all_p_trues = [], []
    all_t_preds, all_t_trues = [], []
    all_f_preds, all_f_trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            p_labels = batch['p_label'].numpy()
            t_labels = batch['t_label'].numpy()
            f_labels = batch['f_label'].numpy()
            lp, lt, lf = model(input_ids, attention_mask)
            p_preds = lp.argmax(dim=1).cpu().numpy()
            t_preds = lt.argmax(dim=1).cpu().numpy()
            f_preds = lf.argmax(dim=1).cpu().numpy()
            all_p_preds.extend(p_preds); all_p_trues.extend(p_labels)
            all_t_preds.extend(t_preds); all_t_trues.extend(t_labels)
            all_f_preds.extend(f_preds); all_f_trues.extend(f_labels)

    print('Epoch', epoch+1, 'P F1', f1_score(all_p_trues, all_p_preds, average='macro'),
          'T F1', f1_score(all_t_trues, all_t_preds, average='macro'),
          'F F1', f1_score(all_f_trues, all_f_preds, average='macro'))

# save model
torch.save(model.state_dict(), 'multihead_model.pt')
print('Saved multihead_model.pt')