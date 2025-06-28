# scripts/evaluate.py

import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.dataset import MemeDataset
from utils.text_tokenizer import TextTokenizer
from model.multi_model import MultiMemeNet

def load_config(path='configs/default.yaml'):
    return yaml.safe_load(open(path))

def setup(cfg):
    device = torch.device(cfg['device'])
    tokenizer = TextTokenizer.from_excel(cfg['data']['labels_excel'])
    dataset = MemeDataset(
        img_dir=cfg['data']['raw_dir'],
        labels_excel=cfg['data']['labels_excel'],
        tokenizer=tokenizer,
        max_len=cfg['model']['max_ocr_len']
    )
    loader = DataLoader(dataset, batch_size=cfg['evaluation']['batch_size'], shuffle=False)

    model = MultiMemeNet(
        tokenizer,
        transformer_dim=cfg['model']['transformer_dim']
    ).to(device)
    model.load_state_dict(torch.load(cfg['evaluation']['ckpt_path'], map_location=device))
    model.eval()
    return device, loader, model

def evaluate():
    cfg = load_config()
    device, loader, model = setup(cfg)

    # Accumulators
    accum = {
        'brands_true': [], 'brands_pred': [],
        'context_true': [], 'context_pred': [],
        'technical_true': [], 'technical_pred': [],
        'sentiment_true': [], 'sentiment_pred': [],
        'humor_true': [], 'humor_pred': [],
        'sarcasm_true': [], 'sarcasm_pred': []
    }
    threshold = 0.5  # for multi-label

    with torch.no_grad():
        for batch in loader:
            imgs   = batch['image'].to(device)
            ocr_ids= batch['ocr_ids'].to(device)
            true   = batch['labels']

            outputs = model(imgs, ocr_ids)

            # Multi-label heads: apply sigmoid + threshold
            for head in ['brands', 'context', 'technical']:
                probs = torch.sigmoid(outputs[head]).cpu().numpy()
                preds = (probs >= threshold).astype(int)
                accum[f'{head}_pred'].append(preds)
                accum[f'{head}_true'].append(true[head].cpu().numpy())

            # Single-class heads: argmax over logits
            # note: 'overall_sentiment' in labels vs. 'sentiment' head
            single_map = {
                'sentiment': 'overall_sentiment',
                'humor':      'humor_mechanism',
                'sarcasm':    'sarcasm_level'
            }
            for head, label_col in single_map.items():
                logits = outputs[head].cpu().numpy()
                preds  = np.argmax(logits, axis=1)
                accum[f'{head}_pred'].append(preds)
                accum[f'{head}_true'].append(true[label_col].cpu().numpy())

    # Concatenate batches
    for k,v in accum.items():
        accum[k] = np.vstack(v) if v and v[0].ndim>1 else np.concatenate(v)

    # Compute metrics
    results = {}

    # Multi-label: precision, recall, F1 (averaged per sample)
    for head in ['brands', 'context', 'technical']:
        y_true = accum[f'{head}_true']
        y_pred = accum[f'{head}_pred']
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='samples', zero_division=0
        )
        results[head] = {'precision': p, 'recall': r, 'f1': f1}

    # Single-class: accuracy + macro precision/recall/F1
    for head in ['sentiment', 'humor', 'sarcasm']:
        y_true = accum[f'{head}_true']
        y_pred = accum[f'{head}_pred']
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        results[head] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}

    # Display
    for head, m in results.items():
        print(f"\n=== {head.upper()} ===")
        for metric, value in m.items():
            print(f"{metric.capitalize():<10}: {value:.4f}")

if __name__ == '__main__':
    evaluate()
