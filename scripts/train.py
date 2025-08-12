# scripts/train.py

import os
import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.dataset import MemeDataset
from utils.text_tokenizer import TextTokenizer
from model.multi_model import MultiMemeNet

def main(cfg):
    # Device
    desired = cfg.get('device', 'cpu')
    device  = torch.device(desired if (desired=='cpu' or torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")

    # Tokenizer + Dataset
    tokenizer = TextTokenizer.from_excel(cfg['data']['labels_excel'])
    dataset   = MemeDataset(
        img_dir=cfg['data']['raw_dir'],
        labels_excel=cfg['data']['labels_excel'],
        tokenizer=tokenizer,
        max_len=cfg['model']['max_ocr_len']
    )
    loader    = DataLoader(dataset,
                           batch_size=cfg['training']['batch_size'],
                           shuffle=True,
                           num_workers=4)

    # Model
    model = MultiMemeNet(tokenizer, transformer_dim=cfg['model']['transformer_dim'])
    model.to(device)

    # Optimizer
    lr        = float(cfg['training']['lr'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Losses
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce  = nn.CrossEntropyLoss()

    # Heads
    multi_heads = ['brands', 'context', 'technical']
    single_map  = {
        'human_perception': 'human_perception',
        'sentiment':        'overall_sentiment',
        'humor':            'humor_mechanism',
        'sarcasm':          'sarcasm_level'
    }

    # Train loop
    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        total_loss = 0.0

        # accumulators
        accum = {}
        for h in multi_heads:
            accum[f'{h}_true'], accum[f'{h}_pred'] = [], []
        for h in single_map:
            accum[f'{h}_true'], accum[f'{h}_pred'] = [], []

        for batch in loader:
            imgs    = batch['image'].to(device)
            ocr_ids = batch['ocr_ids'].to(device)
            labels  = batch['labels']

            outputs = model(imgs, ocr_ids)

            # loss
            loss = 0.0
            loss += criterion_bce(outputs['brands'],            labels['identified_brands'].to(device))
            loss += criterion_bce(outputs['context'],           labels['product_context'].to(device))
            loss += criterion_bce(outputs['technical'],         labels['technical_concepts'].to(device))
            loss += criterion_ce (outputs['human_perception'],  labels['human_perception'].to(device))
            loss += criterion_ce (outputs['sentiment'],         labels['overall_sentiment'].to(device))
            loss += criterion_ce (outputs['humor'],             labels['humor_mechanism'].to(device))
            loss += criterion_ce (outputs['sarcasm'],           labels['sarcasm_level'].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # accumulate metrics
            for head in multi_heads:
                probs   = torch.sigmoid(outputs[head]).detach().cpu().numpy()
                preds   = (probs >= 0.5).astype(int)
                true_ml = labels[
                    'identified_brands'    if head=='brands'  else
                    'product_context'      if head=='context' else
                    'technical_concepts'
                ].detach().cpu().numpy()
                accum[f'{head}_pred'].append(preds)
                accum[f'{head}_true'].append(true_ml)

            for head, key in single_map.items():
                logits  = outputs[head].detach().cpu().numpy()
                preds   = np.argmax(logits, axis=1)
                true_sc = labels[key].detach().cpu().numpy()
                accum[f'{head}_pred'].append(preds)
                accum[f'{head}_true'].append(true_sc)

        # epoch end
        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']} — Loss: {avg_loss:.4f}\n")

        # concatenate
        for k,v in accum.items():
            if not v:
                accum[k] = np.array([])
            else:
                accum[k] = np.vstack(v) if v[0].ndim>1 else np.concatenate(v)

        # report primary head first
        y_t = accum['human_perception_true']
        y_p = accum['human_perception_pred']
        acc = accuracy_score(y_t, y_p)
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)
        print(f"{'Human Perception':<20} — Accuracy: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n")

        # multi-label
        for head in multi_heads:
            y_t = accum[f'{head}_true']
            y_p = accum[f'{head}_pred']
            acc_ml = (y_t == y_p).mean()
            p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='samples', zero_division=0)
            print(f"{head.title():<20} — Acc: {acc_ml:.4f}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

        # other single-class
        for head, col in [('sentiment','Overall Sentiment'),
                          ('humor','Humor Mechanism'),
                          ('sarcasm','Sarcasm Level')]:
            y_t = accum[f'{head}_true']
            y_p = accum[f'{head}_pred']
            acc = accuracy_score(y_t, y_p)
            p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)
            print(f"{col:<20} — Accuracy: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

        # save checkpoint
        ckpt = os.path.join(cfg['output_dir'], f"epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/default.yaml'))
    os.makedirs(cfg['output_dir'], exist_ok=True)
    main(cfg)
