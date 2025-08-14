# scripts/train.py

import os
import random
import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.dataset import MemeDataset
from utils.text_tokenizer import TextTokenizer
from model.multi_model import MultiMemeNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(cfg):
    set_seed(cfg.get('seed', 42))

    # Device
    desired = cfg.get('device', 'cpu')
    device  = torch.device(desired if (desired=='cpu' or torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = TextTokenizer.from_excel(cfg['data']['labels_excel'])

    # Dataset
    max_ocr_len = int(cfg['model'].get('max_ocr_len', 64))
    max_hp_len  = int(cfg['model'].get('max_hp_len', 64))
    dataset   = MemeDataset(
        img_dir=cfg['data']['raw_dir'],
        labels_excel=cfg['data']['labels_excel'],
        tokenizer=tokenizer,
        max_len=max_ocr_len,
        max_hp_len=max_hp_len,
    )

    # Train/val split (optional)
    val_ratio = float(cfg['training'].get('val_ratio', 0.1))
    if 0 < val_ratio < 0.9 and len(dataset) > 1:
        n_val = max(1, int(len(dataset)*val_ratio))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds,
                              batch_size=int(cfg['training']['batch_size']),
                              shuffle=True,
                              num_workers=int(cfg['training'].get('num_workers', 4)),
                              pin_memory=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds,
                                batch_size=int(cfg['training'].get('eval_batch_size', cfg['training']['batch_size'])),
                                shuffle=False,
                                num_workers=int(cfg['training'].get('num_workers', 4)),
                                pin_memory=True)

    # Model
    model = MultiMemeNet(
        tokenizer,
        transformer_dim=int(cfg['model'].get('transformer_dim', 256)),
        n_heads=int(cfg['model'].get('n_heads', 4)),
        n_layers=int(cfg['model'].get('n_layers', 2)),
        decoder_layers=int(cfg['model'].get('decoder_layers', 2)),
        dropout=float(cfg['model'].get('dropout', 0.1)),
        pretrained_cnn=bool(cfg['model'].get('pretrained_cnn', True))
    ).to(device)

    # Optimizer & Scheduler
    lr        = float(cfg['training'].get('lr', 1e-3))
    weight_decay = float(cfg['training'].get('weight_decay', 0.0))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg['training'].get('epochs', 10)))

    # Losses
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce  = nn.CrossEntropyLoss()
    criterion_seq = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    use_amp = bool(cfg['training'].get('amp', False))
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Heads
    multi_heads = ['brands', 'context', 'technical']
    single_map  = {
        'sentiment':        'overall_sentiment',
        'humor':            'humor_mechanism',
        'sarcasm':          'sarcasm_level'
    }

    best_val = float('inf')
    epochs = int(cfg['training'].get('epochs', 5))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # accumulators for metrics
        accum = {}
        for h in multi_heads:
            accum[f'{h}_true'], accum[f'{h}_pred'] = [], []
        for h in single_map:
            accum[f'{h}_true'], accum[f'{h}_pred'] = [], []

        for batch in train_loader:
            imgs    = batch['image'].to(device, non_blocking=True)
            ocr_ids = batch['ocr_ids'].to(device, non_blocking=True)
            hp_inp  = batch['hp_inp'].to(device, non_blocking=True)
            hp_tgt  = batch['hp_tgt'].to(device, non_blocking=True)
            labels  = batch['labels']

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(imgs, ocr_ids, hp_inp=hp_inp)

                # loss
                loss = 0.0
                loss += criterion_bce(outputs['brands'],            labels['identified_brands'].to(device))
                loss += criterion_bce(outputs['context'],           labels['product_context'].to(device))
                loss += criterion_bce(outputs['technical'],         labels['technical_concepts'].to(device))
                loss += criterion_ce (outputs['sentiment'],         labels['overall_sentiment'].to(device))
                loss += criterion_ce (outputs['humor'],             labels['humor_mechanism'].to(device))
                loss += criterion_ce (outputs['sarcasm'],           labels['sarcasm_level'].to(device))

                # sequence CE (flatten)
                hp_logits = outputs['human_perception']  # [B, T, V]
                V = hp_logits.size(-1)
                loss += float(cfg['training'].get('hp_loss_weight', 1.0)) * \
                        criterion_seq(hp_logits.view(-1, V), hp_tgt.view(-1))

            scaler.scale(loss).backward()
            # clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg['training'].get('grad_clip', 1.0)))
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().item())

            # accumulate metrics (classification only)
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
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_loss:.4f}")

        # concatenate for metrics
        for k,v in accum.items():
            if not v:
                accum[k] = np.array([])
            else:
                accum[k] = np.vstack(v) if v[0].ndim>1 else np.concatenate(v)

        # report classification metrics
        for head, col in [('sentiment','Overall Sentiment'),
                          ('humor','Humor Mechanism'),
                          ('sarcasm','Sarcasm Level')]:
            y_t = accum[f'{head}_true']
            y_p = accum[f'{head}_pred']
            if y_t.size:
                acc = accuracy_score(y_t, y_p)
                p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)
                print(f"{col:<20} — Acc: {acc:.4f}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

        for head in ['brands','context','technical']:
            y_t = accum[f'{head}_true']
            y_p = accum[f'{head}_pred']
            if y_t.size:
                acc_ml = (y_t == y_p).mean()
                p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='samples', zero_division=0)
                print(f"{head.title():<20} — Acc: {acc_ml:.4f}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

        # quick sample generations
        model.eval()
        with torch.no_grad():
            sample_batches = 0
            for batch in train_loader:
                imgs    = batch['image'].to(device)
                ocr_ids = batch['ocr_ids'].to(device)
                gen_ids = model.generate_human_perception(imgs, ocr_ids, max_len=max_hp_len)  # [B, T']
                for i in range(min(2, gen_ids.size(0))):
                    print("HP GEN:", tokenizer.decode(gen_ids[i].cpu().tolist()))
                sample_batches += 1
                if sample_batches >= 1:
                    break
        model.train()

        # Validation loss (optional)
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    imgs    = batch['image'].to(device, non_blocking=True)
                    ocr_ids = batch['ocr_ids'].to(device, non_blocking=True)
                    hp_inp  = batch['hp_inp'].to(device, non_blocking=True)
                    hp_tgt  = batch['hp_tgt'].to(device, non_blocking=True)
                    labels  = batch['labels']

                    outputs = model(imgs, ocr_ids, hp_inp=hp_inp)

                    loss = 0.0
                    loss += criterion_bce(outputs['brands'],            labels['identified_brands'].to(device))
                    loss += criterion_bce(outputs['context'],           labels['product_context'].to(device))
                    loss += criterion_bce(outputs['technical'],         labels['technical_concepts'].to(device))
                    loss += criterion_ce (outputs['sentiment'],         labels['overall_sentiment'].to(device))
                    loss += criterion_ce (outputs['humor'],             labels['humor_mechanism'].to(device))
                    loss += criterion_ce (outputs['sarcasm'],           labels['sarcasm_level'].to(device))
                    hp_logits = outputs['human_perception']
                    V = hp_logits.size(-1)
                    loss += float(cfg['training'].get('hp_loss_weight', 1.0)) * \
                            criterion_seq(hp_logits.view(-1, V), hp_tgt.view(-1))

                    val_loss += float(loss.item())
            val_loss /= max(1, len(val_loader))
            print(f"Validation Loss: {val_loss:.4f}")
            model.train()

        # step scheduler
        try:
            sched.step()
        except Exception:
            pass

        # save checkpoint (keep best by val if available else by train loss)
        score = val_loss if val_loader is not None else avg_loss
        is_best = score < best_val
        if is_best:
            best_val = score
        os.makedirs(cfg['output_dir'], exist_ok=True)
        ckpt_path = os.path.join(cfg['output_dir'], f"epoch{epoch}{'-best' if is_best else ''}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved:", ckpt_path)

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/default.yaml'))
    os.makedirs(cfg['output_dir'], exist_ok=True)
    main(cfg)
