import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

import utils.dataset
from utils.dataset import (
    EnhancedMemeDataset, AdvancedTextPreprocessor, 
    AdvancedImagePreprocessor, create_stratified_splits
)
from utils.text_tokenizer import TextTokenizer
from model.multimodal_model import EnhancedMultiModalMemeNet, ModelEMA

class AdvancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize preprocessing components
        self.text_preprocessor = AdvancedTextPreprocessor()
        self.image_preprocessor = AdvancedImagePreprocessor(
            image_size=config.get('image_size', 336)
        )
        
        # Initialize tokenizer
        self.tokenizer = TextTokenizer.from_excel(config['data']['labels_excel'])
        
        # Load data
        self.df = pd.read_excel(config['data']['labels_excel'])
        print(f"Loaded {len(self.df)} samples")
        
        # Create stratified splits
        self.splits = create_stratified_splits(
            self.df, 
            n_splits=config.get('n_folds', 5),
            stratify_col=config.get('stratify_col', 'Audience Perception'),
            random_state=config.get('random_state', 42)
        )
        
        self.fold_results = []
        self.best_models = []
        self.scaler = GradScaler() if config.get('use_mixed_precision', True) else None

    def create_model(self):
        model = EnhancedMultiModalMemeNet(
            tokenizer=self.tokenizer,
            config=self.config['model']
        )
        model = model.to(self.device)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        return model

    def create_optimizer_and_scheduler(self, model, num_training_steps):
        clip_params, bert_params, other_params = [], [], []
        for name, param in model.named_parameters():
            if 'clip_model' in name:
                clip_params.append(param)
            elif 'bert_model' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': clip_params, 'lr': float(self.config['training']['clip_lr'])},
            {'params': bert_params, 'lr': float(self.config['training']['bert_lr'])},
            {'params': other_params, 'lr': float(self.config['training']['lr'])}
        ], weight_decay=float(self.config['training']['weight_decay']))
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
            num_cycles=0.5
        )
        return optimizer, scheduler

    def create_dataloaders(self, train_df, val_df, fold_idx):
        train_dataset = EnhancedMemeDataset(
            df=train_df,
            img_dir=self.config['data']['raw_dir'],
            tokenizer=self.tokenizer,
            text_preprocessor=self.text_preprocessor,
            image_preprocessor=self.image_preprocessor,
            max_length=self.config['model']['max_length'],
            is_training=True,
            use_tta=False
        )
        val_dataset = EnhancedMemeDataset(
            df=val_df,
            img_dir=self.config['data']['raw_dir'],
            tokenizer=self.tokenizer,
            text_preprocessor=self.text_preprocessor,
            image_preprocessor=self.image_preprocessor,
            max_length=self.config['model']['max_length'],
            is_training=False,
            use_tta=self.config.get('use_tta', False)
        )
        train_sampler = train_dataset.get_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['val_batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        return train_loader, val_loader

    def train_epoch(self, model, train_loader, optimizer, scheduler, ema, epoch, fold_idx):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        loss_weights = self.config.get('loss_weights', {
            'brands': 1.0, 'context': 1.0, 'technical': 1.0,
            'sentiment': 2.0, 'humor': 2.0, 'sarcasm': 2.0, 'human_perception': 3.0
        })

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            images_pil = batch['image_pil']
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = {key: value.to(self.device, non_blocking=True) for key, value in batch['labels'].items()}

            if self.scaler:
                with autocast():
                    outputs = model(images, input_ids, attention_mask)
                    loss, loss_dict = model.compute_loss(outputs, labels, loss_weights)
                self.scaler.scale(loss).backward()
                if self.config.get('grad_clip_norm'):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip_norm'])
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images, input_ids, attention_mask, images_pil=images_pil)
                loss, loss_dict = model.compute_loss(outputs, labels, loss_weights)
                loss.backward()
                if self.config.get('grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip_norm'])
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            if ema:
                ema.update()
            total_loss += loss.item()

            if batch_idx % self.config.get('log_interval', 100) == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Fold {fold_idx}, Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, LR: {lr:.2e}")

        return total_loss / num_batches

    def validate_epoch(self, model, val_loader, ema=None):
        if ema:
            ema.apply_shadow()

        model.eval()
        total_loss = 0.0
        all_predictions = {k: [] for k in ['brands', 'context', 'technical', 'sentiment', 'humor', 'sarcasm', 'human_perception']}
        all_labels = {k: [] for k in all_predictions.keys()}

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                images_pil = batch['image_pil']
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = {key: value.to(self.device, non_blocking=True) for key, value in batch['labels'].items()}

                if 'tta_images' in batch:
                    tta_images = batch['tta_images'].to(self.device)
                    batch_size, n_tta, c, h, w = tta_images.shape
                    tta_outputs = [model(tta_images[:, i], input_ids, attention_mask) for i in range(n_tta)]
                    outputs = {key: torch.stack([out[key] for out in tta_outputs]).mean(0) for key in tta_outputs[0].keys()}
                else:
                    outputs = model(images, input_ids, attention_mask, images_pil=images_pil)

                loss, _ = model.compute_loss(outputs, labels)
                total_loss += loss.item()

                for head in ['brands', 'context', 'technical']:
                    preds = (torch.sigmoid(outputs[head]).cpu().numpy() >= 0.5).astype(int)
                    all_predictions[head].append(preds)
                    all_labels[head].append(labels[head].cpu().numpy())

                single_heads = {
                    'sentiment': 'overall_sentiment',
                    'humor': 'humor_mechanism',
                    'sarcasm': 'sarcasm_level',
                    'human_perception': 'human_perception'
                }
                for head, label_key in single_heads.items():
                    preds = torch.argmax(outputs[head], dim=1).cpu().numpy()
                    all_predictions[head].append(preds)
                    all_labels[head].append(labels[label_key].cpu().numpy())

        if ema:
            ema.restore()
        metrics = self.compute_metrics(all_predictions, all_labels)
        return total_loss / len(val_loader), metrics

    def compute_metrics(self, predictions, labels):
        metrics = {}
        for head in predictions.keys():
            predictions[head] = np.concatenate(predictions[head])
            labels[head] = np.concatenate(labels[head])

        for head in ['brands', 'context', 'technical']:
            y_true, y_pred = labels[head], predictions[head]
            metrics[f'{head}_sample_f1'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
            metrics[f'{head}_macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        for head in ['sentiment', 'humor', 'sarcasm', 'human_perception']:
            y_true, y_pred = labels[head], predictions[head]
            metrics[f'{head}_accuracy'] = accuracy_score(y_true, y_pred)
            metrics[f'{head}_macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics[f'{head}_weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        key_metrics = [
            metrics.get('human_perception_accuracy', 0),
            metrics.get('sentiment_macro_f1', 0),
            metrics.get('humor_macro_f1', 0),
            metrics.get('sarcasm_macro_f1', 0)
        ]
        metrics['overall_score'] = np.mean(key_metrics)
        return metrics

    def train_fold(self, fold_idx):
        print(f"\n{'='*50}\nTraining Fold {fold_idx + 1}/{len(self.splits)}\n{'='*50}")
        train_df, val_df = self.splits[fold_idx]
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        model = self.create_model()
        train_loader, val_loader = self.create_dataloaders(train_df, val_df, fold_idx)
        num_training_steps = len(train_loader) * self.config['training']['epochs']
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, num_training_steps)
        ema = ModelEMA(model, decay=self.config.get('ema_decay', 0.9999)) if self.config.get('use_ema', True) else None

        best_score, best_model_state, patience_counter = 0.0, None, 0

        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, ema, epoch, fold_idx)
            val_loss, val_metrics = self.validate_epoch(model, val_loader, ema)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Overall Score: {val_metrics['overall_score']:.4f}, Human Perception Acc: {val_metrics.get('human_perception_accuracy', 0):.4f}")

            if val_metrics['overall_score'] > best_score:
                best_score = val_metrics['overall_score']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                checkpoint_path = os.path.join(self.config['output_dir'], f'best_model_fold_{fold_idx}.pth')
                torch.save({
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'fold': fold_idx,
                    'score': best_score,
                    'metrics': val_metrics
                }, checkpoint_path)
                print(f"New best score: {best_score:.4f}, saved to {checkpoint_path}")
            else:
                patience_counter += 1

            if patience_counter >= self.config.get('patience', 5):
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        return best_score, val_metrics

    def train_all_folds(self):
        fold_scores = []
        for fold_idx in range(len(self.splits)):
            score, metrics = self.train_fold(fold_idx)
            fold_scores.append(score)
            self.fold_results.append(metrics)
            print(f"\nFold {fold_idx + 1} completed with score: {score:.4f}")
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"\n{'='*60}\nCROSS-VALIDATION RESULTS\n{'='*60}")
        print(f"Mean Score: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Individual Fold Scores: {[f'{s:.4f}' for s in fold_scores]}")
        avg_metrics = {}
        for key in self.fold_results[0].keys():
            values = [fold_result[key] for fold_result in self.fold_results]
            avg_metrics[f'cv_{key}_mean'] = np.mean(values)
            avg_metrics[f'cv_{key}_std'] = np.std(values)
        print("\nDetailed Metrics (Mean ± Std):")
        for key, value in avg_metrics.items():
            if 'mean' in key:
                std_key = key.replace('mean', 'std')
                print(f"{key}: {value:.4f} ± {avg_metrics[std_key]:.4f}")
        return mean_score, avg_metrics

def main():
    config_path = 'configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    os.makedirs(config['output_dir'], exist_ok=True)
    trainer = AdvancedTrainer(config)
    final_score, metrics = trainer.train_all_folds()
    print(f"\nFinal Cross-Validation Score: {final_score:.4f}")
    if final_score > 0.89:
        print(f"🎉 SUCCESS! Achieved target accuracy of 89%+ with {final_score*100:.2f}%")
    else:
        print(f"Target not reached. Current best: {final_score*100:.2f}%")

if __name__ == '__main__':
    main()