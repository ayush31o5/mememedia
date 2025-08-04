# scripts/enhanced_train.py

import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
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
        
        # Initialize tracking
        self.fold_results = []
        self.best_models = []
        
        # Mixed precision training
        self.scaler = GradScaler() if config.get('use_mixed_precision', True) else None
        
        # Initialize wandb if specified
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'meme-classification'),
                config=config,
                name=f"enhanced_multimodal_{config.get('experiment_name', 'default')}"
            )
    
    def create_model(self):
        """Create and initialize the model"""
        model = EnhancedMultiModalMemeNet(
            tokenizer=self.tokenizer,
            config=self.config['model']
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model
    
    def create_optimizer_and_scheduler(self, model, num_training_steps):
        """Create optimizer and learning rate scheduler"""
        
        # Separate parameters for different learning rates
        clip_params = []
        bert_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'clip_model' in name:
                clip_params.append(param)
            elif 'bert_model' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        # Use different learning rates for different components
        optimizer = optim.AdamW([
        {'params': clip_params, 'lr': float(self.config['training']['clip_lr'])},
        {'params': bert_params, 'lr': float(self.config['training']['bert_lr'])},
        {'params': other_params, 'lr': float(self.config['training']['lr'])}
        ], weight_decay=float(self.config['training']['weight_decay']))

        # Cosine annealing with warm restarts
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
            num_cycles=0.5
        )
        
        return optimizer, scheduler
    
    def create_dataloaders(self, train_df, val_df, fold_idx):
        """Create training and validation dataloaders"""
        
        # Training dataset
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
        
        # Validation dataset
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
        
        # Create samplers
        train_sampler = train_dataset.get_sampler()
        
        # Create dataloaders
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
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Loss weights for different heads
        loss_weights = self.config.get('loss_weights', {
            'brands': 1.0, 'context': 1.0, 'technical': 1.0,
            'sentiment': 2.0, 'humor': 2.0, 'sarcasm': 2.0, 'human_perception': 3.0
        })
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            images = batch['image'].to(self.device, non_blocking=True)
            images_pil = batch['image_pil']  # list of PIL images for CLIP
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            # Move labels to device
            labels = {}
            for key, value in batch['labels'].items():
                labels[key] = value.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = model(images, input_ids, attention_mask)
                    loss, loss_dict = model.compute_loss(outputs, labels, loss_weights)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images, input_ids, attention_mask, images_pil=images_pil)
                loss, loss_dict = model.compute_loss(outputs, labels, loss_weights)
                
                loss.backward()
                
                if self.config.get('grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            
            # Update EMA
            if ema:
                ema.update()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % self.config.get('log_interval', 100) == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Fold {fold_idx}, Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, LR: {lr:.2e}")
                
                if self.config.get('use_wandb'):
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': lr,
                        'epoch': epoch,
                        'fold': fold_idx
                    })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, model, val_loader, ema=None):
        """Validate for one epoch"""
        if ema:
            ema.apply_shadow()
        
        model.eval()
        total_loss = 0.0
        all_predictions = {
            'brands': [], 'context': [], 'technical': [],
            'sentiment': [], 'humor': [], 'sarcasm': [], 'human_perception': []
        }
        all_labels = {
            'brands': [], 'context': [], 'technical': [],
            'sentiment': [], 'humor': [], 'sarcasm': [], 'human_perception': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                images_pil = batch['image_pil']  # list of PIL images for CLIP
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                labels = {}
                for key, value in batch['labels'].items():
                    labels[key] = value.to(self.device, non_blocking=True)
                
                # Handle TTA if enabled
                if 'tta_images' in batch:
                    tta_images = batch['tta_images'].to(self.device)
                    batch_size, n_tta, c, h, w = tta_images.shape
                    
                    # Process each TTA sample
                    tta_outputs = []
                    for i in range(n_tta):
                        tta_output = model(tta_images[:, i], input_ids, attention_mask)
                        tta_outputs.append(tta_output)
                    
                    # Average TTA predictions
                    outputs = {}
                    for key in tta_outputs[0].keys():
                        outputs[key] = torch.stack([out[key] for out in tta_outputs]).mean(0)
                else:
                    outputs = model(images, input_ids, attention_mask, images_pil=images_pil)
                
                loss, _ = model.compute_loss(outputs, labels)
                total_loss += loss.item()
                
                # Collect predictions and labels
                for head in ['brands', 'context', 'technical']:
                    preds = torch.sigmoid(outputs[head]).cpu().numpy()
                    preds = (preds >= 0.5).astype(int)
                    all_predictions[head].append(preds)
                    all_labels[head].append(labels[head.replace('brand', 'identified_brand') if head == 'brands' else 
                                                    ('product_context' if head == 'context' else 
                                                     'technical_concepts')].cpu().numpy())
                
                # Single-class heads
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
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_labels)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, metrics
    
    def compute_metrics(self, predictions, labels):
        """Compute comprehensive metrics"""
        metrics = {}
        
        # Concatenate all batches
        for head in predictions.keys():
            predictions[head] = np.concatenate(predictions[head])
            labels[head] = np.concatenate(labels[head])
        
        # Multi-label metrics
        for head in ['brands', 'context', 'technical']:
            y_true = labels[head]
            y_pred = predictions[head]
            
            # Sample-wise metrics
            sample_f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            metrics[f'{head}_sample_f1'] = sample_f1
            metrics[f'{head}_macro_f1'] = macro_f1
        
        # Single-class metrics
        for head in ['sentiment', 'humor', 'sarcasm', 'human_perception']:
            y_true = labels[head]
            y_pred = predictions[head]
            
            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics[f'{head}_accuracy'] = accuracy
            metrics[f'{head}_macro_f1'] = macro_f1
            metrics[f'{head}_weighted_f1'] = weighted_f1
        
        # Overall metric (weighted average of key metrics)
        key_metrics = [
            metrics.get('human_perception_accuracy', 0),
            metrics.get('sentiment_macro_f1', 0),
            metrics.get('humor_macro_f1', 0),
            metrics.get('sarcasm_macro_f1', 0)
        ]
        metrics['overall_score'] = np.mean(key_metrics)
        
        return metrics
    
    def train_fold(self, fold_idx):
        """Train a single fold"""
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}/{len(self.splits)}")
        print(f"{'='*50}")
        
        train_df, val_df = self.splits[fold_idx]
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        # Create model
        model = self.create_model()
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(train_df, val_df, fold_idx)
        
        # Create optimizer and scheduler
        num_training_steps = len(train_loader) * self.config['training']['epochs']
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, num_training_steps)
        
        # Initialize EMA
        ema = ModelEMA(model, decay=self.config.get('ema_decay', 0.9999)) if self.config.get('use_ema', True) else None
        
        # Training loop
        best_score = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, ema, epoch, fold_idx)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(model, val_loader, ema)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Overall Score: {val_metrics['overall_score']:.4f}")
            print(f"Human Perception Acc: {val_metrics.get('human_perception_accuracy', 0):.4f}")
            
            # Log to wandb
            if self.config.get('use_wandb'):
                wandb.log({
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'epoch': epoch,
                    'fold': fold_idx
                })
            
            # Save best model
            if val_metrics['overall_score'] > best_score:
                best_score = val_metrics['overall_score']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(
                    self.config['output_dir'], 
                    f'best_model_fold_{fold_idx}.pth'
                )
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
                
            # Early stopping
            if patience_counter >= self.config.get('patience', 5):
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        return best_score, val_metrics
    
    def train_all_folds(self):
        """Train all folds and ensemble results"""
        fold_scores = []
        
        for fold_idx in range(len(self.splits)):
            score, metrics = self.train_fold(fold_idx)
            fold_scores.append(score)
            self.fold_results.append(metrics)
            
            print(f"\nFold {fold_idx + 1} completed with score: {score:.4f}")
        
        # Print overall results
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Mean Score: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Individual Fold Scores: {[f'{s:.4f}' for s in fold_scores]}")
        
        # Compute average metrics across folds
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
        
        if self.config.get('use_wandb'):
            wandb.log(avg_metrics)
            wandb.log({'final_cv_score': mean_score})
        
        return mean_score, avg_metrics

def main():
    # Load configuration
    config_path = 'configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize trainer
    trainer = AdvancedTrainer(config)
    
    # Train all folds
    final_score, metrics = trainer.train_all_folds()
    
    print(f"\nFinal Cross-Validation Score: {final_score:.4f}")
    
    if final_score > 0.89:
        print(f"🎉 SUCCESS! Achieved target accuracy of 89%+ with {final_score*100:.2f}%")
    else:
        print(f"Target not reached. Current best: {final_score*100:.2f}%")

if __name__ == '__main__':
    main()