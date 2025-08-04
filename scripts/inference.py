# scripts/enhanced_inference.py

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from utils.dataset import (
    AdvancedTextPreprocessor, AdvancedImagePreprocessor
)
from utils.text_tokenizer import TextTokenizer
from model.multimodal_model import EnhancedMultiModalMemeNet
from transformers import BertTokenizer

class EnsembleInference:
    def __init__(self, config_path: str, checkpoint_dir: str):
        """Initialize ensemble inference"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize preprocessing components
        self.text_preprocessor = AdvancedTextPreprocessor()
        self.image_preprocessor = AdvancedImagePreprocessor(
            image_size=self.config.get('image_size', 336)
        )
        
        # Initialize tokenizer
        self.tokenizer = TextTokenizer.from_excel(self.config['data']['labels_excel'])
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load ensemble models
        self.models = self._load_ensemble_models(checkpoint_dir)
        print(f"Loaded {len(self.models)} models for ensemble inference")
        
        # Set models to evaluation mode
        for model in self.models:
            model.eval()
    
    def _load_ensemble_models(self, checkpoint_dir: str) -> List[EnhancedMultiModalMemeNet]:
        """Load all fold models for ensemble inference"""
        models = []
        
        # Find all model checkpoints
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('best_model_fold_') and f.endswith('.pth')]
        checkpoint_files.sort()  # Ensure consistent ordering
        
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            
            # Create model
            model = EnhancedMultiModalMemeNet(
                tokenizer=self.tokenizer,
                config=self.config['model']
            )
            
            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            models.append(model)
            print(f"Loaded model from {checkpoint_file}")
        
        return models
    
    def preprocess_inputs(self, image_path: str, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess image and text inputs"""
        
        # Process image
        if self.config['inference'].get('tta_enabled', True):
            images = self.image_preprocessor.apply_tta(image_path)
        else:
            images = [self.image_preprocessor.preprocess_image(image_path, is_training=False)]
        
        # Process text  
        cleaned_text = self.text_preprocessor.clean_text(text)
        
        # Tokenize with BERT
        encoding = self.bert_tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.config['model']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'images': torch.stack(images),  # [n_tta, C, H, W]
            'input_ids': encoding['input_ids'],  # [1, seq_len]
            'attention_mask': encoding['attention_mask']  # [1, seq_len]
        }
    
    def predict_single_model(self, model: EnhancedMultiModalMemeNet, 
                           inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make prediction with a single model"""
        
        images = inputs['images'].to(self.device)  # [n_tta, C, H, W]
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        n_tta = images.shape[0]
        
        with torch.no_grad():
            if n_tta > 1:  # TTA enabled
                tta_predictions = []
                
                for i in range(n_tta):
                    # Repeat text inputs for each TTA image
                    repeated_input_ids = input_ids.repeat(1, 1)
                    repeated_attention_mask = attention_mask.repeat(1, 1)
                    
                    pred = model(
                        images[i:i+1], 
                        repeated_input_ids, 
                        repeated_attention_mask
                    )
                    tta_predictions.append(pred)
                
                # Average TTA predictions
                averaged_pred = {}
                for key in tta_predictions[0].keys():
                    averaged_pred[key] = torch.stack([p[key] for p in tta_predictions]).mean(0)
                
                return averaged_pred
            
            else:  # No TTA
                return model(images, input_ids, attention_mask)
    
    def ensemble_predictions(self, all_predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine predictions from multiple models"""
        
        ensemble_method = self.config['inference'].get('ensemble_method', 'weighted_average')
        
        if ensemble_method == 'simple_average':
            return self._simple_average_ensemble(all_predictions)
        elif ensemble_method == 'weighted_average':
            return self._weighted_average_ensemble(all_predictions)
        elif ensemble_method == 'voting':
            return self._voting_ensemble(all_predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    def _simple_average_ensemble(self, all_predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Simple average ensemble"""
        ensemble_pred = {}
        
        for key in all_predictions[0].keys():
            # Stack predictions from all models
            stacked = torch.stack([pred[key] for pred in all_predictions])
            # Average across models
            ensemble_pred[key] = stacked.mean(0)
        
        return ensemble_pred
    
    def _weighted_average_ensemble(self, all_predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Weighted average ensemble with learned weights"""
        
        # Define weights based on validation performance (you can load these from checkpoints)
        # For now, using equal weights with slight preference for later folds
        model_weights = torch.softmax(torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4]), dim=0)
        model_weights = model_weights[:len(all_predictions)]  # Adjust if fewer models
        model_weights = model_weights.to(self.device)
        
        ensemble_pred = {}
        
        for key in all_predictions[0].keys():
            # Stack predictions from all models
            stacked = torch.stack([pred[key] for pred in all_predictions])  # [n_models, batch, ...]
            
            # Apply weights
            weighted = stacked * model_weights.view(-1, 1, *([1] * (stacked.dim() - 2)))
            ensemble_pred[key] = weighted.sum(0)
        
        return ensemble_pred
    
    def _voting_ensemble(self, all_predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Voting ensemble for classification tasks"""
        ensemble_pred = {}
        
        for key in all_predictions[0].keys():
            stacked = torch.stack([pred[key] for pred in all_predictions])
            
            if key in ['brands', 'context', 'technical']:  # Multi-label
                # Apply sigmoid and threshold
                probs = torch.sigmoid(stacked)
                votes = (probs >= 0.5).float()
                # Majority voting
                ensemble_pred[key] = (votes.mean(0) >= 0.5).float()
                # Convert back to logits for consistency
                ensemble_pred[key] = torch.log(ensemble_pred[key] / (1 - ensemble_pred[key] + 1e-8))
            
            else:  # Single-label classification
                # Soft voting (average probabilities)
                probs = F.softmax(stacked, dim=-1)
                avg_probs = probs.mean(0)
                # Convert back to logits
                ensemble_pred[key] = torch.log(avg_probs + 1e-8)
        
        return ensemble_pred
    
    def predict(self, image_path: str, text: str) -> Dict[str, any]:
        """Make ensemble prediction for a single sample"""
        
        # Preprocess inputs
        inputs = self.preprocess_inputs(image_path, text)
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            pred = self.predict_single_model(model, inputs)
            all_predictions.append(pred)
        
        # Ensemble predictions
        ensemble_pred = self.ensemble_predictions(all_predictions)
        
        # Convert to interpretable format
        results = self._interpret_predictions(ensemble_pred)
        
        return results
    
    def _interpret_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Convert model outputs to interpretable results"""
        
        results = {}
        confidence_threshold = self.config['inference'].get('confidence_threshold', 0.5)
        
        # Multi-label predictions
        multi_label_heads = {
            'brands': 'identified_brands',
            'context': 'product_context', 
            'technical': 'technical_concepts'
        }
        
        for head, vocab_key in multi_label_heads.items():
            vocab = getattr(self.tokenizer, f'{vocab_key.split("_")[1] if "_" in vocab_key else vocab_key}_vocab', [])
            
            if head in predictions:
                probs = torch.sigmoid(predictions[head]).cpu().numpy().flatten()
                predicted_labels = []
                confidences = []
                
                for i, (prob, label) in enumerate(zip(probs, vocab)):
                    if prob >= confidence_threshold:
                        predicted_labels.append(label)
                        confidences.append(float(prob))
                
                results[head] = {
                    'predicted_labels': predicted_labels,
                    'confidences': confidences,
                    'all_probabilities': {label: float(prob) for label, prob in zip(vocab, probs)}
                }
        
        # Single-label predictions
        single_label_heads = {
            'sentiment': 'sentiment_vocab',
            'humor': 'humor_mechanism_vocab',
            'sarcasm': 'sarcasm_level_vocab', 
            'human_perception': 'human_perception_vocab'
        }
        
        for head, vocab_attr in single_label_heads.items():
            vocab = getattr(self.tokenizer, vocab_attr, [])
            
            if head in predictions:
                probs = F.softmax(predictions[head], dim=-1).cpu().numpy().flatten()
                predicted_idx = np.argmax(probs)
                
                results[head] = {
                    'predicted_label': vocab[predicted_idx] if predicted_idx < len(vocab) else 'unknown',
                    'confidence': float(probs[predicted_idx]),
                    'all_probabilities': {label: float(prob) for label, prob in zip(vocab, probs)}
                }
        
        return results
    
    def predict_batch(self, image_paths: List[str], texts: List[str]) -> List[Dict[str, any]]:
        """Make ensemble predictions for a batch of samples"""
        
        results = []
        for image_path, text in zip(image_paths, texts):
            try:
                result = self.predict(image_path, text)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def predict_from_dataframe(self, df: pd.DataFrame, 
                             image_dir: str, 
                             save_path: Optional[str] = None) -> pd.DataFrame:
        """Make predictions for a dataframe of samples"""
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                meme_id = str(row['Meme ID']).strip()
                
                # Find image path
                image_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                    potential_path = os.path.join(image_dir, f"{meme_id}{ext}")
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if image_path is None:
                    results.append({'meme_id': meme_id, 'error': 'Image not found'})
                    continue
                
                text = str(row.get('OCR Text', ''))
                prediction = self.predict(image_path, text)
                
                # Flatten results for DataFrame
                flat_result = {'meme_id': meme_id}
                
                for head, head_result in prediction.items():
                    if 'predicted_label' in head_result:  # Single-label
                        flat_result[f'{head}_prediction'] = head_result['predicted_label']
                        flat_result[f'{head}_confidence'] = head_result['confidence']
                    elif 'predicted_labels' in head_result:  # Multi-label
                        flat_result[f'{head}_predictions'] = ', '.join(head_result['predicted_labels'])
                        flat_result[f'{head}_avg_confidence'] = np.mean(head_result['confidences']) if head_result['confidences'] else 0.0
                
                results.append(flat_result)
                
                if idx % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} samples")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results.append({'meme_id': meme_id, 'error': str(e)})
        
        results_df = pd.DataFrame(results)
        
        if save_path:
            results_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description='Enhanced Meme Classification Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--image_path', type=str, help='Path to single image for prediction')
    parser.add_argument('--text', type=str, help='Text content for single prediction')
    parser.add_argument('--input_csv', type=str, help='CSV file with samples to predict')
    parser.add_argument('--image_dir', type=str, help='Directory containing images for batch prediction')
    parser.add_argument('--output_csv', type=str, help='Output CSV file for batch predictions')
    
    args = parser.parse_args()
    
    # Initialize inference
    ensemble = EnsembleInference(args.config, args.checkpoint_dir)
    
    if args.image_path and args.text:
        # Single prediction
        print(f"Making prediction for image: {args.image_path}")
        result = ensemble.predict(args.image_path, args.text)
        
        print("\nPrediction Results:")
        print("="*50)
        
        for head, head_result in result.items():
            print(f"\n{head.upper()}:")
            if 'predicted_label' in head_result:
                print(f"  Prediction: {head_result['predicted_label']}")
                print(f"  Confidence: {head_result['confidence']:.4f}")
            elif 'predicted_labels' in head_result:
                print(f"  Predictions: {', '.join(head_result['predicted_labels'])}")
                print(f"  Confidences: {[f'{c:.4f}' for c in head_result['confidences']]}")
    
    elif args.input_csv and args.image_dir:
        # Batch prediction
        print(f"Making batch predictions from: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        
        results_df = ensemble.predict_from_dataframe(
            df, 
            args.image_dir, 
            args.output_csv
        )
        
        print(f"\nBatch prediction completed!")
        print(f"Results shape: {results_df.shape}")
        
        if args.output_csv:
            print(f"Results saved to: {args.output_csv}")
    
    else:
        print("Please provide either --image_path and --text for single prediction, "
              "or --input_csv and --image_dir for batch prediction")

if __name__ == '__main__':
    main()