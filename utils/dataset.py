# utils/enhanced_dataset.py

import os
import torch
import pandas as pd
import numpy as np
import cv2
import re
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
import pytesseract
from typing import Dict, List, Tuple, Optional

class AdvancedTextPreprocessor:
    def __init__(self):
        self.contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
        
        # Common OCR errors and corrections
        self.ocr_corrections = {
            "0": "o", "5": "s", "1": "l", "8": "b",
            "rn": "m", "vv": "w", "ii": "n", "cl": "d"
        }
        
    def clean_text(self, text: str) -> str:
        if not text or text == 'nan':
            return ""
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\']', '', text)
        
        # Fix common OCR errors
        for error, correction in self.ocr_corrections.items():
            text = text.replace(error, correction)
        
        return text.strip()
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract additional text features for better understanding"""
        if not text:
            return {'length': 0, 'word_count': 0, 'avg_word_length': 0, 
                   'punctuation_ratio': 0, 'capital_ratio': 0}
        
        words = text.split()
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        punctuation_count = len(re.findall(r'[.,!?;:]', text))
        capital_count = len(re.findall(r'[A-Z]', text))
        
        return {
            'length': len(text),
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'punctuation_ratio': punctuation_count / len(text) if text else 0,
            'capital_ratio': capital_count / len(text) if text else 0
        }

class AdvancedImagePreprocessor:
    def __init__(self, image_size: int = 336):
        self.image_size = image_size
        
        # Advanced augmentation pipeline
        self.train_transform = A.Compose([
            # Geometric transformations
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
            ),
            
            # Color augmentations
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            ], p=0.7),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=0.5),
            ], p=0.3),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            
            # Advanced augmentations
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.5),
            ], p=0.2),
            
            # Cutout/Dropout
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                A.GridDropout(ratio=0.2, p=0.5),
            ], p=0.3),
            
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        # Test Time Augmentation
        self.tta_transforms = [
            A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.ColorJitter(brightness=0.1, contrast=0.1, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
        ]
    
    def preprocess_image(self, image_path: str, is_training: bool = True) -> torch.Tensor:
        """Enhanced image preprocessing with error handling"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Image not found at {image_path}, using blank image")
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply denoising
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            image = cv2.merge([l, a, b])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
            
            # Apply augmentation
            if is_training:
                transformed = self.train_transform(image=image)
                assert transformed['image'].shape == (3, self.image_size, self.image_size), \
                    f"Invalid shape: {transformed['image'].shape} for {image_path}"

            else:
                transformed = self.val_transform(image=image)
                assert transformed['image'].shape == (3, self.image_size, self.image_size), \
                    f"Invalid shape: {transformed['image'].shape} for {image_path}"

                
            return transformed['image']
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return a blank tensor if image processing fails
            if is_training:
                return torch.zeros(3, self.image_size, self.image_size)
            else:
                return torch.zeros(3, self.image_size, self.image_size)
    
    def apply_tta(self, image_path: str) -> List[torch.Tensor]:
        """Apply Test Time Augmentation"""
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            tta_images = []
            for transform in self.tta_transforms:
                transformed = transform(image=image)
                tta_images.append(transformed['image'])
            
            return tta_images
        except Exception as e:
            print(f"Error in TTA for {image_path}: {e}")
            return [torch.zeros(3, self.image_size, self.image_size) for _ in self.tta_transforms]

class EnhancedMemeDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 img_dir: str, 
                 tokenizer,
                 text_preprocessor: AdvancedTextPreprocessor,
                 image_preprocessor: AdvancedImagePreprocessor,
                 max_length: int = 128,
                 is_training: bool = True,
                 use_tta: bool = False):
        
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        self.max_length = max_length
        self.is_training = is_training
        self.use_tta = use_tta
        
        # Initialize BERT tokenizer for advanced text processing
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Precompute class weights for balanced sampling
        self.class_weights = self._compute_class_weights()
        
    def _compute_class_weights(self) -> Dict[str, torch.Tensor]:
        """Compute class weights for balanced training"""
        weights = {}
        
        # Single-class columns
        single_class_cols = {
            'sentiment': 'Overall Sentiment',
            'humor': 'Humor Mechanism', 
            'sarcasm': 'Sarcasm Level',
            'human_perception': 'Human Perception'
        }
        
        for key, col in single_class_cols.items():
            if col in self.df.columns:
                values = self.df[col].dropna()
                unique_vals = values.unique()
                class_counts = values.value_counts()
                total_samples = len(values)
                
                # Inverse frequency weighting
                class_weights = total_samples / (len(unique_vals) * class_counts)
                weights[key] = torch.tensor([class_weights.get(val, 1.0) for val in unique_vals])
        
        return weights
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            meme_id = str(row['Meme ID']).strip()
            
            # Load and preprocess image
            image_path = self._find_image_path(meme_id)
            image_pil = Image.open(image_path).convert("RGB")
            if self.use_tta:
                images = self.image_preprocessor.apply_tta(image_path)
                image = images[0]  # Use first augmentation as primary
            else:
                image = self.image_preprocessor.preprocess_image(image_path, self.is_training)
            
            # Process text
            ocr_text = str(row.get('OCR Text', ''))
            cleaned_text = self.text_preprocessor.clean_text(ocr_text)
            
            # Tokenize with BERT tokenizer
            encoding = self.bert_tokenizer(
                cleaned_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Prepare labels
            labels = self._prepare_labels(row)
            
            # Extract additional features
            text_features = self.text_preprocessor.extract_text_features(cleaned_text)
            
            sample = {
                'image': image,
                'image_pil': image_pil,
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': labels,
                'text_features': torch.tensor(list(text_features.values()), dtype=torch.float),
                'meme_id': meme_id
            }
            
            if self.use_tta:
                sample['tta_images'] = torch.stack(images)
            assert sample['image'].shape == (3, self.image_preprocessor.image_size, self.image_preprocessor.image_size), \
                f"[Dataset Error] Invalid image shape for index {idx}: {sample['image'].shape}"

            return sample
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample
            return self._get_dummy_sample()
    
    def _find_image_path(self, meme_id: str) -> str:
        """Find image path with multiple extensions"""
        extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        for ext in extensions:
            path = os.path.join(self.img_dir, f"{meme_id}{ext}")
            if os.path.exists(path):
                return path
        
        # If no image found, create a dummy image
        dummy_path = os.path.join(self.img_dir, f"{meme_id}_dummy.png")
        if not os.path.exists(dummy_path):
            dummy_image = Image.new('RGB', (224, 224), color='white')
            dummy_image.save(dummy_path)
        return dummy_path
    
    def _prepare_labels(self, row) -> Dict[str, torch.Tensor]:
        """Prepare labels with proper encoding"""
        labels = {}
        
        # Multi-label fields
        multi_label_fields = {
            'identified_brands': 'Identified Brands',
            'product_context': 'Product Context',
            'technical_concepts': 'Technical Concepts'
        }
        
        for key, col in multi_label_fields.items():
            if col in row and hasattr(self.tokenizer, f'{key.split("_")[1] if "_" in key else key}_vocab'):
                vocab = getattr(self.tokenizer, f'{key.split("_")[1] if "_" in key else key}_vocab')
                labels[key] = torch.FloatTensor(
                    self.tokenizer.encode_multilabel(str(row[col]), vocab)
                )
            else:
                labels[key] = torch.zeros(10)  # Default size
        
        # Single-class fields
        single_class_fields = {
            'overall_sentiment': 'Overall Sentiment',
            'humor_mechanism': 'Humor Mechanism',
            'sarcasm_level': 'Sarcasm Level', 
            'human_perception': 'Human Perception'
        }
        
        for key, col in single_class_fields.items():
            if col in row and hasattr(self.tokenizer, f'{key.split("_")[0]}_vocab'):
                vocab = getattr(self.tokenizer, f'{key.split("_")[0]}_vocab')
                labels[key] = torch.tensor(
                    self.tokenizer.class_to_id(str(row[col]), vocab),
                    dtype=torch.long
                )
            else:
                labels[key] = torch.tensor(0, dtype=torch.long)
        
        return labels
    
    def _get_dummy_sample(self):
        """Return a dummy sample for error cases"""
        return {
            'image': torch.zeros(3, 336, 336),
            'input_ids': torch.zeros(128, dtype=torch.long),
            'attention_mask': torch.zeros(128, dtype=torch.long),
            'labels': {
                'identified_brands': torch.zeros(10),
                'product_context': torch.zeros(10),
                'technical_concepts': torch.zeros(10),
                'overall_sentiment': torch.tensor(0, dtype=torch.long),
                'humor_mechanism': torch.tensor(0, dtype=torch.long),
                'sarcasm_level': torch.tensor(0, dtype=torch.long),
                'human_perception': torch.tensor(0, dtype=torch.long)
            },
            'text_features': torch.zeros(5),
            'meme_id': 'dummy'
        }
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Create weighted sampler for balanced training"""
        if not self.is_training:
            return None
        
        # Use human_perception as the primary class for weighting
        if 'Human Perception' in self.df.columns:
            labels = self.df['Human Perception'].fillna('unknown')
            unique_labels = labels.unique()
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            # Compute sample weights
            label_counts = labels.value_counts()
            total_samples = len(labels)
            weights = [total_samples / label_counts[label] for label in labels]
            
            return WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
        
        return None

def create_stratified_splits(df: pd.DataFrame, 
                           n_splits: int = 5, 
                           stratify_col: str = 'Human Perception',
                           random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create stratified k-fold splits"""
    
    if stratify_col not in df.columns:
        print(f"Warning: {stratify_col} not found, using random splits")
        # Fallback to random splits
        splits = []
        indices = np.arange(len(df))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        fold_size = len(df) // n_splits
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else len(df)
            
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            splits.append((df.iloc[train_idx], df.iloc[val_idx]))
        
        return splits
    
    # Stratified splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y = df[stratify_col].fillna('unknown')
    
    splits = []
    for train_idx, val_idx in skf.split(df, y):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        splits.append((train_df, val_df))
    
    return splits