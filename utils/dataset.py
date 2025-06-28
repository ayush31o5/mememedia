import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.ocr import ocr_multi_lingual
from utils.text_tokenizer import TextTokenizer

class MemeDataset(Dataset):
    """
    PyTorch Dataset for Multimodal Meme Annotation Data.
    Reads an Excel file with columns:
      - 'Meme ID': unique identifier (without extension)
      - 'Folder Link'
      - 'OCR Text': extracted text
      - 'Identified Brands', 'Product Context', 'Technical Concepts',
        'Overall Sentiment', 'Sentiment Description', 'Humor Mechanism',
        'Humor Analysis', 'Sarcasm Present', 'Sarcasm Level',
        'Sarcasm Description', 'Audience Perception'

    Returns a dict with:
      - image: Tensor [3,H,W]
      - ocr_ids: LongTensor [max_len]
      - labels: dict containing both encoded targets and raw descriptors, including original OCR text
    """
    def __init__(
        self,
        img_dir: str,
        labels_excel: str,
        tokenizer: TextTokenizer,
        max_len: int = 50,
        transform=None
    ):
        # Load annotations
        self.df = pd.read_excel(labels_excel)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Image transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_id = str(row['Meme ID'])
        # attempt common extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(self.img_dir, base_id + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"Image for {base_id} not found in {self.img_dir}")

        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # OCR text: use stored or fallback
        raw_text = str(row.get('OCR Text', '')).strip()
        if not raw_text or raw_text.lower() in ['nan', '']:
            raw_text = ocr_multi_lingual(img_path) or ''

        # Tokenize OCR text for model input
        ocr_ids = self.tokenizer.encode(raw_text, max_len=self.max_len)
        ocr_ids = torch.LongTensor(ocr_ids)  # [max_len]

        # Prepare labels dict
        labels = {}
        # Include original OCR text as a target label
        labels['ocr_text'] = raw_text

        # Multi-label: Identified Brands
        labels['identified_brands'] = self.tokenizer.encode_multilabel(
            str(row.get('Identified Brands', '')), self.tokenizer.brand_vocab
        )  # FloatTensor [n_brands]

        # Multi-label: Product Context
        labels['product_context'] = self.tokenizer.encode_multilabel(
            str(row.get('Product Context', '')), self.tokenizer.product_context_vocab
        )  # FloatTensor [n_contexts]

        # Multi-label: Technical Concepts
        labels['technical_concepts'] = self.tokenizer.encode_multilabel(
            str(row.get('Technical Concepts', '')), self.tokenizer.technical_concepts_vocab
        )  # FloatTensor [n_tech]

        # Single-class: Overall Sentiment
        labels['overall_sentiment'] = torch.tensor(
            self.tokenizer.class_to_id(
                str(row.get('Overall Sentiment', '')), self.tokenizer.sentiment_vocab
            ), dtype=torch.long
        )

        # Text: Sentiment Description
        labels['sentiment_description'] = str(row.get('Sentiment Description', ''))

        # Single-class: Humor Mechanism
        labels['humor_mechanism'] = torch.tensor(
            self.tokenizer.class_to_id(
                str(row.get('Humor Mechanism', '')), self.tokenizer.humor_mechanism_vocab
            ), dtype=torch.long
        )

        # Text: Humor Analysis
        labels['humor_analysis'] = str(row.get('Humor Analysis', ''))

        # Binary: Sarcasm Present
        sp = str(row.get('Sarcasm Present', '')).strip().lower()
        labels['sarcasm_present'] = torch.tensor(
            1 if sp == 'yes' else 0, dtype=torch.long
        )

        # Single-class: Sarcasm Level
        labels['sarcasm_level'] = torch.tensor(
            self.tokenizer.class_to_id(
                str(row.get('Sarcasm Level', '')), self.tokenizer.sarcasm_level_vocab
            ), dtype=torch.long
        )

        # Text: Sarcasm Description
        labels['sarcasm_description'] = str(row.get('Sarcasm Description', ''))

        # Text: Audience Perception
        labels['audience_perception'] = str(row.get('Audience Perception', ''))

        return {
            'image': image,       # Tensor [3,224,224]
            'ocr_ids': ocr_ids,   # LongTensor [max_len]
            'labels': labels      # dict with encoded and raw labels
        }