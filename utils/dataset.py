# utils/dataset.py

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MemeDataset(Dataset):
    def __init__(self, img_dir, labels_excel, tokenizer, max_len=50):
        df = pd.read_excel(labels_excel)
        self.img_dir   = img_dir
        self.tokenizer = tokenizer
        self.max_len   = max_len

        # filter out missing images
        exts = ['.png', '.jpg', '.jpeg']
        valid_rows = []
        missing    = []
        for _, row in df.iterrows():
            base = str(row['Meme ID']).strip()
            if any(os.path.exists(os.path.join(img_dir, base+e)) for e in exts):
                valid_rows.append(row)
            else:
                missing.append(base)
        if missing:
            print(f"Warning: Skipping {len(missing)} missing images: {missing}")
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406], [.229,.224,.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base = str(row['Meme ID']).strip()
        # load image
        for ext in ['.png','.jpg','.jpeg']:
            path = os.path.join(self.img_dir, base+ext)
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                break

        image = self.transform(img)

        # OCR text from Excel
        ocr_text = str(row['OCR Text'])
        ocr_ids  = torch.LongTensor(self.tokenizer.encode(ocr_text, max_len=self.max_len))

        # prepare labels
        labels = {}
        labels['identified_brands']   = torch.FloatTensor(
            self.tokenizer.encode_multilabel(str(row['Identified Brands']), self.tokenizer.brand_vocab)
        )
        labels['product_context']     = torch.FloatTensor(
            self.tokenizer.encode_multilabel(str(row['Product Context']), self.tokenizer.product_context_vocab)
        )
        labels['technical_concepts']  = torch.FloatTensor(
            self.tokenizer.encode_multilabel(str(row['Technical Concepts']), self.tokenizer.technical_concepts_vocab)
        )
        labels['overall_sentiment']   = torch.tensor(
            self.tokenizer.class_to_id(str(row['Overall Sentiment']), self.tokenizer.sentiment_vocab),
            dtype=torch.long
        )
        labels['humor_mechanism']     = torch.tensor(
            self.tokenizer.class_to_id(str(row['Humor Mechanism']), self.tokenizer.humor_mechanism_vocab),
            dtype=torch.long
        )
        labels['sarcasm_level']       = torch.tensor(
            self.tokenizer.class_to_id(str(row['Sarcasm Level']), self.tokenizer.sarcasm_level_vocab),
            dtype=torch.long
        )
        labels['human_perception']    = torch.tensor(
            self.tokenizer.class_to_id(str(row['Human Perception']), self.tokenizer.human_perception_vocab),
            dtype=torch.long
        )

        return {
            'image':   image,
            'ocr_ids': ocr_ids,
            'labels':  labels
        }
