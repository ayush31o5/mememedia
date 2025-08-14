import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MemeDataset(Dataset):
    def __init__(self, img_dir, labels_excel, tokenizer, max_len=50, max_hp_len=None):
        df = pd.read_excel(labels_excel)
        self.img_dir   = img_dir
        self.tokenizer = tokenizer
        self.max_len   = int(max_len)
        self.max_hp_len = int(max_hp_len if max_hp_len is not None else max_len)

        # Pick the human perception column
        if 'Human Perception Text' in df.columns:
            self.hp_col = 'Human Perception Text'
        elif 'Human Perception' in df.columns:
            self.hp_col = 'Human Perception'
        elif 'Human Perception ?' in df.columns:
            self.hp_col = 'Human Perception ?'
        else:
            raise ValueError("No human perception column found in the Excel file.")

        exts = ['.png', '.jpg', '.jpeg', '.webp']
        valid_rows, missing = [], []

        for _, row in df.iterrows():
            base = str(row.get('Meme ID', '')).strip()
            if not base:
                continue
            found = False
            base_lower = base.lower()
            for ext in exts:
                if os.path.exists(os.path.join(img_dir, base + ext)):
                    found = True
                    break
                for fname in os.listdir(img_dir):
                    if fname.lower() == base_lower + ext:
                        found = True
                        break
                if found:
                    break
            if found:
                valid_rows.append(row)
            else:
                missing.append(base)

        if missing:
            print(f"⚠ Skipping {len(missing)} missing images: {missing}")

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, base):
        base_lower = base.lower()
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            candidate = os.path.join(self.img_dir, base + ext)
            if os.path.exists(candidate):
                return Image.open(candidate).convert('RGB')
            for fname in os.listdir(self.img_dir):
                if fname.lower() == base_lower + ext:
                    return Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        raise FileNotFoundError(f"No image found for {base} in {self.img_dir}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base = str(row['Meme ID']).strip()

        img = self._load_image(base)
        image = self.transform(img)

        ocr_text = str(row.get('OCR Text', '') or '')
        ocr_ids  = torch.LongTensor(self.tokenizer.encode(ocr_text, max_len=self.max_len))

        labels = {}
        labels['identified_brands']   = torch.FloatTensor(
            self.tokenizer.encode_multilabel(str(row.get('Identified Brands', '') or ''), self.tokenizer.brand_vocab)
        )
        labels['product_context']     = torch.FloatTensor(
            self.tokenizer.encode_multilabel(str(row.get('Product Context', '') or ''), self.tokenizer.product_context_vocab)
        )
        labels['technical_concepts']  = torch.FloatTensor(
            self.tokenizer.encode_multilabel(str(row.get('Technical Concepts', '') or ''), self.tokenizer.technical_concepts_vocab)
        )
        labels['overall_sentiment']   = torch.tensor(
            self.tokenizer.class_to_id(str(row.get('Overall Sentiment', '') or ''), self.tokenizer.sentiment_vocab),
            dtype=torch.long
        )
        labels['humor_mechanism']     = torch.tensor(
            self.tokenizer.class_to_id(str(row.get('Humor Mechanism', '') or ''), self.tokenizer.humor_mechanism_vocab),
            dtype=torch.long
        )
        labels['sarcasm_level']       = torch.tensor(
            self.tokenizer.class_to_id(str(row.get('Sarcasm Level', '') or ''), self.tokenizer.sarcasm_level_vocab),
            dtype=torch.long
        )

        hp_text = str(row.get(self.hp_col, '') or '').strip()
        hp_inp_ids, hp_tgt_ids = self.tokenizer.encode_for_generation(hp_text, max_len=self.max_hp_len)
        hp_inp = torch.LongTensor(hp_inp_ids)
        hp_tgt = torch.LongTensor(hp_tgt_ids)

        return {
            'image':   image,
            'ocr_ids': ocr_ids,
            'hp_inp':  hp_inp,
            'hp_tgt':  hp_tgt,
            'labels':  labels
        }
