# utils/text_tokenizer.py
#
# Tokenizer for:
# - OCR tokenization (fixed length, padded)
# - Human Perception caption generation (<pad>, <bos>, <eos>, <unk>)
#
from collections import Counter
import pandas as pd
import re

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3

class TextTokenizer:
    def __init__(self,
                 vocab,
                 brand_vocab,
                 product_context_vocab,
                 technical_concepts_vocab,
                 sentiment_vocab,
                 humor_mechanism_vocab,
                 sarcasm_level_vocab):
        # Core vocab
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(vocab)

        # Classification label vocabs
        self.brand_vocab               = brand_vocab
        self.product_context_vocab     = product_context_vocab
        self.technical_concepts_vocab  = technical_concepts_vocab
        self.sentiment_vocab           = sentiment_vocab
        self.humor_mechanism_vocab     = humor_mechanism_vocab
        self.sarcasm_level_vocab       = sarcasm_level_vocab

        # Special token IDs
        self.pad_id = PAD
        self.bos_id = BOS
        self.eos_id = EOS
        self.unk_id = UNK

    @staticmethod
    def basic_tokenize(text):
        """Simple lowercase tokenizer that preserves punctuation."""
        return re.findall(r"\w+|[^\w\s]", str(text).lower())

    @classmethod
    def from_excel(cls, excel_path, max_vocab=4000):
        df = pd.read_excel(excel_path)

        # Detect Human Perception column
        hp_col = None
        for cand in ["Human Perception Text", "Human Perception", "Human Perception ?"]:
            if cand in df.columns:
                hp_col = cand
                break
        if hp_col is None:
            raise ValueError("No Human Perception column found in Excel.")

        # Build vocab from OCR Text + Human Perception free text
        text_blobs = []
        if 'OCR Text' in df.columns:
            text_blobs.extend(df['OCR Text'].astype(str).tolist())
        text_blobs.extend(df[hp_col].astype(str).tolist())

        tokens = []
        for s in text_blobs:
            tokens.extend(cls.basic_tokenize(s))

        most_common = [w for w, _ in Counter(tokens).most_common(max_vocab - len(SPECIAL_TOKENS))]
        vocab = SPECIAL_TOKENS + most_common

        # Helper to build multi-label vocab
        def build_multilabel_vocab(series):
            out = set()
            for row in series.fillna(''):
                for item in str(row).split(','):
                    item = item.strip()
                    if item:
                        out.add(item)
            return sorted(out)

        brand_vocab              = build_multilabel_vocab(df['Identified Brands']) if 'Identified Brands' in df.columns else []
        product_context_vocab    = build_multilabel_vocab(df['Product Context']) if 'Product Context' in df.columns else []
        technical_concepts_vocab = build_multilabel_vocab(df['Technical Concepts']) if 'Technical Concepts' in df.columns else []
        sentiment_vocab          = sorted(df['Overall Sentiment'].dropna().unique().tolist()) if 'Overall Sentiment' in df.columns else []
        humor_mechanism_vocab    = sorted(df['Humor Mechanism'].dropna().unique().tolist()) if 'Humor Mechanism' in df.columns else []
        sarcasm_level_vocab      = sorted(df['Sarcasm Level'].dropna().unique().tolist()) if 'Sarcasm Level' in df.columns else []

        return cls(vocab,
                   brand_vocab,
                   product_context_vocab,
                   technical_concepts_vocab,
                   sentiment_vocab,
                   humor_mechanism_vocab,
                   sarcasm_level_vocab)

    # ---------- OCR Encoding ----------
    def encode(self, text, max_len):
        toks = self.basic_tokenize(text)
        ids = [self.word2idx.get(t, self.unk_id) for t in toks][:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    # ---------- Human Perception Caption ----------
    def encode_for_generation(self, text, max_len):
        toks = self.basic_tokenize(text)
        ids = [self.word2idx.get(t, self.unk_id) for t in toks]
        ids = ids[:max_len - 1]  # space for BOS/EOS

        inp = [self.bos_id] + ids
        tgt = ids + [self.eos_id]

        if len(inp) < max_len:
            inp += [self.pad_id] * (max_len - len(inp))
        if len(tgt) < max_len:
            tgt += [self.pad_id] * (max_len - len(tgt))

        return inp, tgt

    def decode(self, ids):
        words = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            words.append(self.idx2word.get(int(i), "<unk>"))
        return " ".join(words).strip()

    # ---------- Multi-label ----------
    def encode_multilabel(self, text, label_vocab):
        vec = [0] * len(label_vocab)
        for l in str(text).split(','):
            l = l.strip()
            if l in label_vocab:
                vec[label_vocab.index(l)] = 1
        return vec

    def class_to_id(self, cls, vocab):
        return vocab.index(cls) if cls in vocab else 0
