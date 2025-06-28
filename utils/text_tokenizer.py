import pandas as pd
from collections import Counter
import torch

class TextTokenizer:
    def __init__(self, vocab, brand_vocab, product_context_vocab,
                 technical_concepts_vocab, sentiment_vocab,
                 humor_mechanism_vocab, sarcasm_level_vocab):
        self.vocab = vocab
        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.brand_vocab = brand_vocab
        self.product_context_vocab = product_context_vocab
        self.technical_concepts_vocab = technical_concepts_vocab
        self.sentiment_vocab = sentiment_vocab
        self.humor_mechanism_vocab = humor_mechanism_vocab
        self.sarcasm_level_vocab = sarcasm_level_vocab

    @classmethod
    def from_excel(cls, excel_path):
        df = pd.read_excel(excel_path)
        # build text vocab
        all_text = " ".join(df['OCR Text'].astype(str).tolist()).split()
        vocab = [w for w,c in Counter(all_text).most_common(5000)]
        brand_vocab = sorted({b.strip() for row in df['Identified Brands'] for b in str(row).split(',')})
        product_context_vocab = sorted({c.strip() for row in df['Product Context'] for c in str(row).split(',')})
        technical_concepts_vocab = sorted({t.strip() for row in df['Technical Concepts'] for t in str(row).split(',')})
        sentiment_vocab = sorted(df['Overall Sentiment'].dropna().unique())
        humor_mechanism_vocab = sorted(df['Humor Mechanism'].dropna().unique())
        sarcasm_level_vocab = sorted(df['Sarcasm Level'].dropna().unique())
        return cls(vocab, brand_vocab, product_context_vocab,
                   technical_concepts_vocab, sentiment_vocab,
                   humor_mechanism_vocab, sarcasm_level_vocab)

    def encode(self, text, max_len):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t,0) for t in tokens]
        ids += [0]*(max_len-len(ids))
        return ids

    def encode_multilabel(self, text, vocab):
        vec = torch.zeros(len(vocab))
        for item in str(text).split(','):
            i = item.strip()
            if i in vocab:
                vec[vocab.index(i)] = 1
        return vec

    def class_to_id(self, item, vocab):
        return vocab.index(item) if item in vocab else 0
