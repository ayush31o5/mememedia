# utils/text_tokenizer.py

import pandas as pd
from collections import Counter

class TextTokenizer:
    def __init__(self,
                 vocab,
                 brand_vocab,
                 product_context_vocab,
                 technical_concepts_vocab,
                 sentiment_vocab,
                 humor_mechanism_vocab,
                 sarcasm_level_vocab,
                 human_perception_vocab):
        self.vocab                     = vocab
        self.brand_vocab               = brand_vocab
        self.product_context_vocab     = product_context_vocab
        self.technical_concepts_vocab  = technical_concepts_vocab
        self.sentiment_vocab           = sentiment_vocab
        self.humor_mechanism_vocab     = humor_mechanism_vocab
        self.sarcasm_level_vocab       = sarcasm_level_vocab
        self.human_perception_vocab    = human_perception_vocab

        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.vocab_size = len(vocab)

    @classmethod
    def from_excel(cls, excel_path):
        df = pd.read_excel(excel_path)

        # 1) build shared OCR vocab
        all_text = " ".join(df['OCR Text'].astype(str).tolist()).split()
        vocab = [w for w,_ in Counter(all_text).most_common(2000)]

        # 2) build each output vocab
        brand_vocab              = sorted({b.strip() for row in df['Identified Brands']      for b in str(row).split(',')})
        product_context_vocab    = sorted({c.strip() for row in df['Product Context']         for c in str(row).split(',')})
        technical_concepts_vocab = sorted({t.strip() for row in df['Technical Concepts']     for t in str(row).split(',')})
        sentiment_vocab          = sorted(df['Overall Sentiment'].dropna().unique())
        humor_mechanism_vocab    = sorted(df['Humor Mechanism'].dropna().unique())
        sarcasm_level_vocab      = sorted(df['Sarcasm Level'].dropna().unique())
        human_perception_vocab   = sorted(df['Human Perception'].dropna().unique())

        return cls(vocab,
                   brand_vocab,
                   product_context_vocab,
                   technical_concepts_vocab,
                   sentiment_vocab,
                   humor_mechanism_vocab,
                   sarcasm_level_vocab,
                   human_perception_vocab)

    def encode(self, text, max_len):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 0) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    def encode_multilabel(self, text, label_vocab):
        labels = text.split(',')
        vec = [0] * len(label_vocab)
        for l in labels:
            l = l.strip()
            if l in label_vocab:
                vec[label_vocab.index(l)] = 1
        return vec

    def class_to_id(self, cls, vocab):
        return vocab.index(cls) if cls in vocab else 0
