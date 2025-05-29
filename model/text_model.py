from transformers import BertTokenizer, BertModel
import torch.nn as nn

class TextBERTModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]
