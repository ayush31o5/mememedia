import torch
import torch.nn as nn
import torchvision.models as models

class MultiMemeNet(nn.Module):
    def __init__(self, tokenizer, transformer_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        # image branch
        backbone = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        feat_dim = backbone.fc.in_features
        self.img_proj = nn.Conv2d(feat_dim, transformer_dim, 1)
        # text branch
        self.text_embed = nn.Embedding(tokenizer.vocab_size, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim,
                                                   nhead=n_heads, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # fusion heads
        fuse_dim = transformer_dim*2
        self.brand_head = nn.Linear(fuse_dim, len(tokenizer.brand_vocab))
        self.context_head = nn.Linear(fuse_dim, len(tokenizer.product_context_vocab))
        self.tech_head = nn.Linear(fuse_dim, len(tokenizer.technical_concepts_vocab))
        self.sentiment_head = nn.Linear(fuse_dim, len(tokenizer.sentiment_vocab))
        self.humor_head = nn.Linear(fuse_dim, len(tokenizer.humor_mechanism_vocab))
        self.sarcasm_head = nn.Linear(fuse_dim, len(tokenizer.sarcasm_level_vocab))

    def forward(self, image, ocr_ids):
        # image
        x = self.cnn(image)
        x = self.img_proj(x).flatten(2).transpose(1,2)  # [B, N_img, D]
        x = x.mean(1)  # [B, D]
        # text
        t = self.text_embed(ocr_ids)
        t = self.text_encoder(t)
        t = t.mean(1)
        # fuse
        f = torch.cat([x, t], dim=1)
        return {
            'brands': torch.sigmoid(self.brand_head(f)),
            'context': torch.sigmoid(self.context_head(f)),
            'technical': torch.sigmoid(self.tech_head(f)),
            'sentiment': self.sentiment_head(f),
            'humor': self.humor_head(f),
            'sarcasm': self.sarcasm_head(f),
        }