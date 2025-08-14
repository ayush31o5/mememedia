
# model/multi_model.py
#
# Multi-task model:
# - Image encoder (ResNet18)
# - OCR text encoder (TransformerEncoder)
# - Classification heads: brands (multi-label), context (multi-label), technical (multi-label),
#   sentiment (single), humor (single), sarcasm (single)
# - NEW: Human Perception as FREE-TEXT generation using a TransformerDecoder.
#
# Notes:
# - The decoder attends to a memory built by concatenating image features (as a sequence)
#   and encoded OCR tokens.
# - Greedy generation is implemented in `generate_human_perception`.
#
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
try:
    from torchvision.models import ResNet18_Weights
    _RESNET18_WEIGHTS = ResNet18_Weights.DEFAULT
except Exception:
    _RESNET18_WEIGHTS = None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]


class MultiMemeNet(nn.Module):
    def __init__(
        self,
        tokenizer,
        transformer_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        pretrained_cnn: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        vocab_size = tokenizer.vocab_size

        # ------------------ Image branch ------------------
        if pretrained_cnn and _RESNET18_WEIGHTS is not None:
            backbone = models.resnet18(weights=_RESNET18_WEIGHTS)
        else:
            backbone = models.resnet18(weights=None)
        # use up to last conv block
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # [B, C, H/32, W/32]
        feat_dim = backbone.fc.in_features
        self.img_proj = nn.Conv2d(feat_dim, transformer_dim, kernel_size=1)  # [B, D, H', W']

        # ------------------ OCR text encoder ------------------
        self.text_embed = nn.Embedding(vocab_size, transformer_dim, padding_idx=self.pad_id)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=transformer_dim*4,
            dropout=dropout,
            batch_first=True,
        )
        self.text_pos = PositionalEncoding(transformer_dim)
        self.text_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ------------------ Fusion for classification ------------------
        fuse_dim = transformer_dim * 2
        self.brand_head      = nn.Linear(fuse_dim, len(tokenizer.brand_vocab))
        self.context_head    = nn.Linear(fuse_dim, len(tokenizer.product_context_vocab))
        self.technical_head  = nn.Linear(fuse_dim, len(tokenizer.technical_concepts_vocab))

        self.sentiment_head  = nn.Linear(fuse_dim, len(tokenizer.sentiment_vocab))
        self.humor_head      = nn.Linear(fuse_dim, len(tokenizer.humor_mechanism_vocab))
        self.sarcasm_head    = nn.Linear(fuse_dim, len(tokenizer.sarcasm_level_vocab))

        # ------------------ Human Perception Decoder ------------------
        dec_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=transformer_dim*4,
            dropout=dropout,
            batch_first=True,
        )
        self.hp_pos     = PositionalEncoding(transformer_dim)
        self.hp_decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.hp_out     = nn.Linear(transformer_dim, vocab_size)  # logits over vocab

    def _build_memory(self, image: torch.Tensor, ocr_ids: torch.Tensor):
        """
        Returns:
          mem_seq: [B, N_img + L, D]
          mem_key_padding_mask: [B, N_img + L] with True for PAD positions (mask out)
        """
        # -- image to sequence
        x = self.cnn(image)                              # [B, C, h, w]
        x = self.img_proj(x)                             # [B, D, h, w]
        x = x.flatten(2).transpose(1, 2)                 # [B, N_img, D]

        # -- text to sequence
        t = self.text_embed(ocr_ids)                     # [B, L, D]
        t = self.text_pos(t)                             # pos enc
        t = self.text_encoder(t)                         # [B, L, D]

        # For classification heads we also want pooled features
        x_pooled = x.mean(dim=1)                         # [B, D]
        t_pooled = t.mean(dim=1)                         # [B, D]

        mem_seq = torch.cat([x, t], dim=1)               # [B, N_img+L, D]
        # Create key padding mask: image tokens are always valid (False);
        # text tokens are pad where ocr_ids == pad_id (True)
        B, Nimg, _ = x.size()
        text_pad = (ocr_ids == self.pad_id)              # [B, L]
        img_pad = torch.zeros((B, Nimg), dtype=torch.bool, device=ocr_ids.device)
        mem_kpm = torch.cat([img_pad, text_pad], dim=1)  # [B, N_img+L]
        return mem_seq, mem_kpm, x_pooled, t_pooled

    def _generate_square_subsequent_mask(self, sz: int, device) -> torch.Tensor:
        # Mask out subsequent positions (causal mask)
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(
        self,
        image: torch.Tensor,
        ocr_ids: torch.Tensor,
        hp_inp: Optional[torch.Tensor] = None,   # [B, T] with BOS ... PAD
    ):
        """
        Returns a dict with classification logits and, if hp_inp is provided,
        sequence logits for human perception generation: [B, T, V]
        """
        mem_seq, mem_kpm, x_pooled, t_pooled = self._build_memory(image, ocr_ids)

        # Classification (use pooled fusion)
        f = torch.cat([x_pooled, t_pooled], dim=1)       # [B, 2D]

        outputs = {
            'brands':    self.brand_head(f),
            'context':   self.context_head(f),
            'technical': self.technical_head(f),
            'sentiment': self.sentiment_head(f),
            'humor':     self.humor_head(f),
            'sarcasm':   self.sarcasm_head(f),
        }

        # Human perception generation (teacher-forced if hp_inp is given)
        if hp_inp is not None:
            # Embed + pos-encode the target input sequence
            tgt = self.text_embed(hp_inp)                # [B, T, D]
            tgt = self.hp_pos(tgt)
            T = tgt.size(1)
            causal_mask = self._generate_square_subsequent_mask(T, tgt.device)  # [T, T]

            dec = self.hp_decoder(
                tgt=tgt,
                memory=mem_seq,
                tgt_mask=causal_mask,
                memory_key_padding_mask=mem_kpm,
                tgt_key_padding_mask=(hp_inp == self.pad_id),
            )                                            # [B, T, D]
            logits = self.hp_out(dec)                    # [B, T, V]
            outputs['human_perception'] = logits

        return outputs

    @torch.no_grad()
    def generate_human_perception(
        self,
        image: torch.Tensor,
        ocr_ids: torch.Tensor,
        max_len: int = 50,
    ) -> torch.Tensor:
        """
        Greedy decoding. Returns [B, T] of token ids (without BOS, stopped at EOS).
        """
        self.eval()
        mem_seq, mem_kpm, _, _ = self._build_memory(image, ocr_ids)

        B = image.size(0)
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=image.device)  # [B, 1]

        finished = torch.zeros(B, dtype=torch.bool, device=image.device)

        for _ in range(max_len):
            tgt = self.text_embed(ys)
            tgt = self.hp_pos(tgt)
            T = tgt.size(1)
            causal_mask = self._generate_square_subsequent_mask(T, ys.device)

            dec = self.hp_decoder(
                tgt=tgt,
                memory=mem_seq,
                tgt_mask=causal_mask,
                memory_key_padding_mask=mem_kpm,
                tgt_key_padding_mask=(ys == self.pad_id),
            )
            logits = self.hp_out(dec)                    # [B, T, V]
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [B]

            # Append
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            # Stop if everyone hit EOS
            finished |= (next_token == self.eos_id)
            if torch.all(finished):
                break

        # Remove BOS and trim after EOS per sequence
        out = []
        for b in range(B):
            seq = ys[b, 1:].tolist()  # drop BOS
            if self.eos_id in seq:
                eos_pos = seq.index(self.eos_id)
                seq = seq[:eos_pos]
            out.append(torch.tensor(seq, device=ys.device, dtype=torch.long))
        # Pad to same length for tensor output
        maxT = max(len(s) for s in out) if out else 1
        padded = torch.full((B, maxT), self.pad_id, dtype=torch.long, device=ys.device)
        for b, s in enumerate(out):
            padded[b, :len(s)] = s
        return padded
