# models/multimodal_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from transformers import CLIPModel, CLIPProcessor, BertModel, BertTokenizer
from peft import LoraConfig, get_peft_model
import timm
from einops import rearrange
import clip
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        B, Lq, D = query.shape
        B, Lk, D = key.shape
        
        q = self.q_proj(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out_proj(out)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class EnhancedMultiModalMemeNet(nn.Module):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize CLIP model with LoRA
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Apply LoRA to CLIP
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.clip_model = get_peft_model(self.clip_model, lora_config)
        
        # Additional CNN backbone for fine-grained features
        self.cnn_backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        cnn_dim = self.cnn_backbone.num_features
        
        # Text encoder with BERT
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_dim = self.bert_model.config.hidden_size
        
        # Projection layers
        clip_dim = 512  # CLIP embedding dimension
        unified_dim = config.get('unified_dim', 768)
        
        self.clip_image_proj = nn.Linear(clip_dim, unified_dim)
        self.clip_text_proj = nn.Linear(clip_dim, unified_dim)
        self.cnn_proj = nn.Linear(cnn_dim, unified_dim)
        self.bert_proj = nn.Linear(bert_dim, unified_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(unified_dim)
        
        # Cross-attention layers
        self.image_text_attention = MultiHeadCrossAttention(unified_dim, n_heads=8)
        self.text_image_attention = MultiHeadCrossAttention(unified_dim, n_heads=8)
        
        # Transformer fusion layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=unified_dim,
            nhead=8,
            dim_feedforward=unified_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classification heads with improved architecture
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(unified_dim * 2)
        
        # Multi-label heads
        self.brand_head = self._make_classification_head(
            unified_dim * 2, len(tokenizer.brand_vocab), multilabel=True
        )
        self.context_head = self._make_classification_head(
            unified_dim * 2, len(tokenizer.product_context_vocab), multilabel=True
        )
        self.technical_head = self._make_classification_head(
            unified_dim * 2, len(tokenizer.technical_concepts_vocab), multilabel=True
        )
        
        # Single-class heads
        self.sentiment_head = self._make_classification_head(
            unified_dim * 2, len(tokenizer.sentiment_vocab)
        )
        self.humor_head = self._make_classification_head(
            unified_dim * 2, len(tokenizer.humor_mechanism_vocab)
        )
        self.sarcasm_head = self._make_classification_head(
            unified_dim * 2, len(tokenizer.sarcasm_level_vocab)
        )
        self.human_perception_head = self._make_classification_head(
            unified_dim * 2, len(tokenizer.human_perception_vocab)
        )
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def _make_classification_head(self, input_dim, output_dim, multilabel=False):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, output_dim)
        )
    
    def forward(self, image, text_ids, attention_mask=None, return_embeddings=False):
        batch_size = image.size(0)
        
        # CLIP encodings
        clip_inputs = self.clip_processor(images=image, return_tensors="pt", do_rescale=False)
        clip_image_features = self.clip_model.get_image_features(**clip_inputs)
        
        # CNN features
        cnn_features = self.cnn_backbone(image)
        
        # Text features from BERT
        bert_outputs = self.bert_model(input_ids=text_ids, attention_mask=attention_mask)
        bert_features = bert_outputs.pooler_output
        
        # CLIP text features (convert text_ids to text first)
        # Note: In practice, you'd need to decode text_ids back to text for CLIP
        # For now, we'll use BERT features as the text representation
        
        # Project to unified dimension
        clip_img_proj = self.clip_image_proj(clip_image_features)
        cnn_proj = self.cnn_proj(cnn_features)
        bert_proj = self.bert_proj(bert_features)
        
        # Combine image features
        image_features = (clip_img_proj + cnn_proj) / 2
        text_features = bert_proj
        
        # Add positional encoding and prepare for attention
        image_seq = image_features.unsqueeze(1)  # [B, 1, D]
        text_seq = text_features.unsqueeze(1)    # [B, 1, D]
        
        # Cross-attention between modalities
        attended_image = self.image_text_attention(image_seq, text_seq, text_seq)
        attended_text = self.text_image_attention(text_seq, image_seq, image_seq)
        
        # Combine attended features
        combined_seq = torch.cat([attended_image, attended_text], dim=1)  # [B, 2, D]
        combined_seq = self.pos_encoding(combined_seq)
        
        # Transformer fusion
        fused_features = self.fusion_transformer(combined_seq)  # [B, 2, D]
        
        # Global pooling and final representation
        final_features = fused_features.mean(dim=1)  # [B, D]
        
        # Concatenate original features with fused features
        final_repr = torch.cat([
            final_features,
            (image_features + text_features) / 2
        ], dim=1)  # [B, 2*D]
        
        final_repr = self.layer_norm(final_repr)
        final_repr = self.dropout(final_repr)
        
        if return_embeddings:
            return final_repr
        
        # Classification heads
        outputs = {
            'brands': self.brand_head(final_repr),
            'context': self.context_head(final_repr),
            'technical': self.technical_head(final_repr),
            'sentiment': self.sentiment_head(final_repr),
            'humor': self.humor_head(final_repr),
            'sarcasm': self.sarcasm_head(final_repr),
            'human_perception': self.human_perception_head(final_repr)
        }
        
        return outputs
    
    def compute_loss(self, outputs, labels, loss_weights=None):
        total_loss = 0.0
        losses = {}
        
        # Multi-label losses (BCE)
        for head in ['brands', 'context', 'technical']:
            target_key = {
                'brands': 'identified_brands',
                'context': 'product_context', 
                'technical': 'technical_concepts'
            }[head]
            
            loss = self.bce_loss(outputs[head], labels[target_key].float())
            losses[f'{head}_loss'] = loss
            weight = loss_weights.get(head, 1.0) if loss_weights else 1.0
            total_loss += weight * loss
        
        # Single-class losses (Focal + Label Smoothing)
        single_heads = {
            'sentiment': 'overall_sentiment',
            'humor': 'humor_mechanism', 
            'sarcasm': 'sarcasm_level',
            'human_perception': 'human_perception'
        }
        
        for head, target_key in single_heads.items():
            # Use focal loss for imbalanced classes
            focal_loss = self.focal_loss(outputs[head], labels[target_key])
            smooth_loss = self.label_smoothing_loss(outputs[head], labels[target_key])
            loss = 0.7 * focal_loss + 0.3 * smooth_loss
            
            losses[f'{head}_loss'] = loss
            weight = loss_weights.get(head, 1.0) if loss_weights else 1.0
            total_loss += weight * loss
        
        losses['total_loss'] = total_loss
        return total_loss, losses

class ModelEMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.module = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}