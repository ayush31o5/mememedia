from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn

class ImageTransformerModel(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]
