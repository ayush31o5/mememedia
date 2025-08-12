import sys
import torch
import yaml
import json
import pandas as pd
from PIL import Image
from torchvision import transforms
from utils.text_tokenizer import TextTokenizer
from model.multi_model import MultiMemeNet
from utils.ocr import ocr_multi_lingual

# Load config
cfg = yaml.safe_load(open('configs/default.yaml'))
device = torch.device(cfg['device'])

# Load tokenizer
tokenizer = TextTokenizer.from_excel(cfg['data']['labels_excel'])

# Load model
model = MultiMemeNet(tokenizer, transformer_dim=cfg['model']['transformer_dim'])
model.load_state_dict(torch.load(cfg['predict']['ckpt_path'], map_location=device))
model.to(device).eval()

# Load label names from Excel
df_labels = pd.read_excel(cfg['data']['labels_excel'])

label_names = {
    'brands': df_labels['Identified Brands'].dropna().unique().tolist(),
    'context': df_labels['Product Context'].dropna().unique().tolist(),
    'technical': df_labels['Technical Concepts'].dropna().unique().tolist(),
    'overall_sentiment': df_labels['Overall Sentiment'].dropna().unique().tolist(),
    'humor_mechanism': df_labels['Humor Mechanism'].dropna().unique().tolist(),
    'sarcasm_level': df_labels['Sarcasm Level'].dropna().unique().tolist(),
    'human_perception': df_labels['Human Perception'].dropna().unique().tolist()
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    # OCR text
    raw_text = ocr_multi_lingual(image_path)
    ids = torch.LongTensor(tokenizer.encode(raw_text, cfg['model']['max_ocr_len'])).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        outputs = model(x, ids)

    result = {}

    # Multi-label heads
    for head, col in [('brands', 'Identified Brands'),
                      ('context', 'Product Context'),
                      ('technical', 'Technical Concepts')]:
        probs = torch.sigmoid(outputs[head]).cpu().numpy()[0]
        preds = [label_names[head][i] for i, p in enumerate(probs) if p >= 0.5]
        result[col] = preds

    # Single-label heads
    single_heads = [
        ('Sentiment Description', 'Overall Sentiment'),
        ('humor', 'Humor Mechanism'),
        ('sarcasm', 'Sarcasm Level')
    ]
    for head, col in single_heads:
        pred_idx = int(torch.argmax(outputs[head], dim=1).cpu())
        result[col] = label_names[head][pred_idx]

    # Human Perception as statement
    hp_idx = int(torch.argmax(outputs['human_perception'], dim=1).cpu())
    result['Human Perception'] = label_names['human_perception'][hp_idx]

    # OCR text
    result['OCR Text'] = raw_text

    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.predict <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    output = predict(image_path)
    print(json.dumps(output, indent=4, ensure_ascii=False))
