import torch
import yaml
from PIL import Image
from torchvision import transforms
from utils.text_tokenizer import TextTokenizer
from model.multi_model import MultiMemeNet

cfg = yaml.safe_load(open('configs/default.yaml'))
device = torch.device(cfg['device'])
# load tokenizer and model
tokenizer = TextTokenizer.from_excel(cfg['data']['labels_excel'])
model = MultiMemeNet(tokenizer, transformer_dim=cfg['model']['transformer_dim'])
model.load_state_dict(torch.load(cfg['predict']['ckpt_path'], map_location=device))
model.to(device).eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    # OCR
    from utils.ocr import ocr_multi_lingual
    raw_text = ocr_multi_lingual(image_path)
    ids = torch.LongTensor(tokenizer.encode(raw_text, cfg['model']['max_ocr_len'])).unsqueeze(0).to(device)
    preds = model(x, ids)
    return preds

if __name__=='__main__':
    import sys
    out = predict(sys.argv[1])
    print(out)
