import os
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataset import MemeDataset
from utils.text_tokenizer import TextTokenizer
from model.multi_model import MultiMemeNet

def main(cfg):
    device = torch.device(cfg['device'])
    tokenizer = TextTokenizer.from_excel(cfg['data']['labels_excel'])
    dataset = MemeDataset(cfg['data']['raw_dir'], cfg['data']['labels_excel'],
                          tokenizer, max_len=cfg['model']['max_ocr_len'])
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    model = MultiMemeNet(tokenizer, transformer_dim=cfg['model']['transformer_dim']).to(device)
    crit_bce = nn.BCEWithLogitsLoss()
    crit_ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        model.train()
        for batch in loader:
            imgs = batch['image'].to(device)
            ocr = batch['ocr_ids'].to(device)
            labels = batch['labels']
            preds = model(imgs, ocr)
            loss = 0
            loss += crit_bce(preds['brands'], labels['identified_brands'].to(device))
            loss += crit_bce(preds['context'], labels['product_context'].to(device))
            loss += crit_bce(preds['technical'], labels['technical_concepts'].to(device))
            loss += crit_ce(preds['sentiment'], labels['overall_sentiment'].to(device))
            loss += crit_ce(preds['humor'], labels['humor_mechanism'].to(device))
            loss += crit_ce(preds['sarcasm'], labels['sarcasm_level'].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), os.path.join(cfg['output_dir'], f"epoch{epoch+1}.pth"))

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/default.yaml'))
    os.makedirs(cfg['output_dir'], exist_ok=True)
    main(cfg)