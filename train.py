import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from model.image_model import ImageTransformerModel
from model.text_model import TextBERTModel
from model.fusion_model import FusionClassifier

class MemeDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = row['text']
        label = int(row['label'])
        return image, text, label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_model = ImageTransformerModel().to(device)
    text_model = TextBERTModel().to(device)
    fusion_model = FusionClassifier(image_model.out_dim, text_model.out_dim).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = MemeDataset('data/labels.csv', 'data/images', transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(image_model.parameters()) +
        list(text_model.parameters()) +
        list(fusion_model.parameters()), lr=1e-4)

    for epoch in range(2):
        for images, texts, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            image_feats = image_model(images)
            text_feats = text_model(texts)
            text_feats = text_feats.to(device)

            outputs = fusion_model(image_feats, text_feats)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
