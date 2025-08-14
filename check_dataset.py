from utils.dataset import MemeDataset
from utils.text_tokenizer import TextTokenizer

labels_excel = "data/labels.xlsx"   # adjust if in another folder
img_dir = "data/raw"                # adjust if in another folder

# Load tokenizer
tok = TextTokenizer.from_excel(labels_excel)

# Load dataset
ds = MemeDataset(img_dir, labels_excel, tok, max_len=64, max_hp_len=64)

# Print results
print("Total samples in dataset:", len(ds))
if len(ds) > 0:
    print("First 5 Meme IDs loaded:", list(ds.df["Meme ID"][:5]))
else:
    print("No samples found — check file names and Meme ID column in Excel.")
