# MULTIMEME: A Multimodal Meme Identifier

## Project Overview

**MULTIMEME** is a cutting-edge multimodal classification framework tailored for meme understanding. It blends Computer Vision (CV) and Natural Language Processing (NLP) to analyze and classify memes based on embedded humor, sentiment, and context. By combining visual features from images with extracted text through OCR, MULTIMEME achieves high contextual accuracy in real-world meme classification tasks.

---

## Key Features

* **Multimodal Fusion**: Combines vision and language embeddings for comprehensive meme interpretation.
* **OCR-Driven Text Extraction**: Accurately retrieves text from meme images, including stylized and noisy fonts.
* **Transformer-based NLP**: Leverages advanced models like **BERT**, **GPT**, and **LLAMA** for semantic understanding of meme text.
* **Dual Visual Pathways**:

  * **Vision Transformer (ViT)**: For global context and spatial attention in meme images.
  * **CNN (e.g., ResNet, EfficientNet)**: For fine-grained feature extraction including patterns, textures, and object-level semantics.
* **Flexible Architecture**: Choose between CNN or transformer-based image encoders depending on task complexity or hardware constraints.
* **Multi-class Classification**: Supports labeling memes by humor type, sentiment, intent, or context.

---

## Project Structure

```
MULTIMEME/
├── dataset/
│   ├── memes_images/
│   └── labels.json
├── models/
│   ├── image_model.py
│   ├── text_model.py
│   └── fusion_model.py
├── ocr/
│   └── ocr.py
├── results/
│   └── metrics_results.json
├── requirements.txt
├── train.py
├── evaluate.py
└── README.md
```

---

## Installation

### Prerequisites

* Python 3.8+
* PyTorch
* `transformers` (by HuggingFace)
* `pytesseract` + Tesseract OCR installed
* `Pillow` (PIL)

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Multimodal Classifier

```bash
python train.py \
  --image_model google/vit-base-patch16-224 \
  --text_model bert-base-uncased \
  --epochs 10
```

> Optionally switch `--image_model` to `resnet50` or any compatible CNN-based backbone.

### Evaluate on Validation/Test Set

```bash
python evaluate.py --model_checkpoint path/to/checkpoint
```

---

## Model Architecture

### Textual Analysis

* **Tokenizer + Transformer Encoder** (BERT/GPT/LLAMA)
* Outputs: `text_embedding` (context vector)

### Visual Analysis

* **Option 1: Vision Transformer (ViT)** – for global attention
* **Option 2: CNN (e.g., ResNet, EfficientNet)** – for localized patterns
* Outputs: `image_embedding`

### Fusion Layer

* Concatenates `text_embedding` and `image_embedding`
* Passes through fully connected layers for classification

### Final Output

* Multi-class classification logits (e.g., humor category, sentiment label)

---

## Evaluation Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

These metrics ensure balanced evaluation across classes and linguistic/visual variance.
