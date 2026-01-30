# Sanskrit POS Tagger (BERT-based)

This repository contains code to train and use a Sanskrit Part-of-Speech (POS) tagger. This model is fine-tuned on top of a SanskritBERT base model using a comprehensive Sanskrit POS dataset.

## Model Details
- **Base Model**: SanskritBERT (ALBERT/BERT based)
- **Task**: Token Classification (POS Tagging)
- **Labels**: Noun, Verb, Adjective, Adverb, etc. (UD POS Tags)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

You can use the provided `inference.py` script to tag Sanskrit text.

```bash
python inference.py
```

**Example Code:**

```python
from transformers import pipeline

nlp = pipeline("token-classification", model="your-username/sanskrit-pos-bert")
print(nlp("रामः वनम् गच्छति"))
```

## Training

The `train.py` script contains the training logic.

```bash
python train.py
```

## Folder Structure
- `train.py`: Training script.
- `inference.py`: Inference script.
- `requirements.txt`: Python dependencies.

## License
[MIT](LICENSE)
