import os
import json
import shutil
import inspect
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from torch.optim import AdamW
from transformers import (
    AlbertTokenizerFast, AutoConfig, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification, set_seed,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

# Set seed for reproducibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(42)

# --------- CONFIGURATION ---------
# Update these paths as needed for your local environment
DATA_DIR = "./data"  # Expects jsonl files here
MODEL_DIR = "tanuj/sanskrit-bert" # Placeholder, update with base model
OUTPUT_DIR = "./bert_sanskrit_pos"
# ---------------------------------

def setup_tokenizer(model_name_or_path):
    """Load or set up the tokenizer."""
    # If you have a custom spiece.model, load it here. 
    # Otherwise, load from the base model.
    try:
        return AlbertTokenizerFast.from_pretrained(model_name_or_path)
    except:
        print(f"Could not load tokenizer directly from {model_name_or_path}. Please check path.")
        return None

def align_labels_with_tokens(labels, word_ids, label2id):
    """Enhanced label alignment with better subword handling"""
    out, prev = [], None
    for wid in word_ids:
        if wid is None:  # Special tokens
            out.append(-100)
        elif wid != prev:  # First subword of a word
            out.append(label2id[labels[wid]])
        else:  # Continuation subwords - use same label (not -100)
            out.append(label2id[labels[wid]])
        prev = wid
    return out

def get_preprocess_fn(tokenizer, label2id, max_length=384):
    def preprocess_fn(batch):
        tok = tokenizer(
            batch["tokens"], 
            is_split_into_words=True,
            truncation=True, 
            max_length=max_length,
            padding=False
        )
        
        new_labels = []
        for i in range(len(batch["tokens"])):
            wids = tok.word_ids(batch_index=i)
            # Ensure labels are available in the batch
            if "labels" in batch:
                aligned_labels = align_labels_with_tokens(batch["labels"][i], wids, label2id)
                new_labels.append(aligned_labels)
        
        tok["labels"] = new_labels
        return tok
    return preprocess_fn

# Focal Loss for Class Imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class EnhancedModelCompat(nn.Module):
    def __init__(self, base, use_focal_loss=True):
        super().__init__()
        self.base = base
        self.config = getattr(base, "config", None)
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=1, gamma=2)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, 
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        
        cleaned_kwargs = {k: v for k, v in kwargs.items() if k in [
            'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 
            'head_mask', 'inputs_embeds', 'labels', 'output_attentions', 
            'output_hidden_states', 'return_dict'
        ]}
        
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **cleaned_kwargs
        )
        
        if self.training and labels is not None and self.use_focal_loss:
            logits = outputs.logits
            loss = self.focal_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs.loss = loss
            
        return outputs
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)
    
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        # Helper to load the base model and wrap it
        base = AutoModelForTokenClassification.from_pretrained(path, **kwargs)
        return cls(base)

class SmartDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            labels = batch["labels"]
            if isinstance(labels, torch.Tensor) and labels.dtype != torch.long:
                batch["labels"] = labels.long()
        return batch

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=-1)
    
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for pred_i, label_i in zip(prediction, label):
            if label_i != -100:
                true_predictions.append(pred_i)
                true_labels.append(label_i)
    
    accuracy = accuracy_score(true_labels, true_predictions)
    f1_macro = f1_score(true_labels, true_predictions, average="macro")
    f1_weighted = f1_score(true_labels, true_predictions, average="weighted")
    
    return {
        "accuracy": accuracy,
        "f1": f1_macro,
        "f1_weighted": f1_weighted,
    }

def main():
    # 1. Setup Data
    # Assuming jsonl files are present
    data_files = {}
    if os.path.exists(os.path.join(DATA_DIR, "pos_simple_train.jsonl")):
        data_files["train"] = os.path.join(DATA_DIR, "pos_simple_train.jsonl")
    if os.path.exists(os.path.join(DATA_DIR, "pos_simple_valid.jsonl")):
        data_files["validation"] = os.path.join(DATA_DIR, "pos_simple_valid.jsonl")
    
    if not data_files:
        print(f"No data files found in {DATA_DIR}. Please place pos_simple_train.jsonl there.")
        return

    raw = load_dataset("json", data_files=data_files)
    
    # 2. Labels
    labs = set()
    for ex in raw["train"]:
        labs.update(ex["labels"])
    label_list = sorted(labs)
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}
    
    print(f"Labels: {list(label2id.keys())}")
    
    # 3. Tokenizer
    tokenizer = setup_tokenizer(MODEL_DIR)
    if not tokenizer: return

    # 4. Preprocess
    preprocess = get_preprocess_fn(tokenizer, label2id)
    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    
    # 5. Model
    config = AutoConfig.from_pretrained(
        MODEL_DIR, 
        num_labels=len(label2id), 
        id2label=id2label, 
        label2id=label2id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        classifier_dropout=0.1
    )
    
    base_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR, config=config, ignore_mismatched_sizes=True
    )
    
    # Resize embeddings if needed
    if len(tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))
    
    model = EnhancedModelCompat(base_model, use_focal_loss=True)
    
    # 6. Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        num_train_epochs=30,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
    )
    
    # 7. Trainer
    collator = SmartDataCollator(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    
    # 8. Train
    trainer.train()
    
    # 9. Save final
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
