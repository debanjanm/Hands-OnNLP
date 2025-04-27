import torch
import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 num_labels: int = 2,
                 dropout_prob: float = 0.3):
        super().__init__()
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)  
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)  
        # Final fully-connected layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # BERT outputs: (last_hidden_state, pooled_output, ...)  
        pooled_output = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask)[1]  
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        return logits

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Example usage:
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

class TrainingPipeline:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 lr: float = 2e-5,
                 epochs: int = 3,
                 device: str = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # AdamW optimizer as recommended for Transformers
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss, total_correct = 0, 0
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / (len(self.train_loader.dataset))
        return avg_loss, accuracy

    def eval_epoch(self):
        self.model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / (len(self.val_loader.dataset))
        return avg_loss, accuracy

    def run(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            print(f"Epoch {epoch+1} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

class InferencePipeline:
    def __init__(self, model: nn.Module, tokenizer: BertTokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str], max_length: int = 256, batch_size: int = 16):
        all_logits = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(enc['input_ids'], enc['attention_mask'])
            all_logits.append(logits.cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = nn.functional.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy().tolist()
        return preds, probs.tolist()

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer


def main():
    # 1. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load and sample 10K IMDB reviews
    ds = load_dataset("imdb")                     # Hugging Face IMDB dataset :contentReference[oaicite:0]{index=0}
    train_full = ds["train"].shuffle(seed=42).select(range(100))
    texts, labels = train_full["text"], train_full["label"]

    # 3. Split into 80% train / 20% val
    n_train = int(0.8 * len(labels))
    train_texts, val_texts = texts[:n_train], texts[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]

    # 4. Tokenizer + Datasets
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = IMDBDataset(train_texts, train_labels, tokenizer)    # wraps to PyTorch Dataset :contentReference[oaicite:1]{index=1}
    val_ds   = IMDBDataset(val_texts,   val_labels,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    # 5. Model + Trainer
    model = BERTClassifier().to(device)
    trainer = TrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=2e-5,
        epochs=3,
        device=device
    )

    # 6. Run training
    trainer.run()   # prints per-epoch loss & accuracy :contentReference[oaicite:2]{index=2}

    # 7. Save the fine-tuned model
    # torch.save(model.state_dict(), "bert_imdb_fc.pt")

    # 8. Quick inference test
    infer = InferencePipeline(model, tokenizer, device=device)
    examples = [
        "Absolutely loved it – best movie ever!",
        "It was okay, but I wouldn’t watch it again."
    ]
    preds, probs = infer.predict(examples)
    for txt, p, prob in zip(examples, preds, probs):
        print(f"REVIEW: {txt}\n→ PREDICTED: {p} (prob={prob[p]:.3f})\n")

if __name__ == "__main__":
    main()  # standard entry-point idiom in Python scripts :contentReference[oaicite:3]{index=3}
