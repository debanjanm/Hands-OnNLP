import random
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeaturePipeline:
    """
    Transforms raw text into BERT-based feature embeddings.
    """
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 device: str = None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def transform(self,
                  texts: list[str],
                  batch_size: int = 32,
                  max_length: int = 256) -> np.ndarray:
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                outputs = self.model(**enc)
                # Use [CLS] token representation
                cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeds.append(cls_embeds)
        return np.vstack(all_embeds)


class TrainingPipeline:
    """
    Runs training of an ML classifier on BERT features.
    """
    def __init__(self,
                 feature_pipeline: FeaturePipeline,
                 classifier=None,
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.feature_pipeline = feature_pipeline
        self.classifier = classifier or LogisticRegression(max_iter=1000)
        self.test_size = test_size
        self.random_state = random_state

    def run(self,
            texts: list[str],
            labels: list[int]):
        # Extract features
        X = self.feature_pipeline.transform(texts)
        y = np.array(labels)
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        # Fit classifier
        self.classifier.fit(X_train, y_train)
        # Evaluate
        train_acc = self.classifier.score(X_train, y_train)
        val_acc = self.classifier.score(X_val, y_val)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        return self.classifier


class InferencePipeline:
    """
    Performs inference using a trained classifier on new text data.
    """
    def __init__(self,
                 feature_pipeline: FeaturePipeline,
                 classifier):
        self.feature_pipeline = feature_pipeline
        self.classifier = classifier

    def predict(self, texts: list[str]) -> list[int]:
        X = self.feature_pipeline.transform(texts)
        return self.classifier.predict(X).tolist()

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        X = self.feature_pipeline.transform(texts)
        return self.classifier.predict_proba(X).tolist()


if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed(42)

    # Load only 10k examples from IMDB
    dataset = load_dataset('imdb')
    train_ds = dataset['train'].shuffle(seed=42).select(range(100))

    texts = train_ds['text']
    labels = train_ds['label']

    # Initialize pipelines
    feature_pipe = FeaturePipeline()
    trainer = TrainingPipeline(feature_pipe)
    clf = trainer.run(texts, labels)

    # Saving the trained classifier for later use
    # with open('classifier.pkl', 'wb') as f:
    #     pickle.dump(clf, f)

    # Example inference
    infer = InferencePipeline(feature_pipe, clf)
    samples = ["This movie was fantastic!", "I did not like the film."]
    print("Predictions:", infer.predict(samples))
    print("Probabilities:", infer.predict_proba(samples))
