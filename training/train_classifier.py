import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import pickle
import os

from models.classifier import MultiTaskTicketClassifier  # Ensure models/__init__.py exists

# ----- Step 1: Dataset Preparation -----

class TicketDataset(Dataset):
    def __init__(self, dataframe, tokenizer, issue_labels, urgency_labels, max_len=256):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.issue_labels = issue_labels
        self.urgency_labels = urgency_labels
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["ticket_text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "issue_label": torch.tensor(self.issue_labels[idx]),
            "urgency_label": torch.tensor(self.urgency_labels[idx])
        }

# ----- Step 2: Training Function -----

def train():
    df = pd.read_csv("data/sample_data.csv")  # <-- Update if needed

    # Ensure strings
    df["ticket_text"] = df["ticket_text"].astype(str).fillna("")
    df["issue_type"] = df["issue_type"].astype(str).fillna("Unknown")
    df["urgency_level"] = df["urgency_level"].astype(str).fillna("Unknown")

    # Encode labels
    issue_encoder = LabelEncoder()
    urgency_encoder = LabelEncoder()
    issue_labels = issue_encoder.fit_transform(df["issue_type"])
    urgency_labels = urgency_encoder.fit_transform(df["urgency_level"])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = TicketDataset(df, tokenizer, issue_labels, urgency_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MultiTaskTicketClassifier(
        issue_type_classes=len(issue_encoder.classes_),
        urgency_level_classes=len(urgency_encoder.classes_)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            issue_label = batch["issue_label"].to(device)
            urgency_label = batch["urgency_label"].to(device)

            optimizer.zero_grad()
            issue_logits, urgency_logits = model(input_ids, attention_mask)

            loss_issue = criterion(issue_logits, issue_label)
            loss_urgency = criterion(urgency_logits, urgency_label)
            loss = loss_issue + loss_urgency

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/multi_task_model.pt")
    print("✅ Model saved.")

    with open("models/issue_encoder.pkl", "wb") as f:
        pickle.dump(issue_encoder, f)
    with open("models/urgency_encoder.pkl", "wb") as f:
        pickle.dump(urgency_encoder, f)

if __name__ == "__main__":
    train()
