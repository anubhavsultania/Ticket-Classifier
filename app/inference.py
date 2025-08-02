# app/inference.py

import torch
import pickle
from transformers import DistilBertTokenizerFast
from models.classifier import MultiTaskTicketClassifier
from models.ner_extractor import extract_entities

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load label encoders
with open("models/issue_encoder.pkl", "rb") as f:
    issue_encoder = pickle.load(f)

with open("models/urgency_encoder.pkl", "rb") as f:
    urgency_encoder = pickle.load(f)

# Load model
model = MultiTaskTicketClassifier(
    issue_type_classes=len(issue_encoder.classes_),
    urgency_level_classes=len(urgency_encoder.classes_)
)
model.load_state_dict(torch.load("models/multi_task_model.pt", map_location=torch.device("cpu")))
model.eval()


def predict_ticket(text: str) -> dict:
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors="pt")

    with torch.no_grad():
        issue_logits, urgency_logits = model(encoding["input_ids"], encoding["attention_mask"])
        issue_pred = torch.argmax(issue_logits, dim=1).item()
        urgency_pred = torch.argmax(urgency_logits, dim=1).item()

    predicted_issue = issue_encoder.inverse_transform([issue_pred])[0]
    predicted_urgency = urgency_encoder.inverse_transform([urgency_pred])[0]
    entities = extract_entities(text)

    return {
        "Predicted Issue Type": predicted_issue,
        "Predicted Urgency Level": predicted_urgency,
        "Extracted Entities": entities
    }
