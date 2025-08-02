# models/classifier.py

import torch.nn as nn
from transformers import DistilBertModel


class MultiTaskTicketClassifier(nn.Module):
    def __init__(self, issue_type_classes, urgency_level_classes):
        super(MultiTaskTicketClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Task-specific heads
        self.issue_classifier = nn.Linear(self.bert.config.hidden_size, issue_type_classes)
        self.urgency_classifier = nn.Linear(self.bert.config.hidden_size, urgency_level_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token

        issue_logits = self.issue_classifier(cls_output)
        urgency_logits = self.urgency_classifier(cls_output)

        return issue_logits, urgency_logits
