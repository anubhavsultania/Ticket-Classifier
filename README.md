# 🎫 Multi-Task Ticket Classifier + Entity Extractor

A lightweight Gradio web app to classify customer support tickets into:
- Issue Type
- Urgency Level

It also extracts key entities such as:
- Product Names
- Order IDs
- Dates
- Complaint Keywords

> Built using **DistilBERT**, PyTorch, Scikit-learn, and Transformers.


---

## 📸 Screenshots

<img width="1731" height="875" alt="image" src="https://github.com/user-attachments/assets/fa5f0bc5-9aa7-46e7-930c-ef869422430a" />
<img width="1734" height="860" alt="image" src="https://github.com/user-attachments/assets/12e3cad4-a66e-4eb6-a13f-75878a7a9328" />

---

## 🧠 Model Overview

This project uses a **Multi-Task Learning** architecture:

- Shared DistilBERT encoder
- Two classification heads:
  - Issue Type classifier
  - Urgency Level classifier

NER is handled via a simple rule-based extractor for relevant entities.

---

## 🛠️ Setup Locally

```bash
git clone https://github.com/YOUR_USERNAME/Ticket-Classifier.git
cd Ticket-Classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
python app/app.py
```
