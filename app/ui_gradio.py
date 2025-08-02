# app/ui_gradio.py

import gradio as gr
from .inference import predict_ticket


iface = gr.Interface(
    fn=predict_ticket,
    inputs=gr.Textbox(lines=5, placeholder="Paste ticket description here..."),
    outputs="json",
    title="🎫 Multi-Task Ticket Classifier + Entity Extractor",
    description="Predicts issue type and urgency level from customer tickets and extracts key entities (product, order ID, dates, and complaint keywords)."
)

if __name__ == "__main__":
    iface.launch()
