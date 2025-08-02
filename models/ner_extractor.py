# models/ner_extractor.py

import re
import json

# Load product list and complaint keywords from JSON
with open("resources/product_list.json") as f:
    product_list = json.load(f)

with open("resources/complaint_keywords.json") as f:
    complaint_keywords = json.load(f)

# Date regex: handles dd/mm/yyyy, dd-mm-yyyy, dd/mm/yy etc.
date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b'

def extract_entities(text: str) -> dict:
    text_lower = text.lower()

    # Product match (exact match from list)
    matched_product = None
    for product in product_list:
        if product.lower() in text_lower:
            matched_product = product
            break

    # Complaint keywords match
    found_keywords = [kw for kw in complaint_keywords if kw.lower() in text_lower]

    # Date extraction
    found_dates = re.findall(date_pattern, text)

    # Order ID extraction (e.g. "#12345")
    found_order_id = re.findall(r"#\d{4,6}", text)

    return {
        "product": matched_product,
        "order_id": found_order_id[0] if found_order_id else None,
        "dates": found_dates,
        "complaint_keywords": found_keywords
    }

# Test
if __name__ == "__main__":
    sample = "Refund my SmartWatch V2 ordered on 12/07/2024. Order #34521 is incorrect."
    print(extract_entities(sample))
