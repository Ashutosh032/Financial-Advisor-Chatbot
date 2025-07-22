# Demonstration: EMI calculation and FinBERT sentiment
from math import pow

def calculate_emi(p, annual_rate, years):
    r = annual_rate / (12*100)  # monthly rate
    n = years*12
    emi = p * r * pow(1+r, n) / (pow(1+r, n) - 1)
    return emi

emi_example = calculate_emi(500000, 8.0, 15)
print(f"Monthly EMI for ₹5,00,000 at 8% for 15 years: ₹{emi_example:.2f}")

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
model_name = "ProsusAI/finbert"
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
text = "Apple reported record revenue in the last quarter and announced an increase in dividend."
print("News headline:", text)
print("FinBERT sentiment:", classifier(text))