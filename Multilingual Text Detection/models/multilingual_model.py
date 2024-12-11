import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class MultilingualSentimentModel:
    def __init__(self, model_name="xlm-roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Positive, Negative, Neutral

    def predict(self, text, lang):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs