from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

# Load pre-trained DistilBERT (2 labels by default, but we just care about logits)
tokenizer = DistilBertTokenizerFast.from_pretrained("/home/ubuntu/nlu/bert-nlu/model_distilbert/nlu-script/distilbert_cockpit_intents")
model = DistilBertForSequenceClassification.from_pretrained("/home/ubuntu/nlu/bert-nlu/model_distilbert/nlu-script/distilbert_cockpit_intents")

# Example cockpit command
text = "lower the driver window"

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)

print("Logits:", logits)
print("Probabilities:", probs)
