import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# 1️⃣ Load fine-tuned model and tokenizer
model_path = "./distilbert_cockpit_intents_2nd"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# 2️⃣ Put model in eval mode
model.eval()

# 3️⃣ Inference function
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
        pred_intent = model.config.id2label[pred_id]
    return pred_intent

# 4️⃣ Example usage
examples = [
    "lower the temperature to 20",
    "roll down the left window",
    "turn on the radio",
    "increase fan speed"
]

for cmd in examples:
    intent = predict_intent(cmd)
    print(f"Command: '{cmd}' → Predicted intent: '{intent}'")
