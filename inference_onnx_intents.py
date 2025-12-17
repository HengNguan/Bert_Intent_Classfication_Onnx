import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer from your original model (or the onnx_model folder)
tokenizer = AutoTokenizer.from_pretrained("./distilbert_cockpit_intents_2nd")

id2label = {
    0: "lower_window",
    1: "raise_window",
    2: "set_temperature",
    3: "mute_media",
    4: "unmute_media"
}

examples = [
    "lower the temperature to 20",
    "roll down the left window",
    "turn on the radio",
    "increase fan speed"
]

# Load ONNX Runtime session
session = ort.InferenceSession("./onnx_model/model.onnx")

# Example input

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="np")
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })
    logits = outputs[0]
    pred_id = np.argmax(logits, axis=1)[0]
    pred_label = id2label[pred_id]
    #print("Logits:", logits)
    #print("Predicted ID:", pred_id)
    print(f"Command: '{text}' → Predicted intent: '{pred_label}'")

# Loop through examples
for text in examples:
    predict_intent(text)
