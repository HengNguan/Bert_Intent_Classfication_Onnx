import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

text = "roll down the left window"
enc = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=16)

np.save("input_ids.npy", enc["input_ids"])
np.save("attention_mask.npy", enc["attention_mask"])
