from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# 1. Load CSVs
dataset = load_dataset("csv", data_files={
    "train": "training_data.csv",
    "validation": "validation_data.csv",
    "test": "testing_data.csv"
})

# 2. Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 3. Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

# 4. Apply tokenization
tokenized_dataset = dataset.map(tokenize, batched=True)

# 5. Set format for PyTorch/TensorFlow
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
