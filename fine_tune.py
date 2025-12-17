from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate

# 1️⃣ Load datasets
dataset = load_dataset("csv", data_files={
    "train": "training_data.csv",
    "validation": "validation_data.csv",
    "test": "testing_data.csv"
})

# 2️⃣ Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 3️⃣ Tokenize function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

# 4️⃣ Map 'intent' -> numeric 'labels'
label_list = dataset["train"].unique("intent")
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

def encode_labels(batch):
    batch["labels"] = [label2id[i] for i in batch["intent"]]
    return batch

dataset = dataset.map(encode_labels, batched=True)

# 5️⃣ Set format for PyTorch
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 6️⃣ Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "/home/ubuntu/nlu/bert-nlu/model_distilbert/nlu-script/distilbert_cockpit_intents",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 7️⃣ Metrics
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 8️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# 9️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#  🔟 Start training
trainer.train()

# 1️⃣1️⃣ Evaluate on test set
results = trainer.evaluate(dataset["test"])
print("Test results:", results)

# 1️⃣2️⃣ Save final model
trainer.save_model("./distilbert_cockpit_intents_2nd")
tokenizer.save_pretrained("./distilbert_cockpit_intents_2nd")
