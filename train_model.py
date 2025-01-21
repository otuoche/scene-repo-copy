import torch
torch.backends.mps.is_available = lambda: False  # Disable MPS explicitly

device = torch.device("cpu")

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json  # <-- Needed for saving label_dict

# Define the dataset class
class SceneDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val).to(device) for key, val in encodings.items()}
        self.labels = torch.tensor(labels).to(device)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define label dictionary
label_dict = {
    0: "Gifting",
    1: "Call to Action",
    2: "Product Review: Key Feature",
    3: "First Reaction",
    4: "Emotional Dependency",
    5: "Unboxing",
    6: "Use Case Scenarios",
    7: "Family Reactions",
    8: "Product Review: Features",
    9: "Product Results",
    10: "Customer Testimonial Snippets",
    11: "Problem-Solution Narratives",
    12: "Product-in-Use Montage",
    13: "Product Review: Benefits",
    14: "Unboxing: First Reaction",
    15: "Broad Appeal",
    16: "Unboxing: Anticipation",
    17: "Benefit of Benefits",
    18: "Introduction",
    19: "Moment of Surprise",
    20: "Interactive Q&A Moments",
    21: "Personal Insights",
    22: "Teaser Introduction",
    23: "Hero's Journey Before",
    24: "Product Routine / Use Case"
}

# Load and split the dataset (adjust the path to your CSV file)
df = pd.read_csv('path_to_your_dataset.csv')  # Replace with your dataset path
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# Tokenizer and encoding function
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return SceneDataset(encodings, labels)

# Prepare datasets
train_dataset = encode_texts(df_train['sentence'].tolist(), df_train['scene_type'].tolist())
val_dataset = encode_texts(df_val['sentence'].tolist(), df_val['scene_type'].tolist())
test_dataset = encode_texts(df_test['sentence'].tolist(), df_test['scene_type'].tolist())

# Load the model and move it to the CPU
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_dict))
model.to(device)

# Set up training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=17,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate(test_dataset)
print("Evaluation Results:", evaluation_results)

# -----------------------------
# SAVE MODEL, TOKENIZER, LABELS
# -----------------------------
# Create directories and save the trained model & tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")

# Save label dictionary to JSON
with open("label_dict.json", "w") as f:
    json.dump(label_dict, f)

# Prediction function with labeling
def predict_scene_type(texts, model, tokenizer, label_dict):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    encodings = {key: val.to(device) for key, val in encodings.items()}  # Move inputs to CPU
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(**encodings)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(probs, dim=1).tolist()
    # If label_dict's keys are integers, this is fine. If you store them as strings, adjust indexing.
    predicted_labels = [label_dict[pred] for pred in predictions]
    return predicted_labels

# Example prediction
texts = [
    "wow. this feels so soft. it’s amazing!",
    "if you want to sleep on a cloud and have the support of a customized pillow, you need to get a Marshmalloo."
]
predictions = predict_scene_type(texts, model, tokenizer, label_dict)
print("Predictions:", predictions)
