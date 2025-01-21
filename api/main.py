# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

app = FastAPI()

# 1. Load the model, tokenizer, label_dict
model_path = "./model"        # Where you saved the model after training
tokenizer_path = "./tokenizer"
label_dict_path = "./label_dict.json"

device = torch.device("cpu")  # or "cuda" if you have a GPU available

model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

with open(label_dict_path, "r") as f:
    label_dict = json.load(f)  # e.g. {"0": "Gifting", "1": "Call to Action", ...}
    
# 2. Create request schema using Pydantic
class TextRequest(BaseModel):
    texts: list[str]  # user can pass multiple texts

# 3. Prediction function
def predict_scene_type(texts):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(probs, dim=1).tolist()
    # label_dict keys might be string type if loaded from JSON:
    predicted_labels = [label_dict[str(pred)] for pred in predictions]
    return predicted_labels

# 4. Define the FastAPI route
@app.post("/predict")
async def predict_scenes(request: TextRequest):
    try:
        # Get predictions
        results = predict_scene_type(request.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
