from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "sileod/deberta-v3-base-tasksource-bias"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ["left", "center", "right"]

def check_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    pred = probs.argmax().item()
    return {
        "bias": labels[pred],
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))}
    }

# EXAMPLE
if __name__ == "__main__":
    text = input("Enter news text: ")
    result = check_bias(text)
    print(result)
