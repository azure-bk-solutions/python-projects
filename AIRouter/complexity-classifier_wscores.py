import onnxruntime
from transformers import BertTokenizer
import numpy as np

# Load tokenizer (must match training)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# Load ONNX model (choose FP32 or INT8)
session = onnxruntime.InferenceSession("bert_complexity_int8.onnx")

# Define label map
id2label = {0: "simple", 1: "medium", 2: "complex"}

# Test real-world queries
test_queries = [
    "Cancel my subscription.",
    "Email the report to Steve and update the dashboard before 5 PM.",
    "Play soft jazz music.",
    "What is the capital of Brazil?",
    "Fetch last month's metrics and generate a PowerPoint summary.",
    "I need a document in Word showing vendor performance this quarter. Use the data in SharePoint.",
    "Book a meeting with security after you create a report based on Steve's incident email.",
    "Create a PowerPoint summary of product roadmap from Amit's notes and email it to the product team.",
]


def predict_complexity(query):
    # Tokenize input
    inputs = tokenizer(
        query, return_tensors="np", padding="max_length", truncation=True, max_length=64
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    # Run inference
    outputs = session.run(None, ort_inputs)
    logits = outputs[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return id2label[pred_class], round(confidence, 4)


# Run test
for query in test_queries:
    label, confidence = predict_complexity(query)
    print(
        f"🔹 Query: {query}\n→ Predicted: {label.upper()} (Confidence: {confidence})\n"
    )
