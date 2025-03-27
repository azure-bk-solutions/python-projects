import time
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer
import torch.nn.functional as F

# Load models
session_fp32 = ort.InferenceSession("bert_complexity_fp32.onnx")
session_int8 = ort.InferenceSession("bert_complexity_int8.onnx")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# Sample queries
queries = [
    "What is the weather today?",
    "Email the summary report to the CFO and update the dashboard.",
    "Play relaxing jazz music.",
    "Cancel my subscription effective immediately.",
    "Summarize Steve's email and generate a Word document report.",
]

labels = ["simple", "medium", "complex"]

print(
    "{:<60} {:<10} {:<8} {:<10} {:<8}".format("Query", "FP32", "Time", "INT8", "Time")
)
print("=" * 100)

for q in queries:
    inputs = tokenizer(
        q, return_tensors="np", padding="max_length", truncation=True, max_length=64
    )

    # FP32 inference
    start = time.time()
    logits_fp32 = session_fp32.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )[0]
    fp32_time = (time.time() - start) * 1000
    fp32_pred = np.argmax(logits_fp32)

    # INT8 inference
    start = time.time()
    logits_int8 = session_int8.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )[0]
    int8_time = (time.time() - start) * 1000
    int8_pred = np.argmax(logits_int8)

    print(
        "{:<60} {:<10} {:<8.2f} {:<10} {:<8.2f}".format(
            q[:57] + ("..." if len(q) > 57 else ""),
            labels[fp32_pred],
            fp32_time,
            labels[int8_pred],
            int8_time,
        )
    )
