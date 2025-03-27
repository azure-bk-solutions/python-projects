from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import BertTokenizer, pipeline
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableMap

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import BertTokenizer, pipeline

# Path to your ONNX model directory (must contain config.json and tokenizer files too)
model_path = "bert_complexity_router"  # NOT .onnx file directly

# Load ONNX-compatible model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = ORTModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


# Step 2: Define label mapping (must match your fine-tuned labels)
LABELS = ["simple", "medium", "complex"]


# Step 3: Simulate LLM actions (to be replaced with actual LangChain ChatLLM chains later)
def route_to_llm(intent_complexity: str, query: str):
    routing_map = {
        "simple": f"[Routed to Mistral Large] → {query}",
        "medium": f"[Routed to LLaMA 3 70B] → {query}",
        "complex": f"[Routed to GPT-4o] → {query}",
    }
    return routing_map.get(intent_complexity, "[Unknown LLM]")


# Step 4: LangChain-compatible classification and routing pipeline
def classify_and_route(input: dict) -> str:
    query = input["query"]
    pred = classifier(query)[0]
    label_index = int(pred["label"].split("_")[-1]) if "label" in pred else 0
    complexity = LABELS[label_index]
    return route_to_llm(complexity, query)


# Step 5: Build LangChain Runnable
router_chain = RunnableMap(
    {
        "query": lambda x: x,
    }
) | RunnableLambda(classify_and_route)

# Step 6: Test Cases
test_queries = [
    "What's the weather today?",
    "Generate a financial summary report from SharePoint and SQL.",
    "Create a PowerPoint for Q4 including revenue trends, customer churn analysis, and email it to CFO.",
    "Send a PowerPoint to the product team based on Veda's product notes.",
    "CFO needs the Q3 summary. Get it posted to SharePoint first.",
    "Let legal know when the contracts are uploaded to SharePoint. They're in Dropbox.",
]

for i, q in enumerate(test_queries, 1):
    result = router_chain.invoke(q)
    print(f"Query {i}:\n{result}\n")
