# Extending BERT Tokenizer with Domain-Specific Vocabulary

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO_NAME/blob/main/Extending_tokenizer.ipynb)

## Description

This project demonstrates how to extend a pre-trained BERT tokenizer with domain-specific vocabulary (medical terms in this example). By adding custom tokens to the tokenizer and fine-tuning the model, you can improve performance on specialized text domains without training a model from scratch. **[Refer to this blog for more information - CloudAIApp.dev](https://cloudaiapp.dev/extending-pretrained-transformers-with-domain-specific-vocabulary-a-hugging-face-walkthrough/)**.

## Features

- Load a pre-trained BERT model and tokenizer
- Extend the tokenizer with custom domain-specific vocabulary
- Resize the model's embedding layer to accommodate new tokens
- Fine-tune the model on a custom corpus
- Save and load the extended tokenizer and model
- Test the extended tokenizer on new text

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- GPU (recommended for faster training)

## Installation

```bash
# Clone the repository
git clone https://github.com/azure-bk-solutions/python-projects.git
cd REPO_NAME

# Install dependencies
pip install torch transformers
```

## Usage

### 1. Load Pre-trained Model and Tokenizer

```python
from transformers import BertTokenizer, BertForMaskedLM

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
```

### 2. Extend Tokenizer with New Tokens

```python
# Add domain-specific vocabulary
new_tokens = ["angiocardiography", "echocardiogram", "neurofibromatosis"]
num_added = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added} tokens.")

# Resize model embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))
```

### 3. Prepare Custom Dataset

```python
# Example dataset with domain-specific terms
custom_corpus = [
    "The echocardiogram revealed a potential defect.",
    "Angiocardiography is often used in diagnostic imaging.",
    "Neurofibromatosis can lead to tumor formation."
]

# Tokenize dataset
tokenized_data = tokenizer(custom_corpus, return_tensors='pt', padding=True, truncation=True)

# Create a simple dataset class
from torch.utils.data import Dataset

class SimpleTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["input_ids"])

dataset = SimpleTextDataset(tokenized_data)
```

### 4. Fine-tune the Model

```python
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Setup for Masked Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert-custom",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    report_to="none"  # disables wandb
)

# Initialize trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()
```

### 5. Save and Load the Extended Model and Tokenizer

```python
# Save
model.save_pretrained("bert-custom")
tokenizer.save_pretrained("bert-custom")

# Load
tokenizer = BertTokenizer.from_pretrained("bert-custom")
```

### 6. Verify Added Tokens

```python
# Check added vocabulary
added = tokenizer.get_added_vocab()
print(added)
# {'angiocardiography': 30522, 'echocardiogram': 30523, 'neurofibromatosis': 30524}
```

### 7. Test the Extended Tokenizer

```python
# Test tokenization with custom tokens
test_text = "The patient's echocardiogram showed no abnormalities after the angiocardiography procedure."
tokens = tokenizer.tokenize(test_text)
print(tokens)
# ['the', 'patient', "'", 's', 'echocardiogram', 'showed', 'no', 'abnormal', '##ities', 'after', 'the', 'angiocardiography', 'procedure', '.']
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
