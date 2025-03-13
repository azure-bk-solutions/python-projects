import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModel
import time

# Set device to CPU explicitly
device = torch.device("cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sentence to process
sentence = "Mary had a little lamb its fleece was white as snow"
tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
input_ids = tokens["input_ids"].to(device)

# Convert tokens to readable words
token_list = tokenizer.convert_ids_to_tokens(input_ids[0])
print("\n🔹 Tokenized Sentence:", token_list)

# Transformer Block (FE and BE)
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

# Belief State Transformer Model
class BeliefStateTransformer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.forward_encoder = TransformerBlock(hidden_dim)
        self.backward_encoder = TransformerBlock(hidden_dim)

    def forward(self, embeddings):
        seq_len, batch_size, hidden_dim = embeddings.shape
        midpoint = seq_len // 2

        fe_input = embeddings[:midpoint]
        be_input = embeddings[midpoint:].flip(0)

        fe_output = self.forward_encoder(fe_input)
        be_output = self.backward_encoder(be_input).flip(0)

        output = torch.cat([fe_output, be_output], dim=0)
        return output

# Convert tokens into embeddings
embedding_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
with torch.no_grad():
    embeddings = embedding_model.embeddings.word_embeddings(input_ids)

print("\n🔹 Embedding Shape:", embeddings.shape)

# Run BST Model
bst_model = BeliefStateTransformer(hidden_dim=embeddings.shape[-1]).to(device)
output = bst_model(embeddings)

print("\n✅ Final BST Output Shape:", output.shape)


# ** Visualization of BE in action **
def visualize_be_reconstruction(token_list):
    future_tokens = token_list[len(token_list) // 2:]  # Future context for BE
    reconstructed = ["[MASK]"] * len(future_tokens)  # Initially masked
    fig, ax = plt.subplots()
    plt.ion()  # Enable interactive mode

    for i in range(len(future_tokens)):
        reconstructed[i] = future_tokens[-(i + 1)]  # Reverse reconstruct step-by-step

        ax.clear()
        ax.set_title("🔄 Backward Encoder (BE) Step-by-Step Reconstruction")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Token")
        ax.set_xticks(np.arange(len(future_tokens)))
        ax.set_xticklabels(future_tokens, rotation=45, ha="right")

        colors = ["red" if tok == "[MASK]" else "green" for tok in reconstructed]
        ax.bar(np.arange(len(future_tokens)), [1] * len(future_tokens), color=colors)
        for j, tok in enumerate(reconstructed):
            ax.text(j, 0.5, tok, ha="center", va="center", fontsize=12, color="black")

        plt.pause(0.5)  # Pause for animation effect

    plt.ioff()  # Turn off interactive mode
    plt.show()


# Call visualization function
visualize_be_reconstruction(token_list)
