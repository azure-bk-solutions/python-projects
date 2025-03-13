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


# ** Visualization of FE vs. BE in action **
def visualize_fe_be_reconstruction_fixed(token_list):
    midpoint = len(token_list) // 2  # Midpoint for splitting FE and BE
    fe_tokens = token_list[:midpoint]  # FE tokens
    be_tokens = list(
        reversed(token_list[midpoint:])
    )  # BE tokens (reversed for proper order)

    reconstructed_fe = ["[MASK]"] * len(fe_tokens)
    reconstructed_be = ["[MASK]"] * len(be_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.ion()  # Enable interactive mode

    for i in range(len(fe_tokens)):
        # FE moves forward
        reconstructed_fe[i] = fe_tokens[i]
        # BE moves backward **starting from the last token**
        reconstructed_be[i] = be_tokens[i]

        # Plot Forward Encoder Progress
        axes[0].clear()
        axes[0].set_title("🚀 Forward Encoder (FE) Step-by-Step")
        axes[0].set_xlabel("Token Position")
        axes[0].set_ylabel("Token")
        axes[0].set_xticks(np.arange(len(fe_tokens)))
        axes[0].set_xticklabels(fe_tokens, rotation=45, ha="right")
        fe_colors = ["green" if tok != "[MASK]" else "red" for tok in reconstructed_fe]
        axes[0].bar(np.arange(len(fe_tokens)), [1] * len(fe_tokens), color=fe_colors)
        for j, tok in enumerate(reconstructed_fe):
            axes[0].text(
                j, 0.5, tok, ha="center", va="center", fontsize=12, color="black"
            )

        # Plot Backward Encoder Progress
        axes[1].clear()
        axes[1].set_title("🔄 Backward Encoder (BE) Step-by-Step (Fixed)")
        axes[1].set_xlabel("Token Position")
        axes[1].set_ylabel("Token")
        axes[1].set_xticks(np.arange(len(be_tokens)))
        axes[1].set_xticklabels(
            list(reversed(be_tokens)), rotation=45, ha="right"
        )  # Flip labels back
        be_colors = ["green" if tok != "[MASK]" else "red" for tok in reconstructed_be]
        axes[1].bar(np.arange(len(be_tokens)), [1] * len(be_tokens), color=be_colors)
        for j, tok in enumerate(reconstructed_be):
            axes[1].text(
                j, 0.5, tok, ha="center", va="center", fontsize=12, color="black"
            )

        plt.pause(0.5)  # Smooth transition effect

    plt.ioff()  # Turn off interactive mode
    plt.show()


# Call the corrected visualization function
visualize_fe_be_reconstruction_fixed(token_list)
