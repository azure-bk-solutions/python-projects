import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Set device to CPU explicitly
device = torch.device("cpu")

# Load a pre-trained tokenizer (like BERT tokenizer)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sentence to process
sentence = "Mary had a little lamb its fleece was white as snow"

# Tokenize and Convert to Tensor
tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
input_ids = tokens["input_ids"].to(device)  # Token IDs
seq_len = input_ids.shape[1]  # Get sequence length

# Print tokenized output
print("\n🔹 **Tokenized Sentence:**")
print(tokenizer.convert_ids_to_tokens(input_ids[0]))


# Simulated Transformer Block (FE and BE)
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


# Belief State Transformer Model (Parallel FE & BE)
class BeliefStateTransformer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.forward_encoder = TransformerBlock(hidden_dim)  # FE
        self.backward_encoder = TransformerBlock(hidden_dim)  # BE

    def forward(self, embeddings):
        seq_len, batch_size, hidden_dim = embeddings.shape

        # Split into Forward and Backward segments
        midpoint = seq_len // 2
        fe_input = embeddings[:midpoint]  # FE gets first half
        be_input = embeddings[midpoint:].flip(0)  # BE gets second half reversed

        # Print FE and BE inputs
        print("\n🔹 **FE Input Shape:**", fe_input.shape)
        print("🔹 **BE Input Shape (Reversed):**", be_input.shape)
        #  print("\n🔹 **BE...", be_input)

        # Parallel processing
        fe_output = self.forward_encoder(fe_input)  # Left-to-right
        be_output = self.backward_encoder(be_input)  # Right-to-left

        # Flip BE output back to original order
        be_output = be_output.flip(0)

        # Print intermediate FE and BE outputs
        print("\n🔹 **FE Output Shape:**", fe_output.shape)
        print("🔹 **BE Output Shape (Re-Flipped):**", be_output.shape)

        # Merge outputs
        output = torch.cat([fe_output, be_output], dim=0)

        return output


# Convert tokens into embeddings using a pre-trained model
embedding_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
with torch.no_grad():
    embeddings = embedding_model.embeddings.word_embeddings(input_ids)

# Print Embedding Shape
print("\n🔹 **Embedding Shape:**", embeddings.shape)

# Run BST model
bst_model = BeliefStateTransformer(hidden_dim=embeddings.shape[-1]).to(device)
output = bst_model(embeddings)

# Print Final Output Shape
print("\n🔹 **Final BST Output Shape:**", output.shape)
