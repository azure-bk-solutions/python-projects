import torch
import torch.nn as nn


# Simulated Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


# Belief State Transformer Model (Parallelized FE & BE)
class BeliefStateTransformer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.forward_encoder = TransformerBlock(hidden_dim)  # FE
        self.backward_encoder = TransformerBlock(hidden_dim)  # BE

    def forward(self, input_seq):
        seq_len, batch_size, hidden_dim = input_seq.shape

        # Split sequence into two parts: left for FE, right for BE
        midpoint = seq_len // 2
        fe_input = input_seq[:midpoint]  # First half (FE processes this)
        be_input = input_seq[midpoint:].flip(
            0
        )  # Second half reversed (BE processes this)

        # Run FE and BE **simultaneously**
        fe_output = self.forward_encoder(fe_input)  # Left-to-right pass
        be_output = self.backward_encoder(be_input)  # Right-to-left pass

        # Flip BE output back to match sequence order
        be_output = be_output.flip(0)

        # Combine FE and BE outputs
        output = torch.cat([fe_output, be_output], dim=0)

        return output


# Simulated Input Sequence (Batch of 3, 10 tokens, 512-dim embeddings)
input_seq = torch.randn(10, 3, 512)

# Initialize BST model
bst_model = BeliefStateTransformer(hidden_dim=512)

# Forward pass
output = bst_model(input_seq)

print("Output Shape:", output.shape)  # Should match input shape
