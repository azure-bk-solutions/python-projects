import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Simulate a toy neural network using NumPy instead of PyTorch

matplotlib.use('TkAgg',force=True)

print("Switched to:",matplotlib.get_backend())

# Input values (100 values from -2 to 2)
X = np.linspace(-2, 2, 100).reshape(-1, 1)

print("Input shape:", X.shape)  # Should be (100, 1)
print("Input values:", X[:5])  # Print first 5 input values

# Random seed for reproducibility
np.random.seed(0)

#If you're running multiple experiments and want some randomness but still control it for reproducibility, you can set the random seed before generating random numbers. This way, every time you run the code with the same seed, you'll get the same random numbers.
# This is useful for debugging or comparing results across different runs.

# for i in range(3):
#     np.random.seed(i)  # change seed each time
#     print(np.random.randn(3))

# Define network weights and biases (fixed for fairness across activations)
W1 = np.random.randn(1, 10)  # Input layer to hidden layer (1 input, 10 neurons)
b1 = np.random.randn(10)     # Bias for hidden layer
W2 = np.random.randn(10, 1)  # Hidden to output layer
b2 = np.random.randn(1)      # Bias for output

# print ("Weights and biases initialized:")
# print("W1 shape:", W1.shape)  # Should be (1, 10)   
# print("b1 shape:", b1.shape)  # Should be (10,)
# print("W2 shape:", W2.shape)  # Should be (10, 1)   
# print("b2 shape:", b2.shape)  # Should be (1,)
# print("W1 values:", W1[:5])  # Print first 5 weights
# print("b1 values:", b1[:5])  # Print first 5 biases 
# print("W2 values:", W2[:5])  # Print first 5 weights
# print("b2 value:", b2)        # Print bias  

# Define activation functions
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)

activations = {
    "ReLU": relu,
    "Sigmoid": sigmoid,
    "Tanh": tanh,
    "LeakyReLU": leaky_relu
}

# Plot results
plt.figure(figsize=(12, 8))

for i, (name, fn) in enumerate(activations.items()):
    # Forward pass
    dotproduct = np.dot(X, W1) + b1  # Linear transformation to hidden layer
    # print(f"Linear transformation to hidden layer ({name}):", dotproduct[:5])  # Print first 5 values
    # # Apply activation function to hidden layer
    # print(dotproduct.shape)  # Should be (100, 10)
    hidden = fn(dotproduct)     # Apply activation in hidden layer
    # print(f"After activation ({name}):", hidden[:5])  # Print first 5 values
    # print(hidden.shape)  # Should be (100, 10)
    # Linear transformation to output layer     
    output = np.dot(hidden, W2) + b2    # Linear output layer (no activation)

    plt.subplot(2, 2, i + 1)
    plt.plot(X, output)
    plt.title(f"Output with {name}")
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Toy Neural Network (NumPy) - Effect of Activation Functions", fontsize=16, y=1.02)
plt.show()
print("Plot displayed successfully.")
input("Press Enter to exit...")
# Note: The script uses NumPy for matrix operations and matplotlib for plotting.    