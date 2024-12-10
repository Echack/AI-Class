import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.text import Annotation

# Defined the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Defined activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
np.random.seed(42)
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1
weights_input_hidden = np.random.uniform(-1, 1, (input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_layer_neurons, output_neurons))
bias_hidden = np.random.uniform(-1, 1, (1, hidden_layer_neurons))
bias_output = np.random.uniform(-1, 1, (1, output_neurons))

# My training parameters
learning_rate = 0.1
epochs = 10000

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - output_layer_output
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print loss occasionally
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch}, Loss: {loss}")

# Test the neural network
hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
output_layer_output = sigmoid(output_layer_input)
print("\nFinal Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {output_layer_output[i][0]:.4f}, Actual Output: {y[i][0]}")

# Weights and biases
print("\nWeights and Biases:")
print("Weights Input to Hidden:\n", weights_input_hidden)
print("Bias Hidden:\n", bias_hidden)
print("Weights Hidden to Output:\n", weights_hidden_output)
print("Bias Output:\n", bias_output)

# Visualize the neural network structure
def plot_nn():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Input layer
    input_nodes = [(2, 8), (2, 6)]
    for node in input_nodes:
        ax.add_artist(plt.Circle(node, 0.3, color='blue', fill=True))
        ax.text(node[0] - 0.7, node[1], 'Input', fontsize=10, color='black')

    # Hidden layer
    hidden_nodes = [(5, 8), (5, 6)]
    for node in hidden_nodes:
        ax.add_artist(plt.Circle(node, 0.3, color='green', fill=True))
        ax.text(node[0] - 1.2, node[1], 'Hidden', fontsize=10, color='black')

    # Output layer
    output_nodes = [(8, 7)]
    for node in output_nodes:
        ax.add_artist(plt.Circle(node, 0.3, color='red', fill=True))
        ax.text(node[0] - 1.2, node[1], 'Output', fontsize=10, color='black')

    # Connections
    for i in input_nodes:
        for h in hidden_nodes:
            ax.add_artist(FancyArrow(i[0] + 0.3, i[1], h[0] - i[0] - 0.6, h[1] - i[1], 
                                     width=0.01, color='black'))
    for h in hidden_nodes:
        for o in output_nodes:
            ax.add_artist(FancyArrow(h[0] + 0.3, h[1], o[0] - h[0] - 0.6, o[1] - h[1], 
                                     width=0.01, color='black'))

    plt.show()

plot_nn()
