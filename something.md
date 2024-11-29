# Half Adder Neural Network

### 1) Simple Neural Network for a Half Adder

A **half adder** adds two binary digits and produces a **sum** and a **carry**. 

#### Neural Network Design:

- **Inputs**: Two nodes for the binary digits (A, B).
- **Hidden Layer**: One or more nodes that process the inputs.
- **Outputs**: Two nodes representing the **sum** and **carry**.

#### Code:

```python
# Import necessary libraries
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Inputs: A and B (Binary values)
A = np.array([0, 0, 1, 1])
B = np.array([0, 1, 0, 1])

# Expected output: Sum and Carry
# Sum = A XOR B, Carry = A AND B
Sum = np.array([0, 1, 1, 0])  # Expected Sum output
Carry = np.array([0, 0, 0, 1])  # Expected Carry output

# Initializing weights and biases
input_layer_size = 2  # Two inputs: A and B
hidden_layer_size = 1  # One hidden node
output_layer_size = 2  # Two outputs: Sum and Carry

# Random weights and biases initialization
weights_input_hidden = np.random.uniform(size=(input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(size=(hidden_layer_size, output_layer_size))
bias_hidden = np.random.uniform(size=(1, hidden_layer_size))
bias_output = np.random.uniform(size=(1, output_layer_size))

# Learning rate
learning_rate = 0.1

# Training the neural network
for epoch in range(10000):
    # Forward pass: Input to hidden layer
    hidden_input = np.dot(np.column_stack((A, B)), weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    # Forward pass: Hidden to output layer
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Backpropagation: Calculate error
    output_error = final_output - np.column_stack((Sum, Carry))
    d_output = output_error * sigmoid_derivative(final_output)

    # Backpropagation: Calculate hidden layer error
    hidden_error = d_output.dot(weights_hidden_output.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights and biases using gradient descent
    weights_hidden_output -= hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden -= np.column_stack((A, B)).T.dot(d_hidden) * learning_rate
    bias_output -= np.sum(d_output, axis=0) * learning_rate
    bias_hidden -= np.sum(d_hidden, axis=0) * learning_rate

# Display final results
print("Predicted Sum and Carry for each input:")
print(final_output)
