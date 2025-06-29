# Part 1.2 - MLP Implementation and Training with Multiple Activations
# File 1 mlp.py
# MLP Implementation for XOR Problem with Multiple Activations
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def activate(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
        elif self.activation == 'relu':
            return self.relu(x)

    def activate_derivative(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'tanh':
            return self.tanh_derivative(x)
        elif self.activation == 'relu':
            return self.relu_derivative(x)

    def forward(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activate(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.activate(self.final_input)
        return self.output

    def backward(self, inputs, targets, learning_rate):
        output_error = self.output - targets
        output_delta = output_error * self.activate_derivative(self.final_input)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activate_derivative(self.hidden_input)

        # Update weights and biases
        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden -= learning_rate * np.dot(inputs.T, hidden_delta)
        self.bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
