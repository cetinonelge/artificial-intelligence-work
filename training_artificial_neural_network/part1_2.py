# Part 1.2 - MLP Implementation and Training with Multiple Activations
# File 2 part1_2.py
# To run execute: python part1_2.py
import numpy as np
import matplotlib.pyplot as plt
from utils import part1CreateDataset, part1PlotBoundary
from mlp import MLP

# Generate XOR dataset
x_train, y_train, x_val, y_val = part1CreateDataset(train_samples=1000, val_samples=100, std=0.4)

# Neural network parameters
input_size = 2
hidden_size = 8
output_size = 1
learning_rate = 0.001
epochs = 10000

activations = ['sigmoid', 'tanh', 'relu']

for activation in activations:
    print(f'\nTraining with {activation} activation function...')

    # Initialize MLP model with the chosen activation function
    model = MLP(input_size, hidden_size, output_size, activation=activation)

    # Training loop
    for epoch in range(epochs):
        # Forward propagation
        output = model.forward(x_train)
        loss = np.mean((output - y_train) ** 2)

        # Backpropagation
        model.backward(x_train, y_train, learning_rate)

        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}: Loss = {loss}')

    # Evaluation
    y_pred = model.forward(x_val)
    accuracy = np.mean((y_pred > 0.5) == y_val) * 100
    print(f'Validation Accuracy with {activation}: {accuracy:.2f}%')

    # Plot decision boundary and save
    plt.figure()
    part1PlotBoundary(x_val, y_val, model)
    plt.title(f'Decision Boundary - {activation}')
    plt.savefig(f'results/decision_boundary_{activation}.png')
    plt.close()
    print(f'Decision boundary plot saved for {activation} activation.')
