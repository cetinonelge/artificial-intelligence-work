# Part 2: Implementing a Convolutional Layer with NumPy
import numpy as np
import matplotlib.pyplot as plt
from utils import part2Plots

# Load input and kernel
input = np.load('data/samples_6.npy')
kernel = np.load('data/kernel.npy')

# My Conv2D function
def my_conv2d(input, kernel):
    batch_size, input_channels, input_height, input_width = input.shape
    output_channels, input_channels, filter_height, filter_width = kernel.shape
    
    # Calculate output dimensions
    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1
    
    # Initialize the output tensor
    out = np.zeros((batch_size, output_channels, output_height, output_width))

    # Perform convolution
    for batch in range(batch_size):
        for out_ch in range(output_channels):
            for in_ch in range(input_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        region = input[batch, in_ch, i:i+filter_height, j:j+filter_width]
                        out[batch, out_ch, i, j] += np.sum(region * kernel[out_ch, in_ch])

    # Save the output
    np.save('out.npy', out)
    return out

# Get the convolution output
out = my_conv2d(input, kernel)

# Plot the output using part2Plots
part2Plots(out, filename='results/CNN_out')
print("Convolution completed and output saved.")
