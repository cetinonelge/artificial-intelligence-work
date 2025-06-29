import matplotlib.pyplot as plt

# Replace with your actual 30 values from the console
scheduled_val_acc = [
    42.64, 50.34, 54.84, 55.98, 56.60, 61.62, 61.58, 61.72, 
    61.84, 62.66, 62.14, 63.14, 62.98, 63.18, 63.02, 63.24,
    63.22, 63.30, 63.60, 63.42, 63.52, 63.32, 63.38, 63.44,
    63.54, 63.50, 63.44, 63.66, 63.66, 63.82
]

epochs = range(1, len(scheduled_val_acc)+1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, scheduled_val_acc, marker='o')
plt.title("Validation Accuracy vs. Epoch (Scheduled LR Approach)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.ylim([0, 100])  # CIFAR-10 accuracy range
plt.grid(True)
plt.savefig("scheduled_lr_validation_curve.png")
plt.show()
