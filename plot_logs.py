import argparse
import json

import matplotlib.pyplot as plt

# Get experiment id
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, required=True)
args = parser.parse_args()

log_path = f"logs/{args.id}/logs.json"

with open(log_path, "r") as f:
    logs = json.load(f)

train_loss = logs["train_loss"]
test_loss = logs["test_loss"]
train_acc = logs["train_acc"]
test_acc = logs["test_acc"]

fig, axs = plt.subplots(2, 1, figsize=(5, 10))  # Create 1 row, 2 columns of subplots

# Plot the losses on the first subplot
axs[0].plot(train_loss, label="train loss")
axs[0].plot(test_loss, label="test loss")
axs[0].legend()
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].set_title("Loss")

# Plot the accuracies on the second subplot
axs[1].plot(train_acc, label="train acc")
axs[1].plot(test_acc, label="test acc")
axs[1].legend()
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].set_title("Accuracy")

plt.tight_layout()  # Adjust the layout so plots don't overlap
plt.savefig("logs.png")  # Save the plot to a file
