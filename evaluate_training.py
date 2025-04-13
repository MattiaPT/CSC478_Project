#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np

# ----------------------- General ------------------------
# Authors: Shabnam Tajik and Mattia Taiana
# Script to create plots from model training output
# --------------------------------------------------------

batch_losses = []
epoch_avg_losses = []
eval_metrics = []
current_epoch = None
batches_per_epoch = None

# Regex patterns
batch_pattern = re.compile(r'Batch (\d+)/\d+ \| Loss: ([\d.]+) \| Class: ([\d.]+) \| Box: ([\d.]+)')
epoch_complete_pattern = re.compile(r'Epoch (\d+) Complete \| Avg Loss: ([\d.]+) \| Time: ([\d.]+)min')
eval_pattern = re.compile(r'Evaluation mAP: ([\d.]+) \| mAP50: ([\d.]+)')
epoch_start_pattern = re.compile(r'Epoch (\d+)/\d+ starting')
dataloader_pattern = re.compile(r'DataLoader length: (\d+)')

# Parse log file
with open('dam_rcnn_training.log', 'r') as f:
    for line in f:
        # Check for epoch start
        epoch_start = epoch_start_pattern.search(line)
        if epoch_start:
            current_epoch = int(epoch_start.group(1))

        # Check for DataLoader length
        dataloader = dataloader_pattern.search(line)
        if dataloader:
            batches_per_epoch = int(dataloader.group(1))

        # Check for batch information
        batch_match = batch_pattern.search(line)
        if batch_match and current_epoch and batches_per_epoch:
            batch_num = int(batch_match.group(1))
            loss = float(batch_match.group(2))
            class_loss = float(batch_match.group(3))
            box_loss = float(batch_match.group(4))

            global_batch = (current_epoch - 1) * batches_per_epoch + batch_num
            batch_losses.append((global_batch, loss, class_loss, box_loss))

        # Check for epoch completion
        epoch_complete = epoch_complete_pattern.search(line)
        if epoch_complete:
            epoch = int(epoch_complete.group(1))
            avg_loss = float(epoch_complete.group(2))
            time = float(epoch_complete.group(3))
            epoch_avg_losses.append((epoch, avg_loss, time))

        # Check for evaluation metrics
        eval_match = eval_pattern.search(line)
        if eval_match and current_epoch:
            mAP = float(eval_match.group(1))
            mAP50 = float(eval_match.group(2))
            eval_metrics.append((current_epoch - 1, mAP, mAP50))  # Associate with previous epoch

# Convert to numpy arrays for plotting
batch_losses = np.array(batch_losses)
epoch_avg_losses = np.array(epoch_avg_losses)
eval_metrics = np.array(eval_metrics)

# Create plots
plt.figure(figsize=(15, 18))

# Plot 1: mAP and mAP50 Evolution
plt.subplot(2, 1, 1)
if len(eval_metrics) > 0:
    plt.plot(eval_metrics[:, 0], eval_metrics[:, 1], 'b-o', label='mAP')
    plt.plot(eval_metrics[:, 0], eval_metrics[:, 2], 'g--s', label='mAP50')
    plt.ylabel('mAP Scores')
    plt.title('Evaluation Metric Progression')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No Evaluation Metrics Found', ha='center')

# Plot 2: Class vs Box Loss Components
plt.subplot(2, 1, 2)
plt.semilogy(batch_losses[:, 0], batch_losses[:, 2], alpha=0.4, label='Class Loss')
plt.semilogy(batch_losses[:, 0], batch_losses[:, 3], alpha=0.4, label='Box Loss')
plt.xlabel('Global Batch Number')
plt.ylabel('Loss Components (log scale)')
plt.title('Class vs Box Loss Components')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300)
plt.close()

print("Plots saved as training_analysis.png")
