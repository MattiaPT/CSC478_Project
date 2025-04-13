#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar100
from keras.utils import to_categorical

# ----------------------- General ------------------------
# Authors: Shabnam Tajik and Mattia Taiana
# (this code was written taking into account inputs from generative models)
# --------------------------------------------------------

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train = X_train.astype(np.float32)/255.0
X_test = X_test.astype(np.float32)/255.0
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T

# One-hot labels
Nc = 100
y_train = to_categorical(y_train, Nc).T
y_test = to_categorical(y_test, Nc).T

# DAM Hyperparameters
n = 3                  # Interaction power for energy function
m = 4                  # Interaction power for retrieval
K = 40                 # Prototypes per class
eps0 = 0.001           # Initial learning rate
f = 0.99               # Decay rate
p = 0.92               # Momentum
weight_decay = 5e-4    # Weight decay
grad_clip = 0.1        # Gradient clipping
Nep = 3000             # Epochs
Num = 1024             # Batch size
stable_eps = 1e-5      # Numerical stability
smooth = 0.01          # Label smoothing

# Polynomial activation for DAM
def energy_activation(x, power):
    return np.maximum(x, 0)**power

# Initialize weights
def glorot_init(shape):
    scale = np.sqrt(2.0/(shape[0]+shape[1]))
    return np.random.randn(*shape) * scale

synapses = glorot_init((Nc*K, X_train.shape[0]))

# Training setup
acc_tr, acc_test = [], []
plt.ion()
fig = plt.figure(figsize=(20, 8))

for nep in range(Nep):
    # --- Training ---
    idx = np.random.permutation(X_train.shape[1])[:Num]
    x, lab = X_train[:, idx], y_train[:, idx]
    lab = lab * (1 - smooth) + smooth/Nc  # Label smoothing

    # Forward pass with polynomial interactions
    S = np.dot(synapses, x)
    F = energy_activation(S, m)  # Using m for forward pass

    # Class-wise energy aggregation
    energies = F.reshape(Nc, K, -1).sum(axis=1)  # Sum over prototypes
    energies = energy_activation(energies, n/m)   # Normalize power
    T = 0.5  # Temperature parameter
    energies = energies / T  # Sharpen probabilities

    # Stable softmax
    energies_exp = np.exp(energies - np.max(energies, axis=0, keepdims=True))
    probs = energies_exp / (np.sum(energies_exp, axis=0, keepdims=True) + stable_eps)

    # Accuracy
    pred = np.argmax(probs, axis=0)
    true = np.argmax(lab, axis=0)
    train_acc = np.mean(pred == true)
    acc_tr.append(train_acc)

    # Gradient calculation
    dE = (probs - lab) / Num
    dE_expanded = np.repeat(dE, K, axis=0)

    # Backprop through polynomial activation
    dF = dE_expanded * m * np.maximum(S, 0)**(m-1)
    dW = np.dot(dF, x.T)

    # Gradient clipping
    dW = np.clip(dW, -grad_clip, grad_clip)
    grad_norm = np.linalg.norm(dW)

    # Momentum update
    if nep == 0:
        dsynapses = dW
    else:
        dsynapses = p * dsynapses + (1-p) * dW

    # Learning rate schedule
    eps = max(1e-6, eps0 * f**nep)
    synapses -= eps * dsynapses
    synapses *= (1 - weight_decay)

    # --- Testing ---
    idx = np.random.permutation(X_test.shape[1])[:Num]
    x_test, lab_test = X_test[:, idx], y_test[:, idx]

    # Test forward pass
    S_test = np.dot(synapses, x_test)
    F_test = energy_activation(S_test, m)
    energies_test = F_test.reshape(Nc, K, -1).sum(axis=1)
    energies_test = energy_activation(energies_test, n/m)

    # Test softmax
    energies_test_exp = np.exp(energies_test - np.max(energies_test, axis=0, keepdims=True))
    probs_test = energies_test_exp / (np.sum(energies_test_exp, axis=0, keepdims=True) + stable_eps)

    test_pred = np.argmax(probs_test, axis=0)
    test_true = np.argmax(lab_test, axis=0)
    test_acc = np.mean(test_pred == test_true)
    acc_test.append(test_acc)

    # --- Monitoring ---
    active = np.mean(S > 0)
    print(f"Epoch {nep:3d}: "
          f"Acc = {train_acc:.1%}/{test_acc:.1%} | "
          f"Grad = {grad_norm:.3f} | "
          f"Active = {active:.1%} | "
          f"LR = {eps:.1e}")

    # Visualization
    if nep % 10 == 0:
        plt.clf()
        ax1 = fig.add_subplot(121)
        weights = synapses.reshape(Nc*K, 32, 32, 3)
        prototypes = weights[np.arange(0, Nc*K, K)]
        prototypes = (prototypes - np.min(prototypes, axis=(1,2,3), keepdims=True))
        prototypes = prototypes / (np.max(prototypes, axis=(1,2,3), keepdims=True) + 1e-8)
        combined = np.concatenate(prototypes, axis=1)
        ax1.imshow(combined)
        ax1.set_title(f'Prototypes (Epoch {nep})')
        ax1.axis('off')

        ax2 = fig.add_subplot(122)
        ax2.plot(acc_tr, 'b', label='Train')
        ax2.plot(acc_test, 'r', label='Test')
        ax2.set_title('Accuracy Progress')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
