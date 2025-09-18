#!/usr/bin/env python
# coding: utf-8

# # **Machine Learning Homework: Multilayer ANN from Scratch**
# 
# ---
# 
# ## **Objective**  
# Students will implement and train a multilayer artificial neural network from scratch, practicing both manual forward/backward propagation and coding the network in Python.

# ## **Chapter 11: Subtopics and Pages**
# 
# - 11.1 Introduction to Multilayer Perceptrons — 412
# - 11.2 Neuron Model and Activation Functions — 414
# - 11.3 Feedforward Propagation — 417
# - 11.4 Loss Functions (MSE and Cross-Entropy) — 420
# - 11.5 Backpropagation Derivation — 423
# - 11.6 Weight Initialization and Optimization — 427
# - 11.7 Vectorized Implementation in NumPy — 429
# - 11.8 Training the Network: Algorithm and Hyperparameters — 432

# ---
# 
# ## **Part 1: Manual Forward and Backward Propagation**
# 
# Network architecture: 2 inputs → 2 hidden neurons (sigmoid) → 1 output neuron (sigmoid). Learning rate α = 0.1.
# 
# ### Step 1: Given  
# - Input vector: x = [1.0, 2.0]  
# - Hidden layer weights: W1 = [[0.1, -0.2], [0.4, 0.2]]  
# - Hidden layer biases: b1 = [0.0, 0.1]  
# - Output layer weights: W2 = [0.3, -0.1]  
# - Output layer bias: b2 = -0.1  
# - Activation: σ(z) = 1/(1 + e^(−z))  
# - Loss: L = 0.5·(ŷ − y)², with true label y = 1
# 
# ### Step 2: Forward Pass  
# 2.1 Compute hidden pre-activations z¹ = W1·x + b1  
# 2.2 Compute hidden activations a¹ = σ(z¹)  
# 2.3 Compute output pre-activation z² = W2·a¹ + b2  
# 2.4 Compute output ŷ = σ(z²)  
# 2.5 Compute loss L

# ### Step 3: Backpropagation  
# 3.1 Compute δ² = (ŷ − y) · σ′(z²)  
# 3.2 Compute gradients:  
# - ∂L/∂W2 = δ² · a¹ᵀ  
# - ∂L/∂b2 = δ²  
# 3.3 Compute δ¹ = (W2ᵀ · δ²) * σ′(z¹)  
# 3.4 Compute gradients:  
# - ∂L/∂W1 = δ¹ · xᵀ  
# - ∂L/∂b1 = δ¹

# ### Step 4: Weight Updates  
# Update rules:  
# - W2_new = W2 − α · ∂L/∂W2  
# - b2_new = b2 − α · ∂L/∂b2  
# - W1_new = W1 − α · ∂L/∂W1  
# - b1_new = b1 − α · ∂L/∂b1  
# 
# Perform these updates and report the new weight matrices and biases.

# ---
# 
# ## **Part 2: Python Implementation**
# 
# Re-implement the same 2-2-1 network using NumPy. Follow each step closely.

# ### Step 1: Define activation functions and their derivatives
# ```python
# import numpy as np
# 
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
# 
# def sigmoid_prime(z):
#     s = sigmoid(z)
#     return s * (1 - s)
# ```

# ### Step 2: Initialize parameters
# ```python
# def initialize_parameters(n_x, n_h, n_y):
#     W1 = np.random.randn(n_h, n_x) * 0.01
#     b1 = np.zeros((n_h, 1))
#     W2 = np.random.randn(n_y, n_h) * 0.01
#     b2 = np.zeros((n_y, 1))
#     return W1, b1, W2, b2
# ```

# In[ ]:





# ### Step 3: Forward and backward propagation functions
# ```python
# def forward_propagation(X, W1, b1, W2, b2):
#     Z1 = W1.dot(X) + b1
#     A1 = sigmoid(Z1)
#     Z2 = W2.dot(A1) + b2
#     A2 = sigmoid(Z2)
#     cache = { 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2 }
#     return A2, cache
# 
# def compute_cost(A2, Y):
#     m = Y.shape[1]
#     cost = 0.5 * np.sum((A2 - Y)**2) / m
#     return cost
# 
# def backward_propagation(X, Y, cache, W2):
#     m = X.shape[1]
#     A1, A2 = cache['A1'], cache['A2']
#     Z1, Z2 = cache['Z1'], cache['Z2']
# 
#     dZ2 = (A2 - Y) * sigmoid_prime(Z2)
#     dW2 = (dZ2.dot(A1.T)) / m
#     db2 = np.sum(dZ2, axis=1, keepdims=True) / m
# 
#     dZ1 = (W2.T.dot(dZ2)) * sigmoid_prime(Z1)
#     dW1 = (dZ1.dot(X.T)) / m
#     db1 = np.sum(dZ1, axis=1, keepdims=True) / m
# 
#     grads = { 'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2 }
#     return grads
# ```

# In[ ]:





# ### Step 4: Training loop
# ```python
# def train(X, Y, n_h, num_iterations=10000, learning_rate=0.1):
#     n_x = X.shape[0]
#     n_y = Y.shape[0]
#     W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y)
#     costs = []
#     for i in range(num_iterations):
#         A2, cache = forward_propagation(X, W1, b1, W2, b2)
#         cost = compute_cost(A2, Y)
#         grads = backward_propagation(X, Y, cache, W2)
#         W1 -= learning_rate * grads['dW1']
#         b1 -= learning_rate * grads['db1']
#         W2 -= learning_rate * grads['dW2']
#         b2 -= learning_rate * grads['db2']
# 
#         if i % 1000 == 0:
#             costs.append(cost)
#             print(f"Cost after iteration {i}: {cost:.6f}")
#     return { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'costs': costs }
# ```

# In[ ]:





# ### Step 5: Experiment and Analysis  
# - Train the network on the XOR dataset.  
# - Plot the loss curve (use `matplotlib`).  
# - Change the hidden activation to ReLU and compare final loss.  
# 
# ---
# 
# ## **Submission Requirements**  
# Submit a single Jupyter notebook containing:  
# - All manual calculation steps with numeric answers.  
# - The complete Python code with comments.  
# - Plots of the loss curve and decision boundary.  
# - A brief report summarizing your observations and comparing activations.  
# 
# *Ali Fahim*  
# *University of Tehran*
