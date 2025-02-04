import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate  

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    return x * (1 - x)

def formatar_array(array):
    return "\n".join([str(a) for a in array])

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = lines[0].strip()
    data = np.loadtxt(lines[1:], dtype=float)

    if len(header.split()) == 4:
        X = data[:, :-1]
        Y = data[:, -1].reshape(-1, 1)

    elif len(header.split()) > 4:
        X = data[:, :-3]
        Y = data[:, -3:]

    else:
        X = np.arange(len(data)).reshape(-1, 1)
        Y = data.reshape(-1, 1)

    return X, Y

def treinamento(X, Y, layers, epocas=10000, taxa_aprendizagem=0.1, precisao=1e-6):
    np.random.seed()

    weights = [np.random.rand(layers[i], layers[i - 1]) - 0.5 for i in range(1, len(layers))]
    biases  = [np.random.rand(layers[i], 1)             - 0.5 for i in range(1, len(layers))]

    for epoch in range(epocas):
        # print(f'Epoch {epoch} weights = {len(weights)} biases = {len(biases)}')
        activations = [X.T]
        for w, b in zip(weights, biases):
            z = np.dot(w, activations[-1]) + b
            activations.append(sigmoid(z))

        deltas = [activations[-1] - Y.T]
        for i in range(len(layers) - 2, 0, -1):
            deltas.append(np.dot(weights[i].T, deltas[-1]) * derivada_sigmoid(activations[i]))

        deltas.reverse()

        for i in range(len(weights)):
            weights[i] -= taxa_aprendizagem * np.dot(deltas[i], activations[i].T)
            biases[i]  -= taxa_aprendizagem * np.sum(deltas[i], axis=1, keepdims=True)

        if np.max(np.abs(deltas[-1])) < precisao:
            print(f"Convergence reached at epoch {epoch}")
            break

    return weights, biases

def testar(X, pesos, biases):
    ativacoes = [X]
    for w, b in zip(pesos, biases):
        b = b.T  
        # print(f"Peso: {w.shape} Bias: {b.shape}")
        # print("Ativação anterior:", ativacoes[-1].shape)
        z = np.dot(ativacoes[-1], w.T) + b
        a = 1 / (1 + np.exp(-z)) 
        ativacoes.append(a)
    return ativacoes[-1]
