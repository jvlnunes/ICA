import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = lines[0].strip()
    data = np.loadtxt(lines[1:], dtype=float)

    num_colunas = len(header.split())

    if num_colunas == 4:
        X = data[:, :-1]
        Y = data[:,  -1].reshape(-1, 1)
        
    elif num_colunas == 7:
        X = data[:, :4] 
        Y = data[:, 4:] 
        
    else:
        X = np.arange(len(data)).reshape(-1, 1)
        Y = data.reshape(-1, 1)
        
    return X, Y

# Função de ativação (função degrau)
def step_function(z):
    return np.where(z >= 0, 1, -1)

# Função para calcular a saída da rede
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return step_function(z)


# Função sinal para pós-processamento
def sinal(y):
    return np.where(y >= 0, 1, -1)

