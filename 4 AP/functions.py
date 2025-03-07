import numpy as np


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
