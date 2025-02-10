import numpy as np
from tabulate import tabulate

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
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

def treinamento_pmc(X, Y, neuronios=5, taxa_aprendizagem=0.01, precisao=1e-6):
    np.random.seed()
    entrada, saida = X.shape[1], Y.shape[1]
    
    W1 = np.random.uniform(-0.5, 0.5, (entrada   , neuronios))
    B1 = np.random.uniform(-0.5, 0.5, (1         , neuronios))
    W2 = np.random.uniform(-0.5, 0.5, (neuronios , saida    ))
    B2 = np.random.uniform(-0.5, 0.5, (1         , saida    ))
    
    w1_inicial = np.copy(W1)
    w2_inicial = np.copy(W2)
    
    erro_anterior = float('inf')
    epocas        = 0
    erro_x_media  = []
    
    while True:
        epocas += 1
        
        h_i = np.dot(X  , W1) + B1
        h_o = sigmoide(h_i)
        f_i = np.dot(h_o, W2) + B2
        f_o = sigmoide(f_i)
        
        error = Y - f_o
        mse   = np.mean(error ** 2)
        
        erro_x_media.append({'erro': mse.tolist(), 'epocas': epocas})
        
        if abs(erro_anterior - mse) < precisao:
            break
        
        erro_anterior = mse
        
        d_output = error * derivada_sigmoide(f_o)
        d_hidden = np.dot(d_output, W2.T) * derivada_sigmoide(h_o)
    
        W2 += taxa_aprendizagem * np.dot(h_o.T, d_output)
        B2 += taxa_aprendizagem * np.sum(d_output, axis=0, keepdims=True)
        W1 += taxa_aprendizagem * np.dot(X.T, d_hidden)
        B1 += taxa_aprendizagem * np.sum(d_hidden, axis=0, keepdims=True)
    
    return W1, B1, W2, B2, mse, epocas, w1_inicial, w2_inicial, erro_x_media

def forward_pass(X, W1, B1, W2, B2):
    Z1     = np.dot(X, W1) + B1
    A1     = sigmoide(Z1)
    Z2     = np.dot(A1, W2) + B2
    Y_pred = sigmoide(Z2)
    
    return Y_pred

def erro_relativo_medio(Y_pred, Y_true):
    return np.mean(np.abs((Y_pred - Y_true) / Y_true)) * 100

def variancia_erro(Y_pred, Y_true):
    erros = np.abs((Y_pred - Y_true) / Y_true) * 100
    return np.var(erros)


def teste_pmc(X_teste, Y_teste, treinamentos):
    num_amostras = len(Y_teste)
    
    resultados = {
        "Amostra": list(range(1, num_amostras + 1)),
        "x1"     : X_teste[:, 0].tolist(),
        "x2"     : X_teste[:, 1].tolist(),
        "x3"     : X_teste[:, 2].tolist(),
        "d"      : Y_teste.tolist()
    }
    
    for i, treino in enumerate(treinamentos):
        W1 = np.array(treino['W1'])
        B1 = np.array(treino['B1'])
        W2 = np.array(treino['W2'])
        B2 = np.array(treino['B2'])
        
        Y_pred = forward_pass(X_teste, W1, B1, W2, B2).flatten()
        resultados[f"y (T{i+1})"] = Y_pred.tolist()
    
    erros_medios = [erro_relativo_medio(np.array(resultados[f"y (T{i+1})" ]), Y_teste) for i in range(len(treinamentos))]
    variancias   = [variancia_erro(np.array(resultados[f"y (T{i+1})"      ]), Y_teste) for i in range(len(treinamentos))]

    return resultados, erros_medios, variancias

def imprimir_resultados(resultados, erros_medios, variancias):
    colunas = ["Amostra", "x1", "x2", "x3", "d"] + [f"y (T{i+1})" for i in range(len(erros_medios))]

    tabela = []
    for i in range(len(resultados["Amostra"])):
        linha = [
            resultados["Amostra"][i], 
            f"{float(resultados['x1'][i]):.6f}", 
            f"{float(resultados['x2'][i]):.6f}", 
            f"{float(resultados['x3'][i]):.6f}", 
            f"{float(resultados['d'][i][0]):.6f}"  # <-- Correção aqui!
        ] + [f"{float(resultados[f'y (T{j+1})'][i]):.6f}" for j in range(len(erros_medios))]
        
        tabela.append(linha)

    tabela.append(["Erro relativo médio (%)", "-", "-", "-", "-"] + [f"{erro:.6f}" for erro in erros_medios])
    tabela.append(["Variância (%)", "-", "-", "-", "-"] + [f"{var:.6f}" for var in variancias])

    print(tabulate(tabela, headers=colunas, tablefmt="pretty"))


