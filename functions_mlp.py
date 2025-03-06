import time
import numpy    as np
from   tabulate import tabulate

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

    # Verificar o número de colunas no cabeçalho
    num_colunas = len(header.split())

    if num_colunas == 4:
        # Caso padrão: 4 colunas (3 entradas e 1 saída)
        X = data[:, :-1]
        Y = data[:, -1].reshape(-1, 1)
    elif num_colunas == 7:
        # Caso específico: 4 entradas (x1, x2, x3, x4) e 3 saídas (d1, d2, d3)
        X = data[:, :4]  # As primeiras 4 colunas são as entradas
        Y = data[:, 4:]  # As últimas 3 colunas são as saídas
    else:
        # Caso genérico (para outros formatos)
        X = np.arange(len(data)).reshape(-1, 1)
        Y = data.reshape(-1, 1)

    return X, Y

def treinamento_pmc_1(X, Y, neuronios=5, taxa_aprendizagem=0.01, precisao=1e-6):
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


def treinamento_pmc(X, Y, neuronios=5, taxa_aprendizagem=0.1, precisao=1e-6):
    np.random.seed()
    entrada, saida = X.shape[1], Y.shape[1]
    
    W1 = np.random.uniform(-0.5, 0.5, (entrada, neuronios))
    B1 = np.random.uniform(-0.5, 0.5, (1, neuronios))
    W2 = np.random.uniform(-0.5, 0.5, (neuronios, saida))
    B2 = np.random.uniform(-0.5, 0.5, (1, saida))
    
    w1_inicial = np.copy(W1)
    w2_inicial = np.copy(W2)
    
    erro_anterior = float('inf')
    epocas = 0
    erros_em = []  # Lista para armazenar o erro quadrático médio em cada época
    
    start_time = time.time()  # Iniciar a medição do tempo
    
    while True:
        epocas += 1
        
        # Forward pass
        h_i = np.dot(X, W1) + B1
        h_o = sigmoide(h_i)
        f_i = np.dot(h_o, W2) + B2
        f_o = sigmoide(f_i)
        
        # Cálculo do erro quadrático médio
        error = Y - f_o
        mse = np.mean(error ** 2)
        erros_em.append(mse)  # Armazenar o erro quadrático médio
        
        # Critério de parada
        if abs(erro_anterior - mse) < precisao:
            break
        
        erro_anterior = mse
        
        # Backpropagation
        d_output = error * derivada_sigmoide(f_o)
        d_hidden = np.dot(d_output, W2.T) * derivada_sigmoide(h_o)
        
        # Atualização dos pesos
        W2 += taxa_aprendizagem * np.dot(h_o.T, d_output)
        B2 += taxa_aprendizagem * np.sum(d_output, axis=0, keepdims=True)
        W1 += taxa_aprendizagem * np.dot(X.T, d_hidden)
        B1 += taxa_aprendizagem * np.sum(d_hidden, axis=0, keepdims=True)
    
    end_time = time.time()  # Finalizar a medição do tempo
    tempo_processamento = end_time - start_time
    
    return W1, B1, W2, B2, mse, epocas, w1_inicial, w2_inicial, erros_em, tempo_processamento
    
def treinamento_pmc_com_momentum(X, Y, neuronios=5, taxa_aprendizagem=0.1, momentum=0.9, precisao=1e-6):
    np.random.seed()
    entrada, saida = X.shape[1], Y.shape[1]
    
    W1 = np.random.uniform(-0.5, 0.5, (entrada, neuronios))
    B1 = np.random.uniform(-0.5, 0.5, (1, neuronios))
    W2 = np.random.uniform(-0.5, 0.5, (neuronios, saida))
    B2 = np.random.uniform(-0.5, 0.5, (1, saida))
    
    delta_W1_anterior = np.zeros_like(W1)
    delta_B1_anterior = np.zeros_like(B1)
    delta_W2_anterior = np.zeros_like(W2)
    delta_B2_anterior = np.zeros_like(B2)
    
    w1_inicial = np.copy(W1)
    w2_inicial = np.copy(W2)
    
    erro_anterior = float('inf')
    epocas = 0
    erros_em = []  # Lista para armazenar o erro quadrático médio em cada época
    
    start_time = time.time()  # Iniciar a medição do tempo
    
    while True:
        epocas += 1
        
        # Forward pass
        h_i = np.dot(X, W1) + B1
        h_o = sigmoide(h_i)
        f_i = np.dot(h_o, W2) + B2
        f_o = sigmoide(f_i)
        
        # Cálculo do erro quadrático médio
        error = Y - f_o
        mse = np.mean(error ** 2)
        erros_em.append(mse)  # Armazenar o erro quadrático médio
        
        # Critério de parada
        if abs(erro_anterior - mse) < precisao:
            break
        
        erro_anterior = mse
        
        # Backpropagation
        d_output = error * derivada_sigmoide(f_o)
        d_hidden = np.dot(d_output, W2.T) * derivada_sigmoide(h_o)
        
        # Atualização dos pesos com momentum
        delta_W2 = taxa_aprendizagem * np.dot(h_o.T, d_output) + momentum * delta_W2_anterior
        delta_B2 = taxa_aprendizagem * np.sum(d_output, axis=0, keepdims=True) + momentum * delta_B2_anterior
        delta_W1 = taxa_aprendizagem * np.dot(X.T, d_hidden) + momentum * delta_W1_anterior
        delta_B1 = taxa_aprendizagem * np.sum(d_hidden, axis=0, keepdims=True) + momentum * delta_B1_anterior
        
        # Aplicar as atualizações
        W2 += delta_W2
        B2 += delta_B2
        W1 += delta_W1
        B1 += delta_B1
        
        # Salvar as atualizações para o próximo passo
        delta_W2_anterior = delta_W2
        delta_B2_anterior = delta_B2
        delta_W1_anterior = delta_W1
        delta_B1_anterior = delta_B1
    
    end_time = time.time()  # Finalizar a medição do tempo
    tempo_processamento = end_time - start_time
    
    return W1, B1, W2, B2, mse, epocas, w1_inicial, w2_inicial, erros_em, tempo_processamento


def forward_pass(X, W1, B1, W2, B2):
    Z1     = np.dot(X, W1) + B1
    A1     = sigmoide(Z1)
    Z2     = np.dot(A1, W2) + B2
    Y_pred = sigmoide(Z2)
    
    return Y_pred

def erro_relativo_medio(Y_pred, Y_true):
    epsilon = 1e-10  # Valor pequeno para evitar divisão por zero
    erro_relativo = np.abs((Y_pred - Y_true) / (Y_true + epsilon)) * 100
    
    # Ignorar divisões por zero (onde Y_true é zero)
    erro_relativo[Y_true == 0] = 0
    
    return np.mean(erro_relativo)

def variancia_erro(Y_pred, Y_true):
    epsilon = 1e-10  # Valor pequeno para evitar divisão por zero
    erros = np.abs((Y_pred - Y_true) / (Y_true + epsilon)) * 100
    
    # Ignorar divisões por zero (onde Y_true é zero)
    erros[Y_true == 0] = 0
    
    return np.var(erros)

def mae(Y_pred, Y_true):
    return np.mean(np.abs(Y_pred - Y_true))

def mse(Y_pred, Y_true):
    return np.mean((Y_pred - Y_true) ** 2)

# def teste_pmc(X_teste, Y_teste, treinamentos):
#     num_amostras = len(Y_teste)
    
#     resultados = {
#         "Amostra": list(range(1, num_amostras + 1)),
#         "x1"     : X_teste[:, 0].tolist(),
#         "x2"     : X_teste[:, 1].tolist(),
#         "x3"     : X_teste[:, 2].tolist(),
#         "d"      : Y_teste.tolist()
#     }
    
#     for i, treino in enumerate(treinamentos):
#         W1 = np.array(treino['W1'])
#         B1 = np.array(treino['B1'])
#         W2 = np.array(treino['W2'])
#         B2 = np.array(treino['B2'])
        
#         Y_pred = forward_pass(X_teste, W1, B1, W2, B2).flatten()
#         resultados[f"y (T{i+1})"] = Y_pred.tolist()
    
#     erros_medios = [erro_relativo_medio(np.array(resultados[f"y (T{i+1})" ]), Y_teste) for i in range(len(treinamentos))]
#     variancias   = [variancia_erro(np.array(resultados[f"y (T{i+1})"      ]), Y_teste) for i in range(len(treinamentos))]

#     return resultados, erros_medios, variancias

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
        
        Y_pred = forward_pass(X_teste, W1, B1, W2, B2)
        resultados[f"y (T{i+1})"] = Y_pred.tolist()
    
    # Calcular erros médios, variâncias, MAE e MSE
    erros_medios = []
    variancias = []
    maes = []
    mses = []
    
    for i in range(len(treinamentos)):
        Y_pred = np.array(resultados[f"y (T{i+1})"])
        
        # Verificar o número de saídas
        if Y_teste.shape[1] == 1:
            # Caso padrão: 1 saída
            erro_medio = erro_relativo_medio(Y_pred.flatten(), Y_teste.flatten())
            variancia = variancia_erro(Y_pred.flatten(), Y_teste.flatten())
            mae_val = mae(Y_pred.flatten(), Y_teste.flatten())
            mse_val = mse(Y_pred.flatten(), Y_teste.flatten())
        else:
            # Caso específico: múltiplas saídas (3 colunas)
            erro_medio = np.mean([erro_relativo_medio(Y_pred[:, j], Y_teste[:, j]) for j in range(Y_teste.shape[1])])
            variancia = np.mean([variancia_erro(Y_pred[:, j], Y_teste[:, j]) for j in range(Y_teste.shape[1])])
            mae_val = np.mean([mae(Y_pred[:, j], Y_teste[:, j]) for j in range(Y_teste.shape[1])])
            mse_val = np.mean([mse(Y_pred[:, j], Y_teste[:, j]) for j in range(Y_teste.shape[1])])
        
        erros_medios.append(erro_medio)
        variancias.append(variancia)
        maes.append(mae_val)
        mses.append(mse_val)
    
    return resultados, erros_medios, variancias, maes, mses


# def imprimir_resultados(resultados, erros_medios, variancias):
#     colunas = ["Amostra", "x1", "x2", "x3", "d"] + [f"y (T{i+1})" for i in range(len(erros_medios))]

#     tabela = []
#     for i in range(len(resultados["Amostra"])):
#         linha = [
#             resultados["Amostra"][i], 
#             f"{float(resultados['x1'][i]):.6f}", 
#             f"{float(resultados['x2'][i]):.6f}", 
#             f"{float(resultados['x3'][i]):.6f}", 
#             f"{float(resultados['d'][i][0]):.6f}" 
#         ] + [f"{float(resultados[f'y (T{j+1})'][i]):.6f}" for j in range(len(erros_medios))]
        
#         tabela.append(linha)

#     tabela.append(["Erro relativo médio (%)", "-", "-", "-", "-"] + [f"{erro:.6f}" for erro in erros_medios])
#     tabela.append(["Variância (%)", "-", "-", "-", "-"]           + [f"{var:.6f}"  for var  in variancias  ])

#     print(tabulate(tabela, headers=colunas, tablefmt="pretty"))

def imprimir_resultados(resultados, erros_medios, variancias, maes, mses):
    colunas = ["Amostra", "x1", "x2", "x3", "d"] + [f"y (T{i+1})" for i in range(len(erros_medios))]

    tabela = []
    for i in range(len(resultados["Amostra"])):
        linha = [
            resultados["Amostra"][i], 
            f"{float( resultados['x1'][i]            ):.6f}", 
            f"{float( resultados['x2'][i]            ):.6f}", 
            f"{float( resultados['x3'][i]            ):.6f}", 
            f"{float( resultados['d' ][i][0]         ):.6f}"    
        ] + [f"{float(resultados[f'y (T{j+1})'][i][0]):.6f}" for j in range(len(erros_medios))]
        
        tabela.append(linha)

    tabela.append(["Erro relativo médio (%)", "-", "-", "-", "-"] + [f"{erro:.6f}"    for erro    in erros_medios])
    tabela.append(["Variância (%)"          , "-", "-", "-", "-"] + [f"{var:.6f}"     for var     in variancias  ])
    tabela.append(["Erro Absoluto Mediano"  , "-", "-", "-", "-"] + [f"{mae_val:.6f}" for mae_val in maes        ])
    tabela.append(["Erro Quadrático Mediano", "-", "-", "-", "-"] + [f"{mse_val:.6f}" for mse_val in mses        ])
   
    print(tabulate(tabela, headers=colunas, tablefmt="pretty"))

def pos_processamento(Y_pred):
    """
        Realiza o pós-processamento das saídas da rede.
        Arredonda os valores para 0 ou 1 com base no critério de arredondamento simétrico.
        
        Parâmetros:
            Y_pred (numpy array): Saídas da rede (valores reais).
        
        Retorna:
            numpy array: Saídas pós-processadas (valores inteiros 0 ou 1).
    """
    return np.where(Y_pred >= 0.5, 1, 0)