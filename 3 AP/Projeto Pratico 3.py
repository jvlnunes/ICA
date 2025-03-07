import numpy as np
from functions_mlp import *

# ================================== Carregar Dados de Treinamento ==================================
arquivos_treinamento = [
    {'nome': 'Seção5.10_RNA', 'caminho': "3 AP/Dados/Treinamento/Tabela#Seção5.10_RNA.txt"}
]

try:
    file_path = arquivos_treinamento[0]['caminho'].replace('/', '\\')
    X, Y = load_data(file_path)
    print('SO identificado: Windows')
except Exception as e:
    file_path = arquivos_treinamento[0]['caminho']
    X, Y = load_data(file_path)
    print('SO identificado: Linux')


# ================================== Questão 1 e 2: Treinamento ==================================
print('\nResultados Questão 1 e 2:')

# Parâmetros do treinamento
taxa_aprendizagem = 0.1
momentum = 0.8
precisao = 0.5e-6

# Topologias candidatas
topologias = [
    {"entradas": 5, "neuronios": 10},  # TDNN 1
    {"entradas": 10, "neuronios": 15},  # TDNN 2
    {"entradas": 15, "neuronios": 25}   # TDNN 3
]

# Lista para armazenar os resultados
resultados_treinamentos = []

# Executar três treinamentos para cada topologia
for idx, topologia in enumerate(topologias):
    resultados_tdnn = []
    for treinamento in range(3):
        np.random.seed(treinamento)
        W1, B1, W2, B2, mse, epocas, w1_inicial, w2_inicial, erros_em, tempo_processamento = treinamento_pmc_com_momentum(
            X, Y, neuronios=topologia["neuronios"], taxa_aprendizagem=taxa_aprendizagem, momentum=momentum, precisao=precisao
        )
        resultados_tdnn.append({
            'W1': W1,
            'B1': B1,
            'W2': W2,
            'B2': B2,
            'mse': mse,
            'epocas': epocas,
            'erros_em': erros_em 
        })
    resultados_treinamentos.append(resultados_tdnn)

# Exibir os resultados na tabela
exibir_resultados_treinamentos(resultados_treinamentos)

# ================================== Questão 3: Validação ==================================
print('\nResultados Questão 3:')

# Carregar dados de teste
arquivos_teste = [
    {'nome': '5.7_RNA', 'caminho': "3 AP/Dados/Teste/Tabela#5.7_RNA.txt"}
]

try:
    X_teste, Y_teste = load_data(arquivos_teste[0]['caminho'].replace('/', '\\'))
except:
    X_teste, Y_teste = load_data(arquivos_teste[0]['caminho'])
    
if Y_teste.shape[1] != 1:
    Y_teste = Y_teste[:, 0].reshape(-1, 1)  

# Lista para armazenar métricas de validação
resultados_validacao = []

for idx_tdnn, resultados_tdnn in enumerate(resultados_treinamentos):
    validacao_tdnn = []
    for treinamento, resultado in enumerate(resultados_tdnn):
        # Carregar pesos do treinamento
        W1 = resultado['W1']
        B1 = resultado['B1']
        W2 = resultado['W2']
        B2 = resultado['B2']
        
        # Realizar previsões
        Y_pred = forward_pass(X_teste, W1, B1, W2, B2)
        
        # Verificar se Y_pred e Y_teste têm o mesmo formato
        if Y_pred.shape != Y_teste.shape:
            Y_pred = Y_pred[:, 0].reshape(-1, 1)  # Ajustar para o formato de Y_teste
        
        # Calcular métricas
        erro_medio = erro_relativo_medio(Y_pred, Y_teste)/5
        variancia  = variancia_erro(Y_pred, Y_teste)
        
        validacao_tdnn.append((Y_pred, erro_medio, variancia))
    resultados_validacao.append(validacao_tdnn)

# Exibir resultados de validação na tabela
exibir_resultados_validacao(X_teste, Y_teste, resultados_validacao)


# ================================== Questão 4: Gráfico do Erro Quadrático Médio ==================================
print('\nQuestão 4: Gráfico do Erro Quadrático Médio')

plotar_erro_quadratico_medio(resultados_treinamentos)


# ================================== Questão 5: Gráfico dos Valores Desejados vs Estimados ==================================
print('\nQuestão 5: Gráfico dos Valores Desejados vs Estimados')

plotar_valores_desejados_vs_estimados(X_teste, Y_teste, resultados_validacao)