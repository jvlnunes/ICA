import pandas as pd
import numpy  as np
from   functions_mlp     import *
import matplotlib.pyplot as plt

arquivos_treinamento = [
    {'nome': 'Seção5.9_RNA' , 'caminho': "3 AP/Dados/Treinamento/Tabela#Seção5.9_RNA.txt"  }
]

print("Projeto prático 02")
try:
    file_path = arquivos_treinamento[0]['caminho'].replace('/', '\\')
    X, Y = load_data(file_path)
    print('SO identificado: Windows')
    
except Exception as e:
    file_path = arquivos_treinamento[0]['caminho']
    X, Y = load_data(file_path)
    print('SO identificado: Linux') 

# ================================== Questão 1 ==================================
print('\nResultados Questão 1:')
precisao   = 1e-6
neuronios  = 5
taxa_apren = 0.01

W1, B1, W2, B2, mse, epocas, W1_inicial, W2_inicial, erro_x_epocas, tempo_processamento = treinamento_pmc(X, Y, neuronios=neuronios, precisao=precisao, taxa_aprendizagem=taxa_apren)

resultados, erros_medios, variancias, maes, mses = teste_pmc(X, Y, [{
    'W1': W1,
    'B1': B1,
    'W2': W2,
    'B2': B2
}])

imprimir_resultados(resultados, erros_medios, variancias, maes, mses)

# ================================== Questão 2 ==================================
print('\nResultados Questão 2:')
momentum   = 0.9
precisao   = 1e-6
neuronios  = 5  
taxa_apren = 0.1

# Treinar a rede PMC com momentum
W1, B1, W2, B2, mse, epocas, w1_inicial, w2_inicial, erro_x_media, tempo_processamento = treinamento_pmc_com_momentum( X, Y, neuronios=neuronios, taxa_aprendizagem=taxa_apren, momentum=momentum, precisao=precisao )

# Avaliar o desempenho da rede
resultados, erros_medios, variancias, maes, mses = teste_pmc(X, Y, [{
    'W1': W1,
    'B1': B1,
    'W2': W2,
    'B2': B2
}])

imprimir_resultados(resultados, erros_medios, variancias, maes, mses)


# ================================== Questão 3 ==================================
print('\nResultados Questão 3:')

# Parâmetros do treinamento
taxa_apren = 0.1
momentum   = 0.9
precisao   = 1e-6
neuronios  = 5  

# Treinamento sem momentum
W1_sem_momentum, B1_sem_momentum, W2_sem_momentum, B2_sem_momentum, mse_sem_momentum, epocas_sem_momentum, w1_inicial_sem_momentum, w2_inicial_sem_momentum, erros_em_sem_momentum, tempo_sem_momentum = treinamento_pmc(
    X, Y, neuronios=neuronios, taxa_aprendizagem=taxa_apren, precisao=precisao
)

# Treinamento com momentum
W1_com_momentum, B1_com_momentum, W2_com_momentum, B2_com_momentum, mse_com_momentum, epocas_com_momentum, w1_inicial_com_momentum, w2_inicial_com_momentum, erros_em_com_momentum, tempo_com_momentum = treinamento_pmc_com_momentum(
    X, Y, neuronios=neuronios, taxa_aprendizagem=taxa_apren, momentum=momentum, precisao=precisao
)

# Plotar os gráficos
plt.figure(figsize=(12, 6))

# Gráfico sem momentum
plt.subplot(1, 2, 1)
plt.plot(range(1, epocas_sem_momentum + 1), erros_em_sem_momentum, label='Sem Momentum', color='blue')
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Médio (EM)')
plt.title(f'Sem Momentum\nTempo: {tempo_sem_momentum:.2f} segundos')
plt.legend()
plt.grid()

# Gráfico com momentum
plt.subplot(1, 2, 2)
plt.plot(range(1, epocas_com_momentum + 1), erros_em_com_momentum, label='Com Momentum', color='red')
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Médio (EM)')
plt.title(f'Com Momentum\nTempo: {tempo_com_momentum:.2f} segundos')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()




# ================================== Questão 4 ==================================
print('\nResultados Questão 4:')

# Parâmetros do treinamento
taxa_aprendizagem = 0.1
momentum = 0.9
precisao = 1e-6
neuronios = 5  

# Treinar a rede PMC com momentum
W1, B1, W2, B2, mse, epocas, w1_inicial, w2_inicial, erros_em, tempo_processamento = treinamento_pmc_com_momentum(
    X, Y, neuronios=neuronios, taxa_aprendizagem=taxa_aprendizagem, momentum=momentum, precisao=precisao
)

# Fazer previsões com a rede treinada
Y_pred = forward_pass(X, W1, B1, W2, B2)

# Aplicar o pós-processamento
Y_pos = pos_processamento(Y_pred)

# Exibir os resultados
print("Saídas da rede (valores reais):")
print(Y_pred)
print("\nSaídas pós-processadas (valores inteiros):")
print(Y_pos)


# ================================== Questão 5 ==================================
print('\nResultados Questão 5:')

arquivos_teste = [
    {'nome': '5.5_RNA', 'caminho': "3 AP/Dados/Teste/Tabela#5.5_RNA.txt" }
]

try:
    X_teste, Y_teste = load_data(arquivos_teste[0]['caminho'].replace('/', '\\'))
    
except:
    X_teste, Y_teste = load_data(arquivos_teste[0]['caminho'])
    
# Função para fazer previsões com a rede treinada
def fazer_previsoes(X, W1, B1, W2, B2):
    Y_pred = forward_pass(X, W1, B1, W2, B2)
    return Y_pred

# Fazer previsões com a rede treinada (sem momentum)
Y_pred_sem_momentum = fazer_previsoes(X_teste, W1_sem_momentum, B1_sem_momentum, W2_sem_momentum, B2_sem_momentum)

# Fazer previsões com a rede treinada (com momentum)
Y_pred_com_momentum = fazer_previsoes(X_teste, W1_com_momentum, B1_com_momentum, W2_com_momentum, B2_com_momentum)

# Função para pós-processamento das saídas
def pos_processamento(Y_pred):
    return (Y_pred >= 0.5).astype(int)

# Aplicar pós-processamento às previsões
Y_pred_sem_momentum_binario = pos_processamento(Y_pred_sem_momentum)
Y_pred_com_momentum_binario = pos_processamento(Y_pred_com_momentum)

# Função para calcular a taxa de acertos
def calcular_taxa_acertos(Y_pred, Y_true):
    acertos = np.all(Y_pred == Y_true, axis=1)  # Verifica se todas as saídas estão corretas para cada amostra
    taxa_acertos = np.mean(acertos) * 100  # Calcula a porcentagem de acertos
    return taxa_acertos

# Calcular a taxa de acertos para a rede sem momentum
taxa_acertos_sem_momentum = calcular_taxa_acertos(Y_pred_sem_momentum_binario, Y_teste)

# Calcular a taxa de acertos para a rede com momentum
taxa_acertos_com_momentum = calcular_taxa_acertos(Y_pred_com_momentum_binario, Y_teste)

print(f"Taxa de acertos (sem momentum): {taxa_acertos_sem_momentum:.2f}%")
print(f"Taxa de acertos (com momentum): {taxa_acertos_com_momentum:.2f}%")


# Exibir as previsões e as saídas pós-processadas
print("\nPrevisões sem momentum:")
print(Y_pred_sem_momentum)
print("\nSaídas pós-processadas sem momentum:")
print(Y_pred_sem_momentum_binario)

print("\nPrevisões com momentum:")
print(Y_pred_com_momentum)
print("\nSaídas pós-processadas com momentum:")
print(Y_pred_com_momentum_binario)

# Exibir a taxa de acertos
print(f"\nTaxa de acertos (sem momentum): {taxa_acertos_sem_momentum:.2f}%")
print(f"Taxa de acertos (com momentum): {taxa_acertos_com_momentum:.2f}%")

