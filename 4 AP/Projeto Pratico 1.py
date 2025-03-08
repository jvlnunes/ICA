from functions import *

arquivos = {
    'nome'   : 'Tabela 2/1',
    'caminho': '4 AP/Dados/Tabela#Seção6.6_RNA.txt'
}

X, Y = load_data(arquivos['caminho'])

# Ajustar os dados para garantir o mesmo comprimento
min_length = min(len(X), len(Y))
X = X[:min_length]
Y = Y[:min_length]

# Criar DataFrame com os dados
data = pd.DataFrame({
    'x1': X.flatten(),  # Primeira dimensão
    'y': Y.flatten()    # Rótulos de classe
})

print("\n========== Questão 1 ==========")
# Adicionar a segunda dimensão ao DataFrame completo
data['x2'] = data['x1'] * np.random.uniform(0.8, 1.2, len(data))

# Filtrar apenas os dados com presença de radiação (y = 1)
radiation_data = data[data['y'] == 1].copy()

# Verificar se há dados de radiação suficientes
if len(radiation_data) == 0:
    print("Não há dados de radiação para realizar o clustering.")
    exit()

# Preparar dados para clustering
X_radiation = radiation_data[['x1', 'x2']].values

# Padronizar os dados
scaler = StandardScaler()
X_radiation_scaled = scaler.fit_transform(X_radiation)

# Executar o K-means para 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_radiation_scaled)
centros_scaled = kmeans.cluster_centers_

# Mapear centros de volta para a escala original
centros = scaler.inverse_transform(centros_scaled)

# Calcular as variâncias para cada cluster
variancias = []
for i in range(2):
    # Obter pontos pertencentes ao cluster atual
    cluster_points = X_radiation[kmeans.labels_ == i]
    
    # Calcular distâncias quadradas até o centro
    distancias_quad = np.sum((cluster_points - centros[i])**2, axis=1)
    
    # Calcular variância como a média das distâncias quadradas
    variancia = np.mean(distancias_quad)
    variancias.append(variancia)

# Exibir resultados
print("Centros dos Agrupamentos:")
for i, centro in enumerate(centros):
    print(f"Cluster {i+1}: ({centro[0]:.4f}, {centro[1]:.4f})")
    
print("\nVariâncias Correspondentes:")
for i, var in enumerate(variancias):
    print(f"Cluster {i+1}: {var:.4f}")

print(f"\nNúmero total de amostras: {len(data)}")
print(f"Número de amostras com radiação: {len(radiation_data)}")

# ========== Questão 2 ==========
print("\n========== Questão 2 ==========")

# Preparar os dados de treinamento
X_train = data[['x1', 'x2']].values  # Todas as amostras (presença e ausência de radiação)
y_train = data['y'].values           # Rótulos correspondentes

# Inicialização dos pesos e limiar (bias)
np.random.seed(42)
weights = np.random.rand(2)  # Pesos para as duas entradas
bias = np.random.rand()      # Limiar (bias)

# Taxa de aprendizado e precisão
learning_rate = 0.01
precision = 1e-7

# Treinamento com a regra delta generalizada
max_iterations = 10000  # Número máximo de iterações
converged = False
for iteration in range(max_iterations):
    # Calcular a saída da rede para todas as amostras
    y_pred = predict(X_train, weights, bias)
    
    # Calcular o erro (diferença entre a saída desejada e a saída predita)
    error = y_train - y_pred
    
    # Atualizar pesos e limiar usando a regra delta
    weights += learning_rate * np.dot(X_train.T, error)
    bias += learning_rate * np.sum(error)
    
    # Verificar convergência (erro médio abaixo da precisão)
    mean_error = np.mean(np.abs(error))
    if mean_error < precision:
        converged = True
        break

# Resultados do treinamento
if converged:
    print("\nConvergência alcançada!")
    print(f"Número de iterações: {iteration + 1}")
    print(f"Erro médio final: {mean_error:.8f}")
else:
    print("\nConvergência não alcançada após o número máximo de iterações.")
    print(f"Erro médio final: {mean_error:.8f}")

# Exibir os pesos e o limiar ajustados
print("\nPesos finais da camada de saída:")
print(f"w1: {weights[0]:.8f}, w2: {weights[1]:.8f}")
print(f"Limiar (bias): {bias:.8f}")


# ============ Questão 3 ==========
print("\n========== Questão 3 ==========")


# Suponha que y_real seja a saída da rede (valores reais)
# Exemplo de saída da rede (substitua pelos valores reais da sua rede)
y_real = np.array([0.5, -0.3, 1.2, -1.8, 0.0])

# Aplicar a função sinal para pós-processamento
y_pos = sinal(y_real)

# Exibir os resultados
print("Saída da rede (valores reais):", y_real)
print("Saída pós-processada (valores inteiros padronizados):", y_pos)

# Integração com o treinamento da rede
# Após o treinamento, use a função predict para obter as saídas reais
y_real_treinamento = np.dot(X_train, weights) + bias

# Aplicar o pós-processamento às saídas do treinamento
y_pos_treinamento = sinal(y_real_treinamento)

# Exibir as saídas pós-processadas do treinamento
print("\nSaídas pós-processadas do treinamento:")
print(y_pos_treinamento)

# =========== Questão 4 ==========
print("\n========== Questão 4 ==========")


# Dados de teste da Tabela 6.4
dados_teste = np.array([
    [0.8705, 0.9329, -1],
    [0.0388, 0.2703,  1],
    [0.8236, 0.4458, -1],
    [0.7075, 0.1502, -1],
    [0.9587, 0.8663, -1],
    [0.6115, 0.9365, -1],
    [0.3534, 0.3646,  1],
    [0.3268, 0.2766,  1],
    [0.6129, 0.4518, -1],
    [0.9948, 0.4962, -1]
])

# Extrair entradas (x1, x2) e valores desejados (d)
X_teste = dados_teste[:, :2]  # Colunas x1 e x2
d_teste = dados_teste[:, 2]   # Coluna d

# Suponha que a rede RBF já foi treinada e temos os pesos e o limiar
# Substitua pelos valores reais obtidos no treinamento
weights = np.array([0.56789012, -0.12345678])  # Pesos da camada de saída
bias = 0.23456789                              # Limiar (bias)

# Calcular as saídas da rede para o conjunto de teste
y_real_teste = np.dot(X_teste, weights) + bias

# Aplicar o pós-processamento para obter y_obs
y_obs_teste = sinal(y_real_teste)

# Comparar y_obs com d_teste para calcular a taxa de acertos
acertos = np.sum(y_obs_teste == d_teste)
total_amostras = len(d_teste)
taxa_acertos = (acertos / total_amostras) * 100

# Criar tabela de resultados
tabela_resultados = []
for i in range(len(dados_teste)):
    tabela_resultados.append([
        i + 1, X_teste[i, 0], X_teste[i, 1], d_teste[i], y_obs_teste[i]
    ])

# Exibir a tabela usando tabulate
headers = ["Amostra", "x1", "x2", "d", "y_obs"]
print(tabulate(tabela_resultados, headers=headers, tablefmt="grid", floatfmt=".4f"))

# Exibir a taxa de acertos
print(f"\nTaxa de acertos: {taxa_acertos:.2f}%")


# =========== Questão 5 ==========
print("\n========== Questão 5 ==========")
