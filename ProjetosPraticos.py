from functions_mlp import *


print("Projetro prático 01")
arquivos_treinamento = [
    {'nome': 'Seção5.8_RNA' , 'caminho':"Dados\\Treinamento\\Tabela#Seção5.8_RNA.txt" },
    {'nome': 'Seção5.9_RNA' , 'caminho':"Dados\\Treinamento\\Tabela#Seção5.9_RNA.txt" },
    {'nome': 'Seção5.10_RNA', 'caminho':"Dados\\Treinamento\\Tabela#Seção5.10_RNA.txt"}
]

colunas     = ["Arquivo", "Pesos Finais", "Biases Finais"]
p      = []
b      = []
resultados  = []

for arquivo in arquivos_treinamento:
    X, Y = load_data(arquivo['caminho'])

    entrada = X.shape[1]
    saída   = Y.shape[1]

    neurônios = 5
    epocas = 1000
    taxa = 0.01
    prec = 1e-6

    weights, biases = treinamento(X, Y, layers=[entrada, neurônios, saída],epocas=epocas,taxa_aprendizagem=taxa,precisao=prec)
    # print(f' {arquivo["nome"]}\n weights = {weights}\n biases = {biases} ')

    p.append(weights)
    b.append(biases)
    pesos_finais  = formatar_array(weights)
    biases_finais = formatar_array(biases)

    resultados.append([arquivo['nome'], str(pesos_finais), str(biases_finais)])

print(tabulate(resultados, headers=colunas, tablefmt="grid"))



arquivos_teste = [
    {'nome': '5.3_RNA' , 'caminho':"Dados\\Teste\\Tabela#5.3_RNA.txt" },
    {'nome': '5.5_RNA' , 'caminho':"Dados\\Teste\\Tabela#5.5_RNA.txt" },
    {'nome': '5.7_RNA' , 'caminho':"Dados\\Teste\\Tabela#5.7_RNA.txt" }
]

resultados_teste = []


for arquivo, pesos_finais, biases_finais in zip(arquivos_teste, p, b):
    X_teste, Y_teste = load_data(arquivo['caminho'])
    # print(f' {arquivo["nome"]}\n X_teste = {X_teste}\n Y_teste = {Y_teste} ')

    if isinstance(pesos_finais, list):
        pesos_finais = [np.array(p) if not isinstance(p, np.ndarray) else p for p in pesos_finais]

    if isinstance(biases_finais, list):
        biases_finais = [np.array(b) if not isinstance(b, np.ndarray) else b for b in biases_finais]

    previsao = testar(X_teste, pesos_finais, biases_finais)

    erro = np.mean((previsao - Y_teste) ** 2)

    resultados_teste.append([arquivo['nome'], erro, previsao])

colunas_teste = ["Arquivo", "Erro MSE", "Previsões"]
print("Resultados do Teste:")
print(tabulate(resultados_teste, headers=colunas_teste, tablefmt="grid"))