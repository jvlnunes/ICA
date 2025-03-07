from   functions_mlp     import *
import matplotlib.pyplot as plt

arquivos_treinamento = [
    {'nome': 'Seção5.8_RNA' , 'caminho': "3 AP/Dados/Treinamento/Tabela#Seção5.8_RNA.txt"  }
]

arquivos_teste = [
    {'nome': '5.3_RNA', 'caminho': "3 AP/Dados/Teste/Tabela#5.3_RNA.txt" }
]

print("Projeto prático 01")
try:
    file_path = arquivos_treinamento[0]['caminho'].replace('/', '\\')
    X, Y = load_data(file_path)
    print('SO identificado: Windows')
    
except Exception as e:
    file_path = arquivos_treinamento[0]['caminho']
    X, Y = load_data(file_path)
    print('SO identificado: Linux') 

    
resultados = []
treinos    = []

for i in range(5):
    try:
        neuronios = 5
        taxa_aprendizagem = 0.01
        precisao = 1e-6
        W1, B1, W2, B2, mse, epocas, W1_inicial, W2_inicial, erro_x_epocas = treinamento_pmc_1(X, Y, neuronios=neuronios, precisao=precisao, taxa_aprendizagem=taxa_aprendizagem)
        
        treinos.append({
            'i'            : i,
            'W1_inicial'   : W1_inicial.tolist(),
            'W2_inicial'   : W2_inicial.tolist(),
            'W1'           : W1.tolist(),
            'W2'           : W2.tolist(),
            'B1'           : B1.tolist(),
            'B2'           : B2.tolist(),
            'mse'          : mse, 
            'epocas'       : epocas,
            'erro_x_epocas': erro_x_epocas
        })
        
        resultados.append([
            f"T{i+1}",
            mse,
            epocas
        ])
        
    except Exception as e:
        print(e)

colunas = ["Treino", "Erro quadrático médio", "Epocas"]
print('\nTabela de resultados do Treinamento:')
print(tabulate(resultados, headers=colunas, tablefmt="pretty"))
    
maiores_epocas = sorted(treinos, key=lambda x: x['epocas'], reverse=True)[:2]

# Análise dos treinamentos com maior quantidade de épocas
for treino in maiores_epocas:
    # print(f'''
    # Treinamento {treino['i'] + 1}
    #     Épocas: {treino['epocas']}
    #     Pesos Iniciais:
    #         W1 = {treino['W1_inicial']}
    #         W2 = {treino['W2_inicial']}
    
    #     Pesos Finais:
    #         W1 = {treino['W1']}
    #         W2 = {treino['W2']}
    #                             ''')
    
    epocas = [x['epocas'] for x in treino['erro_x_epocas']]
    erros  = [x['erro'  ] for x in treino['erro_x_epocas']]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, erros, label='Erro Quadrático médio')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.title(f'Evolução do Erro Quadrático em T{treino["i"] + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
try:
    X_teste, Y_teste = load_data(arquivos_teste[0]['caminho'].replace('/', '\\'))
    resultados, erros_medios, variancias = teste_pmc(X_teste, Y_teste, treinos)
    
except:
    X_teste, Y_teste = load_data(arquivos_teste[0]['caminho'])
    resultados, erros_medios, variancias, maes, mses = teste_pmc(X_teste, Y_teste, treinos)

print("\nTabela de resultados dos testes:")

# Imprimir os resultados
imprimir_resultados(resultados, erros_medios, variancias, maes, mses)
