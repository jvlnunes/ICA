from functions import *

arquivos = {
    'nome': 'Tabela 2/1',
    'caminho': '4 AP/Dados/Tabela#Seção6.6_RNA.txt'
}

X, Y = load_data(arquivos['caminho'])

print(f"X: \n{X}")
print(f"Y: \n{Y}")