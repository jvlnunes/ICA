# Para rodar a o código sera necessário fazer a instalação das bibliotecas numpy e prettytable, sendo pretty table uma biblioteca que facilita a visualização de tabelas
import numpy as np 
from prettytable import PrettyTable

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    return x * (1 - x)

def erro_quadratico(y, y_pred):
    return np.mean((y - y_pred)**2)
    
def backpropagation_1_iteracao():
    
    x1, x2 =  0.4 ,  0.1
    y1, y2 =  0.5 ,  0.7
    w1, w2 = -0.25,  0.90
    w3, w4 =  0.46,  0.19
    w5, w6 = -0.68, -0.68
    w7, w8 = -0.88,  0.73
    b1, b2 =  0.20,  0.41  

    h1 = sigmoid(w1 * x1 + w2 * x2 + b1)
    h2 = sigmoid(w3 * x1 + w4 * x2 + b1)
    o1 = sigmoid(w5 * h1 + w6 * h2 + b2)
    o2 = sigmoid(w7 * h1 + w8 * h2 + b2)

    Eo1 = 0.5 * (y1 - o1)**2
    Eo2 = 0.5 * (y2 - o2)**2
    Etotal = Eo1 + Eo2

    delta_o1 = -(y1 - o1) * derivada_sigmoid(o1)
    delta_o2 = -(y2 - o2) * derivada_sigmoid(o2)

    w5_grad = delta_o1 * h1
    w6_grad = delta_o1 * h2
    w7_grad = delta_o2 * h1
    w8_grad = delta_o2 * h2

    b2_grad = delta_o1 + delta_o2

    delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * derivada_sigmoid(h1)
    delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * derivada_sigmoid(h2)

    w1_grad = delta_h1 * x1
    w2_grad = delta_h1 * x2
    w3_grad = delta_h2 * x1
    w4_grad = delta_h2 * x2

    b1_grad = delta_h1 + delta_h2

    learning_rate = 0.1

    w1 -= learning_rate * w1_grad
    w2 -= learning_rate * w2_grad
    w3 -= learning_rate * w3_grad
    w4 -= learning_rate * w4_grad
    w5 -= learning_rate * w5_grad
    w6 -= learning_rate * w6_grad
    w7 -= learning_rate * w7_grad
    w8 -= learning_rate * w8_grad
    b1 -= learning_rate * b1_grad
    b2 -= learning_rate * b2_grad

    print("Resultados da primeira iteração do Backpropagation:")
    print(f"Pesos atualizados: w1 = {w1:.4f}, w2 = {w2:.4f}, w3 = {w3:.4f}, w4 = {w4:.4f}, w5 = {w5:.4f}, w6 = {w6:.4f}, w7 = {w7:.4f}, w8 = {w8:.4f}")
    print(f"Bias atualizados:  b1 = {b1:.4f}, b2 = {b2:.4f}")
    print(f"Erro total: {Etotal:.4f}")

def run_backpropagation(learning_rate, tolerance):
    
    x1, x2 =  0.4 ,  0.1
    d1, d2 =  0.5 ,  0.7  
    w1, w2 = -0.25,  0.90
    w3, w4 =  0.46,  0.19
    w5, w6 = -0.68, -0.68
    w7, w8 = -0.88,  0.73
    b1, b2 =  0.20,  0.41  

    epocas = 0
    while True:
        epocas += 1
                
        h1 = sigmoid(w1 * x1 + w2 * x2 + b1)
        h2 = sigmoid(w3 * x1 + w4 * x2 + b1)
        
        o1 = sigmoid(w5 * h1 + w6 * h2 + b2)
        o2 = sigmoid(w7 * h1 + w8 * h2 + b2)
        
        Eo1 = 0.5 * (d1 - o1)**2
        Eo2 = 0.5 * (d2 - o2)**2
        Etotal = Eo1 + Eo2
        
        if Etotal < tolerance:
            break
        
        delta_o1 = -(d1 - o1) * derivada_sigmoid(o1)
        delta_o2 = -(d2 - o2) * derivada_sigmoid(o2)
        
        w5_grad = delta_o1 * h1
        w6_grad = delta_o1 * h2
        w7_grad = delta_o2 * h1
        w8_grad = delta_o2 * h2
        
        b2_grad = delta_o1 + delta_o2
        
        delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * derivada_sigmoid(h1)
        delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * derivada_sigmoid(h2)
        
        w1_grad = delta_h1 * x1
        w2_grad = delta_h1 * x2
        w3_grad = delta_h2 * x1
        w4_grad = delta_h2 * x2
        
        b1_grad = delta_h1 + delta_h2
        
        w1 -= learning_rate * w1_grad
        w2 -= learning_rate * w2_grad
        w3 -= learning_rate * w3_grad
        w4 -= learning_rate * w4_grad
        w5 -= learning_rate * w5_grad
        w6 -= learning_rate * w6_grad
        w7 -= learning_rate * w7_grad
        w8 -= learning_rate * w8_grad

        b1 -= learning_rate * b1_grad
        b2 -= learning_rate * b2_grad

    return epocas, Etotal, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2


def analise_backpropagation():
    taxa_de_aprendizagem = [0.01,   0.1,    0.5]
    tolerancia           = [0.01, 0.001, 0.0001]

    tabela = PrettyTable()
    tabela.field_names = ["Taxa de aprendizado", "Tolerância", "Épocas", "Erro total", 
                          "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "b1", "b2"]
    for lr in taxa_de_aprendizagem:
        for tol in tolerancia:
            epocas, Etotal, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2 = run_backpropagation(lr, tol)
            tabela.add_row([
                lr, tol, epocas, round(Etotal, 6),
                round(w1, 4), round(w2, 4), round(w3, 4), round(w4, 4),
                round(w5, 4), round(w6, 4), round(w7, 4), round(w8, 4),
                round(b1, 4), round(b2, 4)
            ])

    print(tabela)
analise_backpropagation()
