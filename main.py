import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    return x * (1 - x)

def erro_quadratico(y, y_pred):
    return np.mean((y - y_pred)**2)

def main():
    
    x = np.array([[0.4,0.1]])
    y = np.array([[0.5],[0.7]])
    
    num_entradas  = 2
    num_neuronios = 2
    num_saida     = 2  
    
    np.random.seed(42)
    w1 = np.random.rand(num_entradas, num_neuronios)
    w2 = np.random.rand(num_neuronios, num_saida)
    b1 = np.zeros((1, num_neuronios))
    b2 = np.zeros((1, num_saida))
    
    taxa_aprendizagem = 0.1
    
    epocas = 1
    
    hi = np.dot(x, w1) + b1
    ho = sigmoid(hi)
    
    oi = np.dot(ho, w2) + b2
    saida = sigmoid(oi)
    
    
    error = y - saida 
    print(error)
    # for i in range(epocas):
        
    
main()