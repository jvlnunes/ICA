import numpy as np
import matplotlib.pyplot as plt

def GA_1():
    
    print('=============================================================')
    print('Genetic algorithms: the fitness function of a single variable')
    print('=============================================================')
    
    # Define parameters
    obj_fun = "15*x - x*x"  # Objective function
    nind    = 6             # Population size
    ngenes  = 4             # Number of genes per chromosome
    pc      = 0.9           # Crossover probability 
    pm      = 0.001         # Mutation probability
    xmin    = 0             # Minimum value of x
    xmax    = 15            # Maximum value of x
    ngener  = 20            # Number of generations
    
    chrom = np.random.randint(2, size=(nind, ngenes))
    
    powers = 2 ** np.arange(ngenes-1, -1, -1)
    x = np.dot(chrom, powers)
    
    obj_v = eval_obj_fun(obj_fun, x)
    best  = [max(obj_v)]
    ave   = [np.mean(obj_v)]
    
    plt.figure()
    x_plot = np.linspace(xmin, xmax, 100)
    y_plot = 15*x_plot - x_plot*x_plot
    plt.plot(x_plot, y_plot, label=f'Objective function: {obj_fun}')
    plt.plot(x, obj_v, 'r.', markersize=15, label='Initial population')
    plt.legend()
    plt.show()
    
    for gen in range(ngener):
        fitness = obj_v.copy()
        if min(obj_v) < 0:
            fitness = fitness - min(obj_v)
            
        numsel = int(nind * 0.9)
        prob = fitness / np.sum(fitness)
        selected = np.random.choice(nind, numsel, p=prob)
        new_chrom = chrom[selected]
        
        for i in range(0, numsel-1, 2):
            if np.random.random() < pc:
                point = np.random.randint(1, ngenes-1)
                new_chrom[i:i+2, point:] = new_chrom[i:i+2, point:][::-1]
        
        mask = np.random.random((numsel, ngenes)) < pm
        new_chrom[mask] = 1 - new_chrom[mask]
        
        new_x = np.dot(new_chrom, powers)
        new_obj_v = eval_obj_fun(obj_fun, new_x)
        
        # Update population
        if nind > numsel:
            indices = np.argsort(fitness)[::-1][:nind-numsel]
            chrom = np.vstack((chrom[indices], new_chrom))
            x = np.concatenate((x[indices], new_x))
            obj_v = np.concatenate((obj_v[indices], new_obj_v))
        else:
            chrom = new_chrom
            x = new_x
            obj_v = new_obj_v
            
        best.append(max(obj_v))
        ave.append(np.mean(obj_v))
        
        # Plot current state
        plt.clf()
        plt.plot(x_plot, y_plot, label=f'Objective function: {obj_fun}')
        plt.plot(x, obj_v, 'r.', markersize=15, label=f'Generation {gen+1}')
        plt.legend()
        plt.pause(0.2)
    
    # Plot performance
    plt.figure()
    plt.plot(range(ngener+1), best, label='Best')
    plt.plot(range(ngener+1), ave, label='Average')
    plt.legend()
    plt.title(f'Pc = {pc}, Pm = {pm}')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()

def eval_obj_fun(obj_fun: str, x):
    return eval(obj_fun)

if __name__ == "__main__":
    GA_1()
