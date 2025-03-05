import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def GA_2():
    print('=========================================================')
    print('Genetic algorithms: the fitness function of two variables')
    print('=========================================================')
    
    # Define parameters
    obj_fun = "(1-x)**2 * np.exp(-x**2-(y+1)**2) - (x-x**3-y**5) * np.exp(-x**2-y**2)"
    nind = 6      # Population size
    nvar = 2      # Number of variables
    ngenes = 16   # Number of genes in chromosome
    pc = 0.9      # Crossover probability
    pm = 0.005    # Mutation probability
    xymin = -3    # Minimum values for x and y
    xymax = 3     # Maximum values for x and y
    ngener = 100  # Number of generations
    
    # Generate initial population
    chrom = np.random.randint(2, size=(nind, ngenes))
    
    # Decode chromosomes
    lvar = ngenes // nvar
    xy = np.zeros((nind, nvar))
    powers = 2 ** np.arange(lvar-1, -1, -1)
    
    for ind in range(nvar):
        start = lvar * ind
        end = lvar * (ind + 1)
        xy[:, ind] = np.dot(chrom[:, start:end], powers)
        xy[:, ind] = xymin + (xymax - xymin) * (xy[:, ind] + 1) / (2**lvar + 1)
    
    # Calculate initial fitness
    obj_v = eval_obj_fun(obj_fun, xy[:, 0], xy[:, 1])
    best = [max(obj_v)]
    ave = [np.mean(obj_v)]
    
    # Create mesh for surface plot
    x, y = np.meshgrid(np.arange(xymin, xymax, 0.25), np.arange(xymin, xymax, 0.25))
    z = eval_obj_fun(obj_fun, x, y) + 4
    
    # Main GA loop
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for gen in range(ngener):
        # Fitness evaluation
        fitness = obj_v.copy()
        if min(obj_v) < 0:
            fitness = fitness - min(obj_v)
            
        # Selection
        numsel = int(nind * 0.8)
        prob = fitness / np.sum(fitness)
        selected = np.random.choice(nind, numsel, p=prob)
        new_chrom = chrom[selected]
        
        # Crossover
        for i in range(0, numsel-1, 2):
            if np.random.random() < pc:
                point = np.random.randint(1, ngenes-1)
                new_chrom[i:i+2, point:] = new_chrom[i:i+2, point:][::-1]
        
        # Mutation
        mask = np.random.random((numsel, ngenes)) < pm
        new_chrom[mask] = 1 - new_chrom[mask]
        
        # Decode new chromosomes
        new_xy = np.zeros((numsel, nvar))
        for ind in range(nvar):
            start = lvar * ind
            end = lvar * (ind + 1)
            new_xy[:, ind] = np.dot(new_chrom[:, start:end], powers)
            new_xy[:, ind] = xymin + (xymax - xymin) * (new_xy[:, ind] + 1) / (2**lvar + 1)
        
        new_obj_v = eval_obj_fun(obj_fun, new_xy[:, 0], new_xy[:, 1])
        
        # Update population
        if nind > numsel:
            indices = np.argsort(fitness)[::-1][:nind-numsel]
            chrom = np.vstack((chrom[indices], new_chrom))
            xy = np.vstack((xy[indices], new_xy))
            obj_v = np.concatenate((obj_v[indices], new_obj_v))
        else:
            chrom = new_chrom
            xy = new_xy
            obj_v = new_obj_v
            
        best.append(max(obj_v))
        ave.append(np.mean(obj_v))
        
        # Plot current state
        ax.clear()
        ax.plot_surface(x, y, z, alpha=0.5)
        ax.contour(x, y, z, levels=20, offset=0)
        ax.scatter(xy[:, 0], xy[:, 1], obj_v + 4.08, c='r', s=40)
        ax.set_title(f'Generation {gen+1}')
        ax.set_xlabel('Parameter x')
        ax.set_ylabel('Parameter y')
        ax.set_zlabel('Fitness')
        ax.set_xlim(xymin, xymax)
        ax.set_ylim(xymin, xymax)
        ax.set_zlim(0, 6)
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

def eval_obj_fun(obj_fun: str, x, y):
    return eval(obj_fun)

if __name__ == "__main__":
    GA_2()
