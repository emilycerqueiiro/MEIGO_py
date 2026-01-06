import numpy as np

def rosen10(x, return_grad: bool = False):

    x = np.array(x, dtype = float)
    f = 0
    n = 10

    for i in range(n-1):
        f += 100*(x[i]**2 - x[i+1])**2 + (x[i]-1)**2
    
    g = []

    if return_grad:
        return f, g

    else:
        return f
    

def sphere(x, return_grad: bool = False):

    x = np.array(x, dtype = float)
    f = np.dot(x, x)

    if return_grad:
        g = 2 * x
        return f, g

    else:
        return f
    
def rosenbrock(x, return_grad: bool = False):
    x = np.array(x, dtype=float)
    f = 0
    for i in range(len(x) - 1):
        f += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    if return_grad:
        g = np.zeros_like(x)
        for i in range(len(x) - 1):
            g[i] += -400 * (x[i+1] - x[i]**2) * x[i] + 2 * (x[i] - 1)
            g[i+1] += 200 * (x[i+1] - x[i]**2)
        return f, g
    else:
        return f
    
def rastrigin(x):
    A = 10
    x = np.array(x)
    n = len(x)

    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))