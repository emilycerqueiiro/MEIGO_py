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
    
def rastrigin(x):
    A = 10
    x = np.array(x)
    n = len(x)

    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))