import numpy as np
def ex1(x, return_grad: bool = False): #DW
    # f: objective function value; 
    # c: equality constraints; 
    # gf: gradient of objective function; gc: gradient of constraints (not yet implemented)

    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]

    f=4*x1**2-2.1*x1**4+(1/3)*x1**6+x1*x2-4*x2**2+4*x2**4
    
    c=np.array([])
    
    if return_grad:
        gf=np.array([8*x1-8.4*x1**3+2*x1**5+x2,
                    x1-8*x2+16*x2**3])
        gc=np.empty((0, 2))

        return f, c, gf, gc
    
    else:
        return f, c

def old_ex1(x):

    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    f=4*x1**2-2.1*x1**4+1/3*x1**6+x1*x2-4*x2**2+4*x2**4

    return f