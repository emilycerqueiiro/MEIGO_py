import numpy as np
from problems.objective_functions import rosen10, sphere, rastrigin

def test_rosen10_minimum():
    x_test = np.ones(10)
    f = rosen10(x_test)
    print(f)
    assert abs(f - 0) < 1e-8, f


def test_rastrigin():
    x_test = np.zeros(10)
    f = rastrigin(x_test)
    assert abs(f) < 1e-8, f
    