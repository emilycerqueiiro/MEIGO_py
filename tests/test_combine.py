import numpy as np
from ess.combination import go_beyond

z1 = np.array([1.0, 2.0])
z2 = np.array([3.0, 3.5])
x_L = np.array([-5.0, -5.0])
x_U = np.array([5.0, 5.0])

# Prueba de go_beyond():

points = go_beyond(z1, z2, x_L, x_U, prob_bound=0.9, denom=1.0, nrand=5)

for p in points:
    print(p)

