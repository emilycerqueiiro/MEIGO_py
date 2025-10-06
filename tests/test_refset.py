import numpy as np
from ess.main import create_refset, generate_initial_population
from problems.objective_functions import rastrigin

x_L = np.array([-5.0, -5.0])
x_U = np.array([5.0, 5.0])
population = generate_initial_population(10, x_L, x_U)
refset = create_refset(population, rastrigin)
print(refset)


