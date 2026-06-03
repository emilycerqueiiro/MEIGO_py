import numpy as np
from ess.population import generate_diverse_population
from ess.refset import create_refset
from problems.objective_functions import rastrigin


def run_ess(obj_func, x_L, x_U, n_initial=20, refset_size=10, max_iter=50):
    population = generate_diverse_population(n_initial, x_L, x_U)
    f_pop = np.array([obj_func(x) for x in population])
    refset = create_refset(population, f_pop, refset_size)

    print("[run_ess] Final RefSet ready:")
    print(refset)
    return refset


if __name__ == "__main__":
    x_L = np.array([-5.0, -5.0])
    x_U = np.array([5.0, 5.0])
    run_ess(rastrigin, x_L, x_U)
