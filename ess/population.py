import numpy as np

def generate_initial_population(n_points, x_L, x_U):
        # n_points: numero de soluciones a generar
        # x_L, x_U: valores de limites (lower and upper) por variable
    dim = len(x_L)
    population = np.zeros((n_points, dim))
    for i in range(dim):
        population[:, i] = np.random.uniform(x_L[i], x_U[i], size=n_points)
        # generamos 10 num aleatorios por cada columna = para todas las filas, en la columna i
    return population
