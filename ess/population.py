import numpy as np

def generate_initial_population(n_points, x_L, x_U):
        # n_points: numero de soluciones a generar
        # x_L, x_U: valores de limites (lower and uprobprober) por variable
    dim = len(x_L)
    freq = np.ones((n_points, 4))
    prob = np.zeros((4, 1))
    
    population = np.zeros((n_points, dim))

    for i in range(4):
        population[i, :] = (np.random.rand(1, dim) + i) / 4


    for i in range(4, n_points):
        for j in range(n_points):
            prob[k] = (1 / (freq[j, :])) / np.sum(1/freq[j, :])

            a = np.random.rand()
        
            for k in range(4):
                if a <= np.sum(prob[:k+1]):
                    solutions[i, j] = (np.random.rand() + k) / 4
                    freq[j, k] += 1
                    break
    
    a = np.array(x_U) - np.array(x_L)
    b = np.array(x_L)
    solutions = solutions * a + b

    print(f"[generate_initial_population] Generated {n_points} solutions of dimension {dim}")
    print(population)
    
    return population, solutions


def generate_diverse_population(n_points, x_L, x_U, seed=None):
    """
    Igual funcionalidad que ssm_diverse.k, pero vectorizado.
    Devuelve matriz (n_points, dim) dentro de [x_L, x_U].
    """
    rng = np.random.default_rng(seed)
    dim = len(x_L)

    # Frecuencias por variable (dim x 4): empezamos en 1 probara evitar division by zero
    freq = np.ones((dim, 4), dtype=float)

    # Soluciones normalizadas [0,1]
    S = np.empty((n_points, dim), dtype=float)

    # (1) 4 semillas: cubetas 0..3
    S[:4, :] = (rng.random((4, dim)) + np.arange(4)[:, None]) / 4.0

    # (2) Resto: muestreo adaptativo por cubeta, vectorizado por variable
    for i in range(4, n_points):
        inv = 1.0 / freq
        prob = inv / inv.sum(axis=1, keepdims=True)   # probabilidades por variable
        cdf = prob.cumsum(axis=1)                     # CDF por variable (dim x 4)
        u = rng.random(dim)                        # un u∈[0,1) por variable
        k = (u[:, None] >= cdf).sum(axis=1)  # índice de cubeta 0..3 por variable

        # valor dentro de la cubeta elegida
        S[i, :] = (rng.random(dim) + k) / 4.0

        # actualizar la frecuencia de esa cubeta
        freq[np.arange(dim), k] += 1

    print(f"[generate_diverse_population] Calculated Prob {prob}")
    print(f"[generate_diverse_population] Calculated freq {freq}")

    # (3) Escalado a [x_L, x_U]
    a = np.asarray(x_U, float) - np.asarray(x_L, float)
    b = np.asarray(x_L, float)
    population = S * a + b
    

    print(f"[generate_diverse_population] Generated {n_points} solutions of dimension {dim}")
    print(population)
    
    return population


if __name__ == "__main__":
    x_L = np.array([-5.0, -5.0])
    x_U = np.array([5.0, 5.0])
    generate_diverse_population(10, x_L, x_U)