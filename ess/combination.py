import numpy as np

def weighted_combination(x1, x2, w=None):

    # x1: array-like, primer padre.
    # x2: array-like, segundo padre.
    # w: peso opcional. If None = [0.25, 0.75].
    # offspring: nueva solución intermedia combinada dando pesos.
    
    if w is None:
        w = np.random.uniform(0.25, 0.75)  # evita extremos

    x1, x2 = np.array(x1), np.array(x2)
    offspring = w * x1 + (1 - w) * x2
    return offspring

import numpy as np

def go_beyond(z1, z2, x_L, x_U, prob_bound, denom, nrand):
    """
    Genera puntos más allá de z2 en dirección z2 - z1, corrige límites probabilísticamente,
    y devuelve una lista de soluciones candidatas.

    Parámetros:
        z1, z2: soluciones base (np.array)
        x_L, x_U: límites inferiores y superiores (np.array)
        prob_bound: probabilidad de forzar al límite
        denom: factor divisor de la dirección (reduce la agresividad)
        nrand: número de soluciones nuevas a generar

    Retorna:
        Lista de np.array con nuevas soluciones candidatas
    """
    d = (z2 - z1) / denom
    zv2 = z2 + d

    # Corrección probabilística de límites
    mask_low = zv2 < x_L
    mask_high = zv2 > x_U

    for i in np.where(mask_low)[0]:
        if np.random.rand() > prob_bound:
            zv2[i] = x_L[i]

    for i in np.where(mask_high)[0]:
        if np.random.rand() > prob_bound:
            zv2[i] = x_U[i]

    # Muestreo aleatorio entre z2 y zv2
    points = [
        z2 + (zv2 - z2) * np.random.rand()
        for _ in range(nrand)
    ]

    return points

