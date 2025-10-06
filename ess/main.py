import numpy as np

def create_refset(population, obj_func, refset_size=10):

    # population: matriz [n_solutions, dim] con las soluciones iniciales.
    # obj_func: función que toma una solución y devuelve su valor.
    # refset_size: número total de soluciones en el RefSet (par).
    # refset: matriz [refset_size, dim]

    n_solutions = population.shape[0] # n soluciones = n filas de population

    scores = np.array([obj_func(i) for i in population]) # matriz de soluciones sustituidas en obj_func

    idx_sorted = np.argsort(scores) # devuelve los indices ordenados de menor a mayor
    best_half = population[idx_sorted[:refset_size // 2]] # primera mitad de population

    remaining = population[idx_sorted[refset_size // 2:]] # segunda mitad de population
    diverse_half = select_most_diverse(remaining, best_half, refset_size // 2)

    refset = np.vstack([best_half, diverse_half]) 
    return refset


def select_most_diverse(candidates, reference, k):
    """
    Selecciona k soluciones de 'candidates' que estén más alejadas de 'reference':'best references'.
    """
    diverse = []
    for _ in range(k):
        # Calcular distancia mínima a cualquier solución en el RefSet actual
        dists = [np.min([np.linalg.norm(c - r) for r in reference]) for c in candidates]
        # calcula la distancia euclidea para saber cual está mas alejado
        idx = np.argmax(dists)
        diverse.append(candidates[idx])
        # Añadir al RefSet y eliminar del pool
        reference = np.vstack([reference, candidates[idx]])
        candidates = np.delete(candidates, idx, axis=0)
    return np.array(diverse)

# ejemplo
candidates = [np.array([0, 0]), np.array([3, 4]), np.array([5, 5])]
reference = [np.array([1, 1]), np.array([4, 4])]
diverse_half = select_most_diverse(candidates, reference, 10 // 2)