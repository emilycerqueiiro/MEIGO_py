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

    print(f"[create_refset] Best half selected ({best_half})")

    remaining = population[idx_sorted[refset_size // 2:]] # segunda mitad de population
    diverse_half = select_most_diverse(remaining, best_half, refset_size // 2)

    print(f"[create_refset] Remaining for diversity ({remaining})")

    refset = np.vstack([best_half, diverse_half])

    print(f"[create_refset] Final RefSet:")
    print(refset)

    return refset


def select_most_diverse(candidates, reference, k):
    """
    Selecciona k soluciones de 'candidates' que estén más alejadas de 'reference':'best references'.
    """
    diverse = []
    candidates = np.array(candidates)
    
    print(f"[select_most_diverse] Selecting {k} diverse solutions from {len(candidates)} candidates")

    while len(candidates) > 0 and len(diverse) < k:
        # Calcular distancia mínima a cualquier solución en el RefSet actual
        dists = [np.min([np.linalg.norm(c - r) for r in reference]) for c in candidates]
        # calcula la distancia euclidea para saber cual está mas alejado
        idx = np.argmax(dists)
        diverse.append(candidates[idx])
        # Añadir al RefSet y eliminar del pool
        reference = np.vstack([reference, candidates[idx]])
        candidates = np.delete(candidates, idx, axis=0)

    print(f"[select_most_diverse] Diverse solutions selected: {np.array(diverse)}")
   
    return np.array(diverse)

# ejemplo
# candidates = [np.array([0, 0]), np.array([3, 4]), np.array([5, 5])]
# reference = [np.array([1, 1]), np.array([4, 4])]
# diverse_half = select_most_diverse(candidates, reference, 10 // 2)

import numpy as np
from problems.objective_functions import sphere, rosenbrock
from .utils import evaluate, project_bounds

def ess_kernel_min(problem, opts):
    """
    eSS serial mínimo: muestreo uniforme, evaluación, selección y tracking básico.
    problem: dict con 'f' (callable), 'x_L', 'x_U' (np.ndarray).
    opts: dict con 'maxeval' (int), 'seed' (int, opcional).
    Returns: dict Results con 'xbest', 'fbest', 'numeval', 'fbest_trace'.
    """
    rng = np.random.default_rng(opts.get('seed', 42))
    f = problem['f']
    x_L = np.array(problem['x_L'])
    x_U = np.array(problem['x_U'])
    n_var = len(x_L)
    maxeval = opts['maxeval']
    
    numeval = 0
    fbest = np.inf
    xbest = None
    fbest_trace = []
    
    while numeval < maxeval:
        # Muestreo uniforme dentro bounds
        x = rng.uniform(x_L, x_U, size=n_var)
        # Project bounds (aunque uniforme ya está dentro, por consistencia)
        x = project_bounds(x, x_L, x_U)
        # Evaluar
        f_val = evaluate(f, x)
        numeval += 1
        # Tracking
        fbest_trace.append(f_val)
        # Selección
        if f_val < fbest:
            fbest = f_val
            xbest = x.copy()
    
    return {
        'xbest': xbest,
        'fbest': fbest,
        'numeval': numeval,
        'fbest_trace': fbest_trace
    }