import numpy as np
from problems.objective_functions import sphere, rosenbrock
from .utils import evaluate, project_bounds

def generate_initial_population(rng, pop_size, x_L, x_U):
    """
    Genera población inicial uniforme dentro de bounds.
    rng: np.random.Generator
    pop_size: int
    x_L, x_U: np.ndarray
    Returns: np.ndarray (pop_size, dim)
    """
    x_L = np.array(x_L)
    x_U = np.array(x_U)
    dim = len(x_L)
    population = rng.uniform(x_L, x_U, size=(pop_size, dim))
    return population

def create_refset(population, f_pop, refset_size, refset1_size=None):
    """
    Crea RefSet con calidad y diversidad.
    population: np.ndarray (n, dim)
    f_pop: np.ndarray (n,)
    refset_size: int, par
    refset1_size: int or None
    Returns: dict ResultsRefset {'x': (refset_size, dim), 'f': (refset_size,), 'idx_r1': array, 'idx_r2': array, 'idx_all': array}
    """
    if refset1_size is None:
        refset1_size = refset_size // 2

    idx_sorted = np.argsort(f_pop)
    idx_r1 = idx_sorted[:refset1_size]
    best_half = population[idx_r1]

    remaining = population[idx_sorted[refset1_size:]]
    remaining_indices = idx_sorted[refset1_size:]
    diverse_half, idx_r2 = select_most_diverse(remaining, best_half, refset_size - refset1_size, remaining_indices)

    refset = np.vstack([best_half, diverse_half])
    f_refset = np.concatenate([f_pop[idx_r1], f_pop[idx_r2]])
    idx_all = np.concatenate([idx_r1, idx_r2])

    return {
        'x': refset,
        'f': f_refset,
        'idx_r1': idx_r1,
        'idx_r2': idx_r2,
        'idx_all': idx_all
    }


def select_most_diverse(candidates, reference, k, candidate_indices):
    """
    Selecciona k soluciones de 'candidates' que estén más alejadas de 'reference' usando maximin iterativo con S acumulado.
    candidates: np.ndarray (m, dim)
    reference: np.ndarray (r, dim), inicial RefSet1
    k: int
    candidate_indices: np.ndarray (m,), índices originales en population
    Returns: (diverse: np.ndarray (k, dim), indices: np.ndarray (k,))
    """
    diverse = []
    indices = []
    candidates = np.array(candidates)
    candidate_indices = np.array(candidate_indices)
    
    while len(candidates) > 0 and len(diverse) < k:
        # Calcular distancia mínima a cualquier solución en el RefSet actual (S = reference + diverse)
        current_ref = np.array(reference) if len(diverse) == 0 else np.vstack([reference, diverse])
        dists = [np.min([np.linalg.norm(c - r) for r in current_ref]) for c in candidates]
        idx = np.argmax(dists)
        diverse.append(candidates[idx])
        indices.append(candidate_indices[idx])
        # Eliminar del pool
        candidates = np.delete(candidates, idx, axis=0)
        candidate_indices = np.delete(candidate_indices, idx)
    
    return np.array(diverse), np.array(indices)

# ejemplo
# candidates = [np.array([0, 0]), np.array([3, 4]), np.array([5, 5])]
# reference = [np.array([1, 1]), np.array([4, 4])]
# diverse_half = select_most_diverse(candidates, reference, 10 // 2)

def ess_kernel_min(problem, opts):
    """
    minimun serial eSS: muestreo uniforme, evaluación, selección and basic tracking.
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
        # Evaluate
        f_val = evaluate(f, x)
        numeval += 1
        # Tracking
        fbest_trace.append(f_val)
        # Selection
        if f_val < fbest:
            fbest = f_val
            xbest = x.copy()
    
    return {
        'xbest': xbest,
        'fbest': fbest,
        'numeval': numeval,
        'fbest_trace': fbest_trace
    }