import numpy as np


def select_most_diverse(candidates, reference, k, candidate_indices):
    """
    Select k candidate solutions using iterative maximin diversity.

    candidates: np.ndarray (m, dim)
    reference: np.ndarray (r, dim), initial RefSet1
    k: int
    candidate_indices: np.ndarray (m,), original indices in population
    Returns: (diverse: np.ndarray (k, dim), indices: np.ndarray (k,))
    """
    diverse = []
    indices = []
    candidates = np.asarray(candidates, dtype=float)
    candidate_indices = np.asarray(candidate_indices)
    reference = np.asarray(reference, dtype=float)

    while len(candidates) > 0 and len(diverse) < k:
        current_ref = reference if len(diverse) == 0 else np.vstack([reference, diverse])
        dists = [np.min([np.linalg.norm(c - r) for r in current_ref]) for c in candidates]
        idx = int(np.argmax(dists))
        diverse.append(candidates[idx])
        indices.append(candidate_indices[idx])
        candidates = np.delete(candidates, idx, axis=0)
        candidate_indices = np.delete(candidate_indices, idx)

    return np.asarray(diverse, dtype=float), np.asarray(indices)



def create_refset(population, obj_func, refset_size, refset1_size=None):
    """
    Build a RefSet using quality for RefSet1 and maximin diversity for RefSet2.

    population: np.ndarray (n, dim)
    obj_func: np.ndarray (n,)
    refset_size: int
    refset1_size: int or None
    Returns: dict with x, f, idx_r1, idx_r2, idx_all
    """
    population = np.asarray(population, dtype=float)
    obj_func = np.asarray(obj_func, dtype=float)

    if refset1_size is None:
        refset1_size = refset_size // 2

    idx_sorted = np.argsort(obj_func)
    idx_r1 = idx_sorted[:refset1_size]
    best_half = population[idx_r1]

    remaining = population[idx_sorted[refset1_size:]]
    remaining_indices = idx_sorted[refset1_size:]
    diverse_half, idx_r2 = select_most_diverse(
        remaining,
        best_half,
        refset_size - refset1_size,
        remaining_indices,
    )

    refset = np.vstack([best_half, diverse_half])
    f_refset = np.concatenate([obj_func[idx_r1], obj_func[idx_r2]])
    idx_all = np.concatenate([idx_r1, idx_r2])

    return {
        "x": refset,
        "f": f_refset,
        "idx_r1": idx_r1,
        "idx_r2": idx_r2,
        "idx_all": idx_all,
    }
