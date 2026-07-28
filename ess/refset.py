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



def create_refset(population, obj_func, refset_size, refset1_size=None, rng=None):
    """
    Build a RefSet: RefSet1 by quality (best fitness) and RefSet2 at random.

    Fiel a ess_kernel.m (L655-677): la mitad "buena" del RefSet son los mejores por
    valor objetivo; el resto se toma aleatoriamente del pool restante (randperm), no
    por diversidad maximin. select_most_diverse se conserva para uso futuro en los
    filtros de busqueda local (opts.local.balance).

    population: np.ndarray (n, dim)
    obj_func: np.ndarray (n,)
    refset_size: int
    refset1_size: int or None (por defecto refset_size // 2)
    rng: np.random.Generator or None (si None se crea uno; pasar para reproducibilidad)
    Returns: dict with x, f, idx_r1, idx_r2, idx_all
    """
    population = np.asarray(population, dtype=float)
    obj_func = np.asarray(obj_func, dtype=float)

    if rng is None:
        rng = np.random.default_rng()
    if refset1_size is None:
        refset1_size = refset_size // 2

    idx_sorted = np.argsort(obj_func)
    idx_r1 = idx_sorted[:refset1_size]

    remaining_indices = idx_sorted[refset1_size:]
    n_r2 = refset_size - refset1_size
    perm = rng.permutation(len(remaining_indices))
    idx_r2 = remaining_indices[perm[:n_r2]]

    refset = np.vstack([population[idx_r1], population[idx_r2]])
    f_refset = np.concatenate([obj_func[idx_r1], obj_func[idx_r2]])
    idx_all = np.concatenate([idx_r1, idx_r2])

    return {
        "x": refset,
        "f": f_refset,
        "idx_r1": idx_r1,
        "idx_r2": idx_r2,
        "idx_all": idx_all,
    }


def update_refset(refset, cand_x, cand_f, parent_pos):
    """
    Greedy per-slot RefSet update, followed by a re-sort by objective value.

    Fiel a ess_kernel.m: cada candidato j apunta al miembro parent_pos[j] y solo lo
    reemplaza si cand_f[j] mejora su valor (L870-881). Si varios candidatos apuntan al
    mismo slot, gana el de menor valor. Al final el RefSet se reordena por f ascendente
    (mandatorio, L710-714: la combinacion por indices asume RefSet ordenado).

    refset: dict con 'x' (n, dim) y 'f' (n,)
    cand_x: np.ndarray (m, dim)
    cand_f: np.ndarray (m,)
    parent_pos: np.ndarray (m,) indice de slot objetivo por candidato
    Returns: (new_refset, info)
      new_refset: dict con 'x', 'f' ya reordenados
      info: dict con 'changed' (bool por miembro, alineado al RefSet reordenado)
            y 'n_changed' (int)
    """
    x = np.array(refset["x"], dtype=float, copy=True)
    f = np.array(refset["f"], dtype=float, copy=True)
    cand_x = np.asarray(cand_x, dtype=float)
    cand_f = np.asarray(cand_f, dtype=float)
    parent_pos = np.asarray(parent_pos, dtype=int)

    n = x.shape[0]
    changed = np.zeros(n, dtype=bool)

    # Mejor candidato que mejora cada slot
    best_f = f.copy()
    best_row = {}
    for j in range(len(parent_pos)):
        p = int(parent_pos[j])
        if cand_f[j] < best_f[p]:
            best_f[p] = cand_f[j]
            best_row[p] = j

    for p, j in best_row.items():
        x[p] = cand_x[j]
        f[p] = cand_f[j]
        changed[p] = True

    order = np.argsort(f)
    new_refset = {"x": x[order], "f": f[order]}
    info = {"changed": changed[order], "n_changed": int(changed.sum())}
    return new_refset, info
