import numpy as np

from .utils import project_bounds


def _midpoint(a, b):
    return 0.5 * (a + b)


def _remove_rows_equal_to_any(points, reference_rows):
    if points.size == 0:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if np.any(np.all(np.isclose(reference_rows, p), axis=1)):
            keep[i] = False
    return points[keep]


def build_combination_pairs(refset, rng=None, max_pairs=None, strategy="structured"):
    """
    Build parent pairs for combination using RefSet structure.

    Inputs:
    - refset['x'], refset['f'] are required
    - refset['idx_r1'], refset['idx_r2'] are used when present
    Returns:
    - dict with keys: pairs, r1_pos, r2_pos, source
    """
    if strategy != "structured":
        raise ValueError("Only strategy='structured' is supported in AI-CHANGE-003.")

    x = np.asarray(refset["x"], dtype=float)
    f = np.asarray(refset["f"], dtype=float)
    n = x.shape[0]
    if n < 2:
        return {
            "pairs": [],
            "r1_pos": np.array([], dtype=int),
            "r2_pos": np.array([], dtype=int),
            "source": "none",
        }

    source = "fitness_fallback"
    if "idx_r1" in refset and "idx_r2" in refset:
        idx_r1 = np.asarray(refset["idx_r1"], dtype=int)
        idx_r2 = np.asarray(refset["idx_r2"], dtype=int)
        idx_all = np.asarray(refset.get("idx_all", np.concatenate([idx_r1, idx_r2])), dtype=int)
        pos_map = {int(v): i for i, v in enumerate(idx_all)}
        if all(int(v) in pos_map for v in idx_r1) and all(int(v) in pos_map for v in idx_r2):
            r1_pos = np.array([pos_map[int(v)] for v in idx_r1], dtype=int)
            r2_pos = np.array([pos_map[int(v)] for v in idx_r2], dtype=int)
            source = "idx_fields"
        else:
            sorted_pos = np.argsort(f)
            split = n // 2
            r1_pos = sorted_pos[:split]
            r2_pos = sorted_pos[split:]
    else:
        sorted_pos = np.argsort(f)
        split = n // 2
        r1_pos = sorted_pos[:split]
        r2_pos = sorted_pos[split:]

    pairs = []

    for i in r1_pos:
        for j in r2_pos:
            pairs.append((int(i), int(j), "mixed"))
    for a in range(len(r1_pos)):
        for b in range(a + 1, len(r1_pos)):
            pairs.append((int(r1_pos[a]), int(r1_pos[b]), "r1_r1"))
    for a in range(len(r2_pos)):
        for b in range(a + 1, len(r2_pos)):
            pairs.append((int(r2_pos[a]), int(r2_pos[b]), "r2_r2"))

    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]

    return {"pairs": pairs, "r1_pos": r1_pos, "r2_pos": r2_pos, "source": source}


def combine_pair_core(x1, x2, x_L, x_U, pair_type, f1=None, f2=None, include_parents=False):
    """
    Deterministic core inspired by MATLAB ssm_combination geometry.

    Returns only derived candidate points by default (parents excluded).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x_L = np.asarray(x_L, dtype=float)
    x_U = np.asarray(x_U, dtype=float)

    d = 0.5 * (x2 - x1)
    v1 = project_bounds(x1 - d, x_L, x_U)
    v2 = project_bounds(x1, x_L, x_U)
    v3 = project_bounds(x1 + d, x_L, x_U)
    v4 = project_bounds(x2, x_L, x_U)
    v5 = project_bounds(x2 + d, x_L, x_U)

    if pair_type == "r1_r1":
        candidates = np.array(
            [
                _midpoint(v1, v2),
                _midpoint(v2, v3),
                _midpoint(v3, v4),
                _midpoint(v4, v5),
            ],
            dtype=float,
        )
    elif pair_type == "r2_r2":
        if f1 is None or f2 is None:
            side = "left"
        else:
            side = "left" if f1 <= f2 else "right"
        second = _midpoint(v1, v2) if side == "left" else _midpoint(v4, v5)
        candidates = np.array([_midpoint(v2, v3), second], dtype=float)
    elif pair_type == "mixed":
        candidates = np.array([_midpoint(v1, v2), _midpoint(v2, v3), _midpoint(v4, v5)], dtype=float)
    else:
        raise ValueError("pair_type must be one of: 'r1_r1', 'r2_r2', 'mixed'.")

    candidates = np.array([project_bounds(c, x_L, x_U) for c in candidates], dtype=float)

    if not include_parents:
        candidates = _remove_rows_equal_to_any(candidates, np.vstack([x1, x2]))

    if candidates.size == 0:
        return np.empty((0, x1.shape[0]), dtype=float)

    return np.unique(candidates, axis=0)


def generate_candidates_from_refset(
    refset,
    x_L,
    x_U,
    rng=None,
    max_pairs=None,
    strategy="structured",
    deduplicate=True,
    return_parents=False,
):
    """
    Generate deterministic, bounded, unevaluated candidates from RefSet.
    Does not mutate refset and does not evaluate objective functions.

    return_parents=False (default): comportamiento previo. Devuelve solo los candidatos,
    deduplicados entre pares. Uso no evolutivo.

    return_parents=True: devuelve (candidates, parent_pos) SIN deduplicar entre pares,
    preservando el vinculo hijo->padre 1:1 que necesita update_refset. Cada candidato se
    etiqueta con el slot del PEOR padre del par (mayor f); asi el update greedy solo
    reemplaza ese miembro si el candidato lo mejora, manteniendo monotonia.
    (Adaptacion Python: el mapeo fiel v1..v5->padre de ess_kernel.m se recuperara en 004C.)
    """
    x_ref = np.asarray(refset["x"], dtype=float)
    f_ref = np.asarray(refset["f"], dtype=float)

    pairing = build_combination_pairs(refset, rng=rng, max_pairs=max_pairs, strategy=strategy)

    all_candidates = []
    parent_pos = []
    for i, j, pair_type in pairing["pairs"]:
        pair_candidates = combine_pair_core(
            x_ref[i],
            x_ref[j],
            x_L=x_L,
            x_U=x_U,
            pair_type=pair_type,
            f1=f_ref[i],
            f2=f_ref[j],
            include_parents=False,
        )
        if pair_candidates.size > 0:
            all_candidates.append(pair_candidates)
            worse = i if f_ref[i] >= f_ref[j] else j
            parent_pos.extend([worse] * pair_candidates.shape[0])

    if not all_candidates:
        empty = np.empty((0, x_ref.shape[1]), dtype=float)
        if return_parents:
            return empty, np.empty((0,), dtype=int)
        return empty

    candidates = np.vstack(all_candidates)

    if not return_parents:
        candidates = _remove_rows_equal_to_any(candidates, x_ref)
        if deduplicate and candidates.size > 0:
            candidates = np.unique(candidates, axis=0)
        return candidates

    parent_pos = np.asarray(parent_pos, dtype=int)
    keep = np.ones(candidates.shape[0], dtype=bool)
    for k, p in enumerate(candidates):
        if np.any(np.all(np.isclose(x_ref, p), axis=1)):
            keep[k] = False
    return candidates[keep], parent_pos[keep]
