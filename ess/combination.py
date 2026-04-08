import numpy as np

from .utils import project_bounds


def _as_rng(rng=None):
    return rng if rng is not None else np.random.default_rng()


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


def _random_segment_point(a, b, rng):
    return a + (b - a) * rng.random(a.shape[0])


def _parse_eval_output(result):
    """
    Normalize eval callback output to (include, val_penalty, payload).
    """
    if isinstance(result, dict):
        include = bool(result.get("include", True))
        if "val_penalty" in result:
            val_penalty = float(result["val_penalty"])
        elif "value" in result:
            val_penalty = float(result["value"])
        elif "f" in result:
            val_penalty = float(result["f"])
        else:
            val_penalty = np.inf
        return include, val_penalty, result

    if isinstance(result, (tuple, list)):
        if len(result) >= 2 and isinstance(result[0], (bool, np.bool_)):
            return bool(result[0]), float(result[1]), {"raw": result}
        if len(result) >= 1:
            return True, float(result[0]), {"raw": result}

    return True, float(result), {"raw": result}


def _probabilistic_bound_correction(x, x_L, x_U, rng, prob_bound):
    """
    MATLAB-like probabilistic correction used in ssm_combination/ssm_beyond.
    One random draw per low/high side per vector.
    """
    x = np.asarray(x, dtype=float).copy()
    low_mask = x < x_L
    high_mask = x > x_U

    if np.any(low_mask):
        if rng.random() > prob_bound:
            x[low_mask] = x_L[low_mask]

    if np.any(high_mask):
        if rng.random() > prob_bound:
            x[high_mask] = x_U[high_mask]

    return x


def build_v_vectors(x1, x2, x_L, x_U, rng=None, prob_bound=0.5):
    """
    Build v1..v5 with probabilistic bound correction (MATLAB-like).
    """
    rng = _as_rng(rng)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x_L = np.asarray(x_L, dtype=float)
    x_U = np.asarray(x_U, dtype=float)

    d = 0.5 * (x2 - x1)
    v = np.zeros((5, x1.shape[0]), dtype=float)

    v[0] = _probabilistic_bound_correction(x1 - d, x_L, x_U, rng, prob_bound)
    v[1] = x1
    v[2] = _probabilistic_bound_correction(x1 + d, x_L, x_U, rng, prob_bound)
    v[3] = x2
    v[4] = _probabilistic_bound_correction(x2 + d, x_L, x_U, rng, prob_bound)

    return v


def build_combination_pairs(refset, rng=None, max_pairs=None, strategy="structured"):
    """
    Build parent pairs for combination using RefSet structure.

    Inputs:
    - refset['x'], refset['f'] are required
    - refset['idx_r1'], refset['idx_r2'] are used when present
    Returns:
    - dict with keys: pairs, r1_pos, r2_pos, source
    """
    if strategy not in (None, "structured"):
        # Optional non-blocking guard: unsupported values fall back to structured behavior.
        strategy = "structured"

    x = np.asarray(refset["x"], dtype=float)
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
            f = np.asarray(refset["f"], dtype=float)
            sorted_pos = np.argsort(f)
            split = n // 2
            r1_pos = sorted_pos[:split]
            r2_pos = sorted_pos[split:]
    else:
        f = np.asarray(refset["f"], dtype=float)
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


def _pair_combinations_from_v(v, x1, x2, x1_val, x2_val, pair_type, rng):
    """
    Build C candidates and parent references exactly by pair branch (4/2/3).
    """
    c_list = []
    p_list = []
    pval_list = []
    r2r2_branch = None

    if pair_type == "r1_r1":
        c_list.append(_random_segment_point(v[0], v[1], rng))
        p_list.append(x1)
        pval_list.append(x1_val)

        c_list.append(_random_segment_point(v[1], v[2], rng))
        p_list.append(x1)
        pval_list.append(x1_val)

        c_list.append(_random_segment_point(v[3], v[4], rng))
        p_list.append(x2)
        pval_list.append(x2_val)

        c_list.append(_random_segment_point(v[2], v[3], rng))
        p_list.append(x2)
        pval_list.append(x2_val)

    elif pair_type == "r2_r2":
        c_list.append(_random_segment_point(v[1], v[2], rng))
        p_list.append(x1)
        pval_list.append(x1_val)

        a = rng.random()
        if a < 0.5:
            c_list.append(_random_segment_point(v[1], v[0], rng))
            p_list.append(x1)
            pval_list.append(x1_val)
            r2r2_branch = "left"
        else:
            c_list.append(_random_segment_point(v[3], v[4], rng))
            p_list.append(x2)
            pval_list.append(x2_val)
            r2r2_branch = "right"

    elif pair_type == "mixed":
        c_list.append(_random_segment_point(v[1], v[0], rng))
        p_list.append(x1)
        pval_list.append(x1_val)

        c_list.append(_random_segment_point(v[1], v[2], rng))
        p_list.append(x1)
        pval_list.append(x1_val)

        c_list.append(_random_segment_point(v[3], v[4], rng))
        p_list.append(x2)
        pval_list.append(x2_val)
    else:
        raise ValueError("pair_type must be one of: 'r1_r1', 'r2_r2', 'mixed'.")

    return c_list, p_list, pval_list, r2r2_branch


def ssm_beyond_pair(
    z1,
    z2,
    z2_val,
    eval_fn,
    x_L,
    x_U,
    rng=None,
    prob_bound=0.5,
    max_steps=100,
):
    """
    MATLAB-like ssm_beyond (ssm_trascender) using eval callback.
    """
    rng = _as_rng(rng)
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    x_L = np.asarray(x_L, dtype=float)
    x_U = np.asarray(x_U, dtype=float)

    denom = 1.0
    n_improve = 1
    n_eval = 0
    steps = 0

    new_child_x = []
    new_child_val = []

    while steps < max_steps:
        steps += 1

        d = (z2 - z1) / denom
        zv1 = z2
        zv2 = _probabilistic_bound_correction(z2 + d, x_L, x_U, rng, prob_bound)

        xnew = _random_segment_point(zv1, zv2, rng)

        include, val_penalty, payload = _parse_eval_output(eval_fn(xnew))
        n_eval += 1

        if not include:
            break

        new_child_x.append(np.asarray(payload.get("x", xnew), dtype=float))
        new_child_val.append(val_penalty)

        if val_penalty < z2_val:
            z1 = z2
            z2 = xnew
            z2_val = val_penalty

            n_improve += 1
            if n_improve == 2:
                denom = denom / 2.0
                n_improve = 0
        else:
            break

    nvar = z1.shape[0]
    return {
        "x": np.vstack(new_child_x) if new_child_x else np.empty((0, nvar), dtype=float),
        "val": np.asarray(new_child_val, dtype=float),
        "n_eval": n_eval,
    }


def ssm_combination_pair(
    x1,
    x2,
    x1_val,
    x2_val,
    pair_type,
    x_L,
    x_U,
    eval_fn,
    rng=None,
    prob_bound=0.5,
    enable_beyond=True,
):
    """
    Pair-level MATLAB-like combination flow with eval and optional beyond.
    """
    rng = _as_rng(rng)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x_L = np.asarray(x_L, dtype=float)
    x_U = np.asarray(x_U, dtype=float)

    v = build_v_vectors(x1, x2, x_L, x_U, rng=rng, prob_bound=prob_bound)
    c_list, p_list, pval_list, r2r2_branch = _pair_combinations_from_v(
        v, x1, x2, float(x1_val), float(x2_val), pair_type, rng
    )

    n_eval = 0
    n_beyond_calls = 0
    accepted_x = []
    accepted_val = []
    beyond_x = []
    beyond_val = []

    first_parent_is_r1 = pair_type in ("r1_r1", "mixed")

    for i, c in enumerate(c_list):
        include, val_penalty, payload = _parse_eval_output(eval_fn(c))
        n_eval += 1

        if not include:
            continue

        x_eval = np.asarray(payload.get("x", c), dtype=float)
        accepted_x.append(x_eval)
        accepted_val.append(val_penalty)

        if enable_beyond and val_penalty < pval_list[i] and first_parent_is_r1:
            if not (pair_type == "mixed" and i == 2):
                n_beyond_calls += 1
                beyond = ssm_beyond_pair(
                    z1=p_list[i],
                    z2=x_eval,
                    z2_val=val_penalty,
                    eval_fn=eval_fn,
                    x_L=x_L,
                    x_U=x_U,
                    rng=rng,
                    prob_bound=prob_bound,
                )
                n_eval += beyond["n_eval"]
                if beyond["x"].size > 0:
                    beyond_x.append(beyond["x"])
                    beyond_val.append(beyond["val"])

    nvar = x1.shape[0]
    accepted_x_arr = np.vstack(accepted_x) if accepted_x else np.empty((0, nvar), dtype=float)
    accepted_val_arr = np.asarray(accepted_val, dtype=float)
    beyond_x_arr = np.vstack(beyond_x) if beyond_x else np.empty((0, nvar), dtype=float)
    beyond_val_arr = np.concatenate(beyond_val) if beyond_val else np.asarray([], dtype=float)

    return {
        "v": v,
        "pair_type": pair_type,
        "r2r2_branch": r2r2_branch,
        "candidates_raw": np.vstack(c_list) if c_list else np.empty((0, nvar), dtype=float),
        "accepted_x": accepted_x_arr,
        "accepted_val": accepted_val_arr,
        "beyond_x": beyond_x_arr,
        "beyond_val": beyond_val_arr,
        "n_combin": len(c_list),
        "n_eval": n_eval,
        "n_beyond_calls": n_beyond_calls,
    }


def ssm_combination_refset(
    refset,
    x_L,
    x_U,
    eval_fn,
    rng=None,
    prob_bound=0.5,
    max_pairs=None,
    strategy="structured",
    enable_beyond=True,
    deduplicate=True,
):
    """
    RefSet-level driver for MATLAB-like pair combination + evaluation flow.
    """
    rng = _as_rng(rng)
    x_ref = np.asarray(refset["x"], dtype=float)
    f_ref = np.asarray(refset["f"], dtype=float)

    pairing = build_combination_pairs(refset, rng=rng, max_pairs=max_pairs, strategy=strategy)

    all_x = []
    all_val = []
    per_pair = []
    n_eval_total = 0

    for i, j, pair_type in pairing["pairs"]:
        pair_result = ssm_combination_pair(
            x1=x_ref[i],
            x2=x_ref[j],
            x1_val=f_ref[i],
            x2_val=f_ref[j],
            pair_type=pair_type,
            x_L=x_L,
            x_U=x_U,
            eval_fn=eval_fn,
            rng=rng,
            prob_bound=prob_bound,
            enable_beyond=enable_beyond,
        )
        per_pair.append({"i": i, "j": j, **pair_result})
        n_eval_total += pair_result["n_eval"]

        if pair_result["accepted_x"].size > 0:
            all_x.append(pair_result["accepted_x"])
            all_val.append(pair_result["accepted_val"])
        if pair_result["beyond_x"].size > 0:
            all_x.append(pair_result["beyond_x"])
            all_val.append(pair_result["beyond_val"])

    if not all_x:
        nvar = x_ref.shape[1]
        return {
            "x": np.empty((0, nvar), dtype=float),
            "val": np.asarray([], dtype=float),
            "pairing": pairing,
            "per_pair": per_pair,
            "n_eval": n_eval_total,
        }

    x_all = np.vstack(all_x)
    val_all = np.concatenate(all_val)

    if deduplicate:
        uniq_x, uniq_idx = np.unique(x_all, axis=0, return_index=True)
        order = np.argsort(uniq_idx)
        x_all = uniq_x[order]
        val_all = val_all[uniq_idx][order]

    return {
        "x": x_all,
        "val": val_all,
        "pairing": pairing,
        "per_pair": per_pair,
        "n_eval": n_eval_total,
    }


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
):
    """
    Generate deterministic, bounded, unevaluated candidates from RefSet.
    Does not mutate refset and does not evaluate objective functions.
    """
    x_ref = np.asarray(refset["x"], dtype=float)
    f_ref = np.asarray(refset["f"], dtype=float)

    pairing = build_combination_pairs(refset, rng=rng, max_pairs=max_pairs, strategy=strategy)

    all_candidates = []
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

    if not all_candidates:
        return np.empty((0, x_ref.shape[1]), dtype=float)

    candidates = np.vstack(all_candidates)
    candidates = _remove_rows_equal_to_any(candidates, x_ref)
    if deduplicate and candidates.size > 0:
        candidates = np.unique(candidates, axis=0)
    return candidates
