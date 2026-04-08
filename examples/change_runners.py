import numpy as np

from ess.combination import (
    build_combination_pairs,
    build_v_vectors,
    combine_pair_core,
    generate_candidates_from_refset,
    ssm_combination_pair,
    ssm_combination_refset,
)


def run_ai_change_003():
    """
    Manual runner for AI-CHANGE-003.
    Prints deterministic candidate generation from RefSet combination.
    """
    refset = {
        "x": np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 1.5],
                [1.5, 0.5],
            ]
        ),
        "f": np.array([1.0, 2.0, 3.0, 4.0]),
        "idx_r1": np.array([0, 1]),
        "idx_r2": np.array([2, 3]),
        "idx_all": np.array([0, 1, 2, 3]),
    }
    x_L = np.array([-2.0, -2.0])
    x_U = np.array([2.0, 2.0])

    print("AI-CHANGE-003 | refset['x']:")
    print(refset["x"])
    print("AI-CHANGE-003 | refset['f']:")
    print(refset["f"])
    if "idx_r1" in refset:
        print("AI-CHANGE-003 | idx_r1:")
        print(refset["idx_r1"])
    if "idx_r2" in refset:
        print("AI-CHANGE-003 | idx_r2:")
        print(refset["idx_r2"])

    pairing = build_combination_pairs(refset)
    print("AI-CHANGE-003 | selected pairs (i, j, pair_type):")
    print(pairing["pairs"])

    x_ref = refset["x"]
    f_ref = refset["f"]
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
        print(f"AI-CHANGE-003 | pair ({i}, {j}) [{pair_type}] parents:")
        print("x_i =", x_ref[i])
        print("x_j =", x_ref[j])
        print("AI-CHANGE-003 | candidates from this pair:")
        print(pair_candidates)

    candidates = generate_candidates_from_refset(refset, x_L, x_U)
    print("AI-CHANGE-003 | final candidates after deduplication:")
    print("shape =", candidates.shape)
    print(candidates)
    return candidates


def run_change_003c():
    refset = {
        "x": np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 1.5],
                [1.5, 0.5],
            ]
        ),
        "f": np.array([1.0, 2.0, 3.0, 4.0]),
        "idx_r1": np.array([0, 1]),
        "idx_r2": np.array([2, 3]),
        "idx_all": np.array([0, 1, 2, 3]),
    }
    x_L = np.array([-2.0, -2.0])
    x_U = np.array([2.0, 2.0])
    rng = np.random.default_rng(5)

    def eval_fn(x):
        return {"include": True, "val_penalty": float(np.sum(x**2))}

    pairing = build_combination_pairs(refset)
    print("CHANGE-003C | pairs:", pairing["pairs"])

    for i, j, pair_type in pairing["pairs"]:
        x1 = refset["x"][i]
        x2 = refset["x"][j]
        print(f"\nCHANGE-003C | pair ({i},{j}) [{pair_type}]")
        v = build_v_vectors(x1, x2, x_L, x_U, rng=rng, prob_bound=0.5)
        print("v1..v5:")
        print(v)
        pair_out = ssm_combination_pair(
            x1=x1,
            x2=x2,
            x1_val=refset["f"][i],
            x2_val=refset["f"][j],
            pair_type=pair_type,
            x_L=x_L,
            x_U=x_U,
            eval_fn=eval_fn,
            rng=rng,
            prob_bound=0.5,
            enable_beyond=True,
        )
        print("n_combin:", pair_out["n_combin"], "| n_eval:", pair_out["n_eval"], "| n_beyond_calls:", pair_out["n_beyond_calls"])
        print("accepted_x:")
        print(pair_out["accepted_x"])
        if pair_out["beyond_x"].size > 0:
            print("beyond_x:")
            print(pair_out["beyond_x"])

    out = ssm_combination_refset(
        refset=refset,
        x_L=x_L,
        x_U=x_U,
        eval_fn=eval_fn,
        rng=np.random.default_rng(5),
        prob_bound=0.5,
        enable_beyond=True,
        deduplicate=True,
    )
    print("\nCHANGE-003C | final deduplicated x:")
    print(out["x"])
    print("final val:")
    print(out["val"])
    print("total evals:", out["n_eval"])
    return out


def run_ai_change(change_id):
    """
    Dispatch manual execution by AI-CHANGE id.
    """
    if change_id == 3:
        return run_ai_change_003()
    if str(change_id).lower() == "3c":
        return run_change_003c()
    raise ValueError(f"Runner for CHANGE-{change_id} is not implemented yet.")


if __name__ == "__main__":
    run_ai_change_003()
