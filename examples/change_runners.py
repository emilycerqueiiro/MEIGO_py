import numpy as np

from ess.combination import (
    build_combination_pairs,
    combine_pair_core,
    generate_candidates_from_refset,
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


def run_ai_change(change_id):
    """
    Dispatch manual execution by AI-CHANGE id.
    """
    if change_id == 3:
        return run_ai_change_003()
    raise ValueError(f"Runner for AI-CHANGE-{change_id:03d} is not implemented yet.")


if __name__ == "__main__":
    run_ai_change_003()
