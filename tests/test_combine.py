import numpy as np

from ess.combination import (
    build_combination_pairs,
    build_v_vectors,
    combine_pair_core,
    generate_candidates_from_refset,
    ssm_combination_pair,
    ssm_combination_refset,
)


def test_pair_selection_uses_idx_fields_when_present():
    refset = {
        "x": np.array([[0.0], [1.0], [2.0], [3.0]]),
        "f": np.array([0.1, 0.2, 100.0, 101.0]),
        "idx_r1": np.array([2, 3]),
        "idx_r2": np.array([0, 1]),
        "idx_all": np.array([2, 3, 0, 1]),
    }

    info = build_combination_pairs(refset)

    assert info["source"] == "idx_fields"
    np.testing.assert_array_equal(info["r1_pos"], np.array([0, 1]))
    np.testing.assert_array_equal(info["r2_pos"], np.array([2, 3]))


def test_pair_selection_fallback_to_fitness_when_idx_missing():
    refset = {
        "x": np.array([[0.0], [1.0], [2.0], [3.0]]),
        "f": np.array([30.0, 10.0, 20.0, 40.0]),
    }

    info = build_combination_pairs(refset)

    assert info["source"] == "fitness_fallback"
    np.testing.assert_array_equal(info["r1_pos"], np.array([1, 2]))
    np.testing.assert_array_equal(info["r2_pos"], np.array([0, 3]))


def test_pair_selection_fallback_to_fitness_when_idx_do_not_match_idx_all():
    refset = {
        "x": np.array([[0.0], [1.0], [2.0], [3.0]]),
        "f": np.array([30.0, 10.0, 20.0, 40.0]),
        "idx_r1": np.array([10, 11]),
        "idx_r2": np.array([12, 13]),
        "idx_all": np.array([0, 1, 2, 3]),
    }

    info = build_combination_pairs(refset)

    assert info["source"] == "fitness_fallback"
    np.testing.assert_array_equal(info["r1_pos"], np.array([1, 2]))
    np.testing.assert_array_equal(info["r2_pos"], np.array([0, 3]))


def test_strategy_guard_is_non_blocking():
    refset = {
        "x": np.array([[0.0], [1.0], [2.0], [3.0]]),
        "f": np.array([30.0, 10.0, 20.0, 40.0]),
    }

    info = build_combination_pairs(refset, strategy="unsupported_strategy")

    assert isinstance(info["pairs"], list)
    assert info["source"] == "fitness_fallback"


def test_combine_pair_core_respects_bounds_and_excludes_parents():
    x1 = np.array([0.0, 0.0])
    x2 = np.array([2.0, 2.0])
    x_L = np.array([0.0, 0.0])
    x_U = np.array([1.5, 1.5])

    candidates = combine_pair_core(x1, x2, x_L, x_U, pair_type="r1_r1", include_parents=False)

    assert candidates.shape[1] == 2
    assert np.all(candidates >= x_L)
    assert np.all(candidates <= x_U)
    assert not np.any(np.all(np.isclose(candidates, x1), axis=1))
    assert not np.any(np.all(np.isclose(candidates, x2), axis=1))


def test_generate_candidates_does_not_evaluate_objective(monkeypatch):
    import ess.utils as utils

    def _raise_if_called(*args, **kwargs):
        raise RuntimeError("evaluate() should not be called in combination stage")

    monkeypatch.setattr(utils, "evaluate", _raise_if_called)

    refset = {
        "x": np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 1.5], [1.5, 0.5]]),
        "f": np.array([1.0, 2.0, 3.0, 4.0]),
        "idx_r1": np.array([0, 1]),
        "idx_r2": np.array([2, 3]),
        "idx_all": np.array([0, 1, 2, 3]),
    }
    x_L = np.array([-2.0, -2.0])
    x_U = np.array([2.0, 2.0])

    candidates = generate_candidates_from_refset(refset, x_L, x_U)

    assert candidates.ndim == 2


def test_generate_candidates_does_not_mutate_refset_inputs():
    refset = {
        "x": np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 1.5], [1.5, 0.5]]),
        "f": np.array([1.0, 2.0, 3.0, 4.0]),
        "idx_r1": np.array([0, 1]),
        "idx_r2": np.array([2, 3]),
        "idx_all": np.array([0, 1, 2, 3]),
    }

    x_before = refset["x"].copy()
    f_before = refset["f"].copy()

    _ = generate_candidates_from_refset(refset, np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

    np.testing.assert_array_equal(refset["x"], x_before)
    np.testing.assert_array_equal(refset["f"], f_before)


def test_generation_is_deterministic():
    refset = {
        "x": np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 1.5], [1.5, 0.5]]),
        "f": np.array([1.0, 2.0, 3.0, 4.0]),
        "idx_r1": np.array([0, 1]),
        "idx_r2": np.array([2, 3]),
        "idx_all": np.array([0, 1, 2, 3]),
    }
    x_L = np.array([-2.0, -2.0])
    x_U = np.array([2.0, 2.0])

    c1 = generate_candidates_from_refset(refset, x_L, x_U)
    c2 = generate_candidates_from_refset(refset, x_L, x_U)

    np.testing.assert_allclose(c1, c2)


def test_probabilistic_bounds_seeded_golden():
    rng = np.random.default_rng(123)
    x1 = np.array([-1.8, 1.8])
    x2 = np.array([1.8, -1.8])
    x_L = np.array([-2.0, -2.0])
    x_U = np.array([2.0, 2.0])

    v = build_v_vectors(x1, x2, x_L, x_U, rng=rng, prob_bound=0.5)

    expected = np.array(
        [
            [-2.0, 3.6],
            [-1.8, 1.8],
            [0.0, 0.0],
            [1.8, -1.8],
            [3.6, -3.6],
        ]
    )
    np.testing.assert_allclose(v, expected)


def test_r2_r2_random_branch_seeded_golden():
    rng = np.random.default_rng(7)

    def eval_fn(x):
        return {"include": True, "val_penalty": float(np.sum(x))}

    out = ssm_combination_pair(
        x1=np.array([0.2, 0.4]),
        x2=np.array([0.8, 0.6]),
        x1_val=1.0,
        x2_val=2.0,
        pair_type="r2_r2",
        x_L=np.array([0.0, 0.0]),
        x_U=np.array([1.0, 1.0]),
        eval_fn=eval_fn,
        rng=rng,
        prob_bound=0.5,
        enable_beyond=False,
    )

    assert out["n_combin"] == 2
    assert out["r2r2_branch"] in ("left", "right")
    np.testing.assert_allclose(out["candidates_raw"], np.array([[0.43270571, 0.42252072], [0.02528931, 0.39947347]]))


def test_pair_level_eval_filter_flow():
    rng = np.random.default_rng(11)

    def eval_fn(x):
        val = float(np.sum(x))
        return {"include": val < 1.2, "val_penalty": val}

    out = ssm_combination_pair(
        x1=np.array([0.0, 0.0]),
        x2=np.array([1.0, 1.0]),
        x1_val=0.1,
        x2_val=2.0,
        pair_type="r1_r1",
        x_L=np.array([-2.0, -2.0]),
        x_U=np.array([2.0, 2.0]),
        eval_fn=eval_fn,
        rng=rng,
        prob_bound=0.5,
        enable_beyond=False,
    )

    assert out["n_eval"] == out["n_combin"] == 4
    assert out["accepted_x"].shape[0] < 4


def test_beyond_called_only_when_matlab_condition_holds():
    rng = np.random.default_rng(19)

    def eval_fn(x):
        return {"include": True, "val_penalty": 0.0}

    out = ssm_combination_pair(
        x1=np.array([0.0, 0.0]),
        x2=np.array([1.0, 1.0]),
        x1_val=10.0,
        x2_val=10.0,
        pair_type="mixed",
        x_L=np.array([-2.0, -2.0]),
        x_U=np.array([2.0, 2.0]),
        eval_fn=eval_fn,
        rng=rng,
        prob_bound=0.5,
        enable_beyond=True,
    )

    # In mixed branch, beyond is skipped for i==3 (0-based i==2).
    assert out["n_combin"] == 3
    assert out["n_beyond_calls"] == 2


def test_eval_fn_must_return_dict():
    rng = np.random.default_rng(3)

    def eval_fn(_x):
        return 1.0

    try:
        ssm_combination_pair(
            x1=np.array([0.0, 0.0]),
            x2=np.array([1.0, 1.0]),
            x1_val=1.0,
            x2_val=2.0,
            pair_type="mixed",
            x_L=np.array([-2.0, -2.0]),
            x_U=np.array([2.0, 2.0]),
            eval_fn=eval_fn,
            rng=rng,
            prob_bound=0.5,
            enable_beyond=False,
        )
    except TypeError as exc:
        assert "eval_fn must return a dict" in str(exc)
    else:
        raise AssertionError("Expected TypeError when eval_fn does not return a dict.")


def test_refset_driver_full_flow_seeded_golden():
    rng = np.random.default_rng(5)

    def eval_fn(x):
        return {"include": True, "val_penalty": float(np.sum(x**2))}

    refset = {
        "x": np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 1.5], [1.5, 0.5]]),
        "f": np.array([1.0, 2.0, 3.0, 4.0]),
        "idx_r1": np.array([0, 1]),
        "idx_r2": np.array([2, 3]),
        "idx_all": np.array([0, 1, 2, 3]),
    }

    out = ssm_combination_refset(
        refset=refset,
        x_L=np.array([-2.0, -2.0]),
        x_U=np.array([2.0, 2.0]),
        eval_fn=eval_fn,
        rng=rng,
        prob_bound=0.5,
        enable_beyond=False,
        deduplicate=True,
    )

    assert out["x"].ndim == 2
    assert out["val"].ndim == 1
    assert out["x"].shape[0] == out["val"].shape[0]
    np.testing.assert_allclose(
        out["x"][:3],
        np.array([[-0.2019852, -0.38649417], [0.07145035, 0.04044803], [0.59584222, 1.7042366]]),
    )
