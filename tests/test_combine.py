import numpy as np

from ess.combination import (
    build_combination_pairs,
    combine_pair_core,
    generate_candidates_from_refset,
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
