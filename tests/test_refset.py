import numpy as np
from ess.population import generate_diverse_population
from ess.refset import create_refset, update_refset
from problems.objective_functions import sphere


def _pop(seed=42, pop_size=20):
    x_L = np.array([-5.0, -5.0])
    x_U = np.array([5.0, 5.0])
    population = generate_diverse_population(pop_size, x_L, x_U, seed=seed)
    f_pop = np.array([sphere(x) for x in population])
    return population, f_pop


def test_refset_sizes_and_unique_indices():
    population, f_pop = _pop()
    refset_size = 10
    results = create_refset(population, f_pop, refset_size, rng=np.random.default_rng(0))

    assert results['x'].shape == (refset_size, 2)
    assert results['f'].shape == (refset_size,)
    assert len(results['idx_r1']) == refset_size // 2
    assert len(results['idx_r2']) == refset_size // 2
    assert len(results['idx_all']) == refset_size
    assert len(np.unique(results['idx_all'])) == refset_size


def test_refset1_quality_matches_sorted_fitness():
    population, f_pop = _pop()
    refset_size = 10
    results = create_refset(population, f_pop, refset_size, rng=np.random.default_rng(0))

    idx_sorted = np.argsort(f_pop)
    expected_r1 = idx_sorted[:refset_size // 2]
    np.testing.assert_array_equal(results['idx_r1'], expected_r1)
    np.testing.assert_array_equal(results['f'][:refset_size // 2], f_pop[expected_r1])


def test_refset2_is_random_subset_of_remaining_and_reproducible():
    population, f_pop = _pop(pop_size=50)
    refset_size = 10
    idx_sorted = np.argsort(f_pop)
    remaining = set(idx_sorted[refset_size // 2:].tolist())

    r_a = create_refset(population, f_pop, refset_size, rng=np.random.default_rng(7))
    r_b = create_refset(population, f_pop, refset_size, rng=np.random.default_rng(7))

    # RefSet2 se toma del pool restante (no de los mejores de RefSet1)
    assert set(r_a['idx_r2'].tolist()).issubset(remaining)
    # Reproducible con la misma semilla
    np.testing.assert_array_equal(r_a['idx_r2'], r_b['idx_r2'])


def test_update_refset_replaces_only_on_improvement_and_sorts():
    refset = {
        "x": np.array([[0.0], [1.0], [2.0], [3.0]]),
        "f": np.array([10.0, 20.0, 30.0, 40.0]),
    }
    # candidato mejora slot 2 (30 -> 5) y NO mejora slot 1 (25 > 20)
    cand_x = np.array([[9.0], [8.0]])
    cand_f = np.array([5.0, 25.0])
    parent_pos = np.array([2, 1])

    new_refset, info = update_refset(refset, cand_x, cand_f, parent_pos)

    assert 5.0 in new_refset["f"].tolist()
    assert 25.0 not in new_refset["f"].tolist()
    # RefSet reordenado ascendente
    assert np.all(np.diff(new_refset["f"]) >= 0)
    assert info["n_changed"] == 1


def test_update_refset_best_candidate_wins_per_slot():
    refset = {"x": np.array([[0.0], [1.0]]), "f": np.array([10.0, 20.0])}
    # dos candidatos al mismo slot 0; gana el menor (3.0)
    cand_x = np.array([[7.0], [8.0]])
    cand_f = np.array([6.0, 3.0])
    parent_pos = np.array([0, 0])

    new_refset, info = update_refset(refset, cand_x, cand_f, parent_pos)

    assert new_refset["f"].min() == 3.0
    assert info["n_changed"] == 1


def test_update_refset_no_candidates_only_sorts():
    refset = {"x": np.array([[0.0], [1.0]]), "f": np.array([20.0, 10.0])}
    new_refset, info = update_refset(
        refset, np.empty((0, 1)), np.empty((0,)), np.empty((0,), dtype=int)
    )

    np.testing.assert_array_equal(new_refset["f"], np.array([10.0, 20.0]))
    assert info["n_changed"] == 0
