import numpy as np
from ess.population import generate_diverse_population
from ess.refset import create_refset
from problems.objective_functions import sphere


def test_refset_sizes_and_unique_indices(opts=None):
    seed = opts.get('seed', 42) if opts else 42
    pop_size = 20
    x_L = np.array([-5.0, -5.0])
    x_U = np.array([5.0, 5.0])
    population = generate_diverse_population(pop_size, x_L, x_U, seed=seed)
    f_pop = np.array([sphere(x) for x in population])
    refset_size = 10
    results = create_refset(population, f_pop, refset_size)

    assert results['x'].shape == (refset_size, 2)
    assert results['f'].shape == (refset_size,)
    assert len(results['idx_r1']) == refset_size // 2
    assert len(results['idx_r2']) == refset_size // 2
    assert len(results['idx_all']) == refset_size
    assert len(np.unique(results['idx_all'])) == refset_size


def test_refset1_quality_matches_sorted_fitness(opts=None):
    seed = opts.get('seed', 42) if opts else 42
    pop_size = 20
    x_L = np.array([-5.0, -5.0])
    x_U = np.array([5.0, 5.0])
    population = generate_diverse_population(pop_size, x_L, x_U, seed=seed)
    f_pop = np.array([sphere(x) for x in population])
    refset_size = 10
    results = create_refset(population, f_pop, refset_size)

    idx_sorted = np.argsort(f_pop)
    expected_r1 = idx_sorted[:refset_size // 2]
    np.testing.assert_array_equal(results['idx_r1'], expected_r1)
    np.testing.assert_array_equal(results['f'][:refset_size // 2], f_pop[expected_r1])


def test_refset2_diversity_beats_random(opts=None):
    seed_base = opts.get('seed', 42) if opts else 42
    seeds = [seed_base, seed_base + 1, seed_base + 2]
    pop_size = 50
    refset_size = 10
    x_L = np.array([-5.0, -5.0])
    x_U = np.array([5.0, 5.0])

    selected_min_dists = []
    random_min_dists = []

    for seed in seeds:
        population = generate_diverse_population(pop_size, x_L, x_U, seed=seed)
        f_pop = np.array([sphere(x) for x in population])
        results = create_refset(population, f_pop, refset_size)

        refset2 = results['x'][refset_size // 2:]
        min_dists = []
        for x in refset2:
            dists = [np.linalg.norm(x - r) for r in results['x'][:refset_size // 2]]
            min_dists.append(np.min(dists))
        selected_min_dists.append(np.mean(min_dists))

        rng2 = np.random.default_rng(seed + 100)
        random_idx = rng2.choice(pop_size, refset_size // 2, replace=False)
        random_refset2 = population[random_idx]
        min_dists_rand = []
        for x in random_refset2:
            dists = [np.linalg.norm(x - r) for r in results['x'][:refset_size // 2]]
            min_dists_rand.append(np.min(dists))
        random_min_dists.append(np.mean(min_dists_rand))

    assert np.mean(selected_min_dists) > np.mean(random_min_dists)
