import numpy as np
from ess.main import ess_init
from problems.objective_functions import sphere


def _problem():
    return {"f": sphere, "x_L": np.array([-5.0, -5.0]), "x_U": np.array([5.0, 5.0])}


def test_ess_init_refset_sizes_and_reproducible():
    opts = {"seed": 42, "dim_refset": 10, "ndiverse": 30}
    s_a = ess_init(_problem(), opts, rng=np.random.default_rng(42))
    s_b = ess_init(_problem(), opts, rng=np.random.default_rng(42))

    assert s_a["refset"]["x"].shape == (10, 2)
    assert s_a["refset"]["f"].shape == (10,)
    assert s_a["numeval"] == 30
    np.testing.assert_allclose(s_a["refset"]["x"], s_b["refset"]["x"])
    assert s_a["fbest"] == float(np.min(s_a["refset"]["f"]))


def test_ess_init_uses_x0_with_f0_without_reevaluating():
    prob = _problem()
    # punto x_0 con f_0 conocido = optimo de sphere (0.0)
    prob["x_0"] = np.array([[0.0, 0.0]])
    prob["f_0"] = np.array([0.0])
    opts = {"seed": 1, "dim_refset": 6, "ndiverse": 20}

    s = ess_init(prob, opts, rng=np.random.default_rng(1))

    assert s["fbest"] == 0.0
    np.testing.assert_allclose(s["xbest"], np.array([0.0, 0.0]))
    # x_0 con f_0 no cuenta como evaluacion
    assert s["numeval"] == 20


def test_ess_init_x0_without_f0_is_evaluated():
    prob = _problem()
    prob["x_0"] = np.array([[1.0, 1.0]])  # sin f_0 -> se evalua
    opts = {"seed": 2, "dim_refset": 6, "ndiverse": 15}

    s = ess_init(prob, opts, rng=np.random.default_rng(2))

    # 15 diversos + 1 punto x_0 sin f_0 = 16 evaluaciones
    assert s["numeval"] == 16
