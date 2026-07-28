import numpy as np

from .utils import evaluate
from .population import generate_diverse_population
from .refset import create_refset


def _default_dim_refset(n_var):
    """dim_refset automatico de ess_kernel.m (L337-341): raiz positiva de
    x^2 - x - 10*nvar = 0, redondeada hacia arriba y forzada a par."""
    dim_refset = int(np.ceil((1.0 + np.sqrt(1.0 + 40.0 * n_var)) / 2.0))
    if dim_refset % 2:
        dim_refset += 1
    return dim_refset


def ess_init(problem, opts, rng=None):
    """
    Fase de inicializacion de eSS, fiel a ess_kernel.m (antes del bucle iterativo).

    - Genera poblacion diversa (generate_diverse_population) y la evalua.
    - Incorpora x_0/f_0 opcionales: filas de x_0 sin f_0 se anaden a la poblacion a
      evaluar; filas con f_0 se anteponen sin reevaluar (se asumen factibles).
    - Construye el RefSet inicial (create_refset) y calcula fbest/xbest.
    - Inicializa trazas estilo MATLAB en 'results'.

    No abre el bucle, ni gemelos, ni beyond, ni busqueda local (fuera de alcance 004B).

    problem: dict con 'f', 'x_L', 'x_U' y opcionalmente 'x_0', 'f_0'.
    opts: dict; usa 'seed', 'dim_refset', 'ndiverse' si estan presentes.
    rng: np.random.Generator or None.
    Returns: dict de estado inicial.
    """
    if rng is None:
        rng = np.random.default_rng(opts.get("seed"))

    f = problem["f"]
    x_L = np.asarray(problem["x_L"], dtype=float)
    x_U = np.asarray(problem["x_U"], dtype=float)
    n_var = len(x_L)

    refset_size = opts.get("dim_refset") or _default_dim_refset(n_var)
    ndiverse = opts.get("ndiverse") or 10 * n_var
    if ndiverse < refset_size:
        ndiverse = refset_size

    x_0 = problem.get("x_0")
    f_0 = problem.get("f_0")
    x_0 = None if x_0 is None else np.atleast_2d(np.asarray(x_0, dtype=float))
    f_0 = None if f_0 is None else np.asarray(f_0, dtype=float).ravel()
    l_f0 = 0 if f_0 is None else len(f_0)
    l_x0 = 0 if x_0 is None else x_0.shape[0]

    diverse = generate_diverse_population(ndiverse, x_L, x_U, rng=rng)

    # x_0 sin f_0 conocido: se anaden para evaluar (ess_kernel.m L586-590)
    if x_0 is not None and l_x0 > l_f0:
        to_eval = np.vstack([x_0[l_f0:], diverse])
    else:
        to_eval = diverse

    f_eval = np.array([evaluate(f, xi) for xi in to_eval])
    numeval = int(len(f_eval))

    # x_0 con f_0 conocido: se anteponen sin reevaluar (ess_kernel.m L627-628)
    if l_f0 > 0:
        sol_x = np.vstack([x_0[:l_f0], to_eval])
        sol_f = np.concatenate([f_0, f_eval])
    else:
        sol_x = to_eval
        sol_f = f_eval

    refset = create_refset(sol_x, sol_f, refset_size, rng=rng)

    fbest_pos = int(np.argmin(refset["f"]))
    fbest = float(refset["f"][fbest_pos])
    xbest = refset["x"][fbest_pos].copy()

    results = {
        "f": [fbest],
        "x": [xbest],
        "neval": [numeval],
        "numeval": numeval,
        "fbest": fbest,
        "xbest": xbest,
    }

    return {
        "refset": refset,
        "fbest": fbest,
        "xbest": xbest,
        "numeval": numeval,
        "results": results,
        "x_L": x_L,
        "x_U": x_U,
        "n_var": n_var,
        "refset_size": refset_size,
        "ndiverse": ndiverse,
        "rng": rng,
    }


def ess_kernel_min(problem, opts):
    """
    Minimal serial eSS baseline: random sampling, evaluation and best tracking.

    problem: dict with 'f', 'x_L', 'x_U'
    opts: dict with 'maxeval' and optional 'seed'
    """
    rng = np.random.default_rng(opts.get("seed", 42))
    f = problem["f"]
    x_L = np.asarray(problem["x_L"], dtype=float)
    x_U = np.asarray(problem["x_U"], dtype=float)
    n_var = len(x_L)
    maxeval = opts["maxeval"]

    numeval = 0
    fbest = np.inf
    xbest = None
    fbest_trace = []

    while numeval < maxeval:
        x = rng.uniform(x_L, x_U, size=n_var)
        f_val = evaluate(f, x)
        numeval += 1
        fbest_trace.append(f_val)
        if f_val < fbest:
            fbest = f_val
            xbest = x.copy()

    return {
        "xbest": xbest,
        "fbest": fbest,
        "numeval": numeval,
        "fbest_trace": fbest_trace,
    }
