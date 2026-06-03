import numpy as np

from .utils import evaluate



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
