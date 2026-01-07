import numpy as np
import pytest
from ess.main import ess_kernel_min
from problems.objective_functions import sphere, rosenbrock

def test_ess_min_invariants():
    # Configuration
    problem = {
        'f': sphere,
        'x_L': [-5.0, -5.0],
        'x_U': [5.0, 5.0]
    }
    opts = {'maxeval': 100, 'seed': 42}
    
    # Execute
    results = ess_kernel_min(problem, opts)
    
    # Invariants
    assert results['xbest'] is not None, "xbest must exist"
    assert np.all(results['xbest'] >= problem['x_L']), "xbest in bounds inferiores"
    assert np.all(results['xbest'] <= problem['x_U']), "xbest in bounds superiores"
    assert results['numeval'] == opts['maxeval'], "numeval must be maxeval"
    assert len(results['fbest_trace']) == opts['maxeval'], "fbest_trace must have maxeval entries"
    assert results['fbest'] == min(results['fbest_trace']), "fbest must be the minimum of trace"
    # Improvement: fbest <= initial fbest (first evaluation)
    assert results['fbest'] <= results['fbest_trace'][0], "fbest must improve or equal initial"
    
    # Reproducibility
    results2 = ess_kernel_min(problem, opts)
    assert np.allclose(results['xbest'], results2['xbest']), "xbest reproducible with seed"
    assert results['fbest'] == results2['fbest'], "fbest reproducible with seed"
    assert results['fbest_trace'] == results2['fbest_trace'], "fbest_trace reproducible with seed"

def test_ess_min_rosenbrock():
    # Test with Rosenbrock
    problem = {
        'f': rosenbrock,
        'x_L': [-2.0, -2.0],
        'x_U': [2.0, 2.0]
    }
    opts = {'maxeval': 50, 'seed': 123}
    
    results = ess_kernel_min(problem, opts)
    
    assert results['numeval'] == 50
    assert np.all(results['xbest'] >= problem['x_L'])
    assert np.all(results['xbest'] <= problem['x_U'])
    # Minimum Rosenbrock in [1,1,...], f=0
    # Few evaluations, not necessarily reaches it, but sanity
    assert results['fbest'] >= 0