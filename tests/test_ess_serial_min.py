import numpy as np
import pytest
from ess.main import ess_kernel_min
from problems.objective_functions import sphere, rosenbrock

def test_ess_min_invariants():
    # Configuración
    problem = {
        'f': sphere,
        'x_L': [-5.0, -5.0],
        'x_U': [5.0, 5.0]
    }
    opts = {'maxeval': 100, 'seed': 42}
    
    # Ejecutar
    results = ess_kernel_min(problem, opts)
    
    # Invariantes
    assert results['xbest'] is not None, "xbest debe existir"
    assert np.all(results['xbest'] >= problem['x_L']), "xbest dentro bounds inferiores"
    assert np.all(results['xbest'] <= problem['x_U']), "xbest dentro bounds superiores"
    assert results['numeval'] == opts['maxeval'], "numeval debe ser maxeval"
    assert len(results['fbest_trace']) == opts['maxeval'], "fbest_trace debe tener maxeval entradas"
    assert results['fbest'] == min(results['fbest_trace']), "fbest debe ser el mínimo de trace"
    # Mejora: fbest <= fbest inicial (primera evaluación)
    assert results['fbest'] <= results['fbest_trace'][0], "fbest debe mejorar o igualar inicial"
    
    # Reproducibilidad
    results2 = ess_kernel_min(problem, opts)
    assert np.allclose(results['xbest'], results2['xbest']), "xbest reproducible con seed"
    assert results['fbest'] == results2['fbest'], "fbest reproducible con seed"
    assert results['fbest_trace'] == results2['fbest_trace'], "fbest_trace reproducible con seed"

def test_ess_min_rosenbrock():
    # Prueba con Rosenbrock
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
    # Rosenbrock mínimo en [1,1,...], f=0
    # Con pocas evaluaciones, no necesariamente llega, pero sanity
    assert results['fbest'] >= 0