import numpy as np

def evaluate(f, x):
    """
    Evalúa la función objetivo f en x.
    f: callable, función objetivo.
    x: np.ndarray, punto de evaluación.
    Returns: float, valor de f(x).
    """
    return f(x)

def project_bounds(x, x_L, x_U):
    """
    Proyecta x dentro de los bounds [x_L, x_U] mediante clip.
    x: np.ndarray, punto.
    x_L, x_U: np.ndarray, bounds inferiores y superiores.
    Returns: np.ndarray, x clipped.
    """
    return np.clip(x, x_L, x_U)