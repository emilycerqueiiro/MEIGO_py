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

def euclidean_distances(A, B):
    """
    Calcula distancias euclideas entre cada fila de A y cada fila de B.
    A, B: np.ndarray (n, dim), (m, dim)
    Returns: np.ndarray (n, m)
    """
    A = np.array(A)
    B = np.array(B)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return dists