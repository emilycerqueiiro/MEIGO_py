import numpy as np

def evaluate(f, x):
    """
    Evalúa la función objetivo f en x.
    f: callable, función objetivo.
    x: np.ndarray, punto de evaluación.
    Returns: float, valor de f(x).
    """
    return f(x)

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

def project_bounds(x, x_L, x_U):
    """
    Proyecta un punto x al hipercubo [x_L, x_U] clipeando cada componente.
    
    Uso:
    - Búsqueda local: solvers pueden explorar fuera de bounds; esto asegura factibilidad.
    - Combinación de soluciones: offspring pueden violar bounds; clipear es una estrategia simple.
    - Inicialización defensiva: aunque muestreo uniforme ya garantiza, usar como safeguard.
    
    x: np.ndarray (dim,)
    x_L, x_U: np.ndarray (dim,)
    Returns: x_proj con x_L[i] <= x_proj[i] <= x_U[i]
    """
    x = np.array(x)
    x_L = np.array(x_L)
    x_U = np.array(x_U)
    return np.clip(x, x_L, x_U)