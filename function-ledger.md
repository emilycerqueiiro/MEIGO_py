# Function Ledger - MEIGO_py

## Propósito
Registro centralizado de todas las funciones propias (OWN) y externas relevantes (EXT) implementadas en MEIGO_py, organizado por área funcional y nivel de dificultad. Facilita la documentación, defensa del TFG y comprensión del proyecto por parte de desarrolladores.

## Convenciones
- **OWN:** Función propia, desarrollada en MEIGO_py.
- **EXT:** Función externa (NumPy, SciPy, etc.), usada por primera vez o crítica para entender la implementación.
- **L1:** Básico — operaciones simples, entrada/salida directa, fácil de entender.
- **L2:** Medio — lógica condicional, iteración, o uso de primitivas NumPy comunes (argsort, norm).
- **L3:** Avanzado — algoritmos complejos, broadcasting avanzado, operaciones multidimensionales.

## Índice de Categorías
1. Core eSS / Kernels
2. RefSet / Diversidad / Distancias
3. Combinación / Generación de soluciones
4. Evaluación / Bounds / Penalizaciones
5. RNG / Muestreo / Inicialización
6. Externas clave (NumPy/SciPy)

---

## 1. Core eSS / Kernels

### OWN ess_kernel_min (L2)
**Dónde se usa:** `ess/main.py:ess_kernel_min`

**Origen:** AI-CHANGE-001

**Qué hace:** Implementa un kernel mínimo y serial de eSS que itera muestreo uniforme, evaluación y selección greedy.

**Por qué se usa aquí:** Es el punto de entrada para la versión mínima de eSS, permitiendo verificación básica del flujo de optimización sin complejidad de RefSet, combinación o búsqueda local.

**Inputs & shapes:**
- `problem` (dict): `{'f': callable, 'x_L': ndarray (dim,), 'x_U': ndarray (dim,)}`
- `opts` (dict): `{'maxeval': int, 'seed': int (opcional)}`

**Outputs & shapes:**
- `Results` (dict): `{'xbest': ndarray (dim,), 'fbest': float, 'numeval': int, 'fbest_trace': list}`

**Ejemplo mínimo:**
```python
problem = {'f': sphere, 'x_L': [-5, -5], 'x_U': [5, 5]}
opts = {'maxeval': 100, 'seed': 42}
results = ess_kernel_min(problem, opts)
# results['fbest'] ≈ 0.xxx después de 100 evaluaciones
```

**Pitfalls:**
- Sin seed en `opts`, usa 42 por defecto; comportamiento no reproducible si `opts` es None.
- `fbest_trace` registra todos los valores, incluso si empeoran; el usuario debe buscar el mínimo manualmente si no actualiza `fbest` correctamente.
- Sin manejo de bounds en la evaluación (confía en que el muestreo está dentro).
- No paraleliza; apropiado solo para pruebas y baselines.

**Impacto en eSS/VNS:** Baseline serial de optimización global; establece los requisitos mínimos (reproducibilidad, tracking de traza).

---

## 2. RefSet / Diversidad / Distancias

### OWN create_refset (L2)
**Dónde se usa:** `ess/refset.py:create_refset`

**Origen:** AI-CHANGE-001

**Qué hace:** Particiona una población evaluada en RefSet1 (mejores por fitness) y RefSet2 (más diversos), combinándolas en un RefSet balanceado.

**Por qué se usa aquí:** Implementa el balance calidad–diversidad fundamental en eSS, preparando el conjunto de referencia para futuras combinaciones y búsqueda local.

**Inputs & shapes:**
- `population` (ndarray): `(n, dim)`
- `f_pop` (ndarray): `(n,)` — valores de fitness para cada punto.
- `refset_size` (int): tamaño total del RefSet (debe ser par).
- `refset1_size` (int, opcional): tamaño de RefSet1; por defecto `refset_size // 2`.

**Outputs & shapes:**
- `Results` (dict): `{'x': (refset_size, dim), 'f': (refset_size,), 'idx_r1': (refset1_size,), 'idx_r2': (refset_size - refset1_size,), 'idx_all': (refset_size,)}`

**Ejemplo mínimo:**
```python
population = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
f_pop = np.array([5.0, 25.0, 61.0, 113.0])
refset = create_refset(population, f_pop, refset_size=4)
# refset['x'][0:2] = [[1, 2], [3, 4]]  # RefSet1 (mejores)
# refset['x'][2:4] = [[5, 6], [7, 8]]  # RefSet2 (diversos, seleccionados)
```

**Pitfalls:**
- Requiere `refset_size` par; si es impar, el comportamiento es indefinido.
- Los índices en `idx_r1`, `idx_r2` refieren a las filas de `population`, no a una población global; mantener consistencia con el pool original.
- `select_most_diverse` es iterativo; puede ser lento si `refset_size` es muy grande (O(k * (n - k) * r) con k = tamaño RefSet2, n = población, r = RefSet1).
- Si la población es pequeña (n < refset_size), el refset estará incompleto.

**Impacto en eSS/VNS:** Core de eSS; balancea intensificación (RefSet1) y diversificación (RefSet2), afectando la trayectoria de búsqueda.

---

### OWN select_most_diverse (L3)
**Dónde se usa:** `ess/refset.py:select_most_diverse`

**Origen:** AI-CHANGE-001

**Qué hace:** Selecciona k candidatos que maximizan la distancia mínima al RefSet actual (máximin iterativo con acumulación).

**Por qué se usa aquí:** Implementa la estrategia de diversidad en eSS; cada iteración elige el candidato más lejano al RefSet hasta ese momento, asegurando distribución global.

**Inputs & shapes:**
- `candidates` (ndarray): `(m, dim)` — soluciones del pool no seleccionado.
- `reference` (ndarray): `(r, dim)` — RefSet1 o RefSet actual (crece con `diverse`).
- `k` (int): número de soluciones a seleccionar.
- `candidate_indices` (ndarray): `(m,)` — índices originales en la población global.

**Outputs & shapes:**
- `(diverse: ndarray (k, dim), indices: ndarray (k,))`

**Ejemplo mínimo:**
```python
candidates = np.array([[0, 0], [3, 4], [5, 5]])
reference = np.array([[1, 1], [4, 4]])
diverse, idx = select_most_diverse(candidates, reference, k=1, candidate_indices=np.array([10, 11, 12]))
# Selecciona [5, 5] (más lejano a reference), retorna índice 12
```

**Pitfalls:**
- Complejidad O(k * (n - k) * r): evitar usar con m >> 1000 si r >> 10.
- Broadcasting incorrecto en `np.linalg.norm`: asegúrate de que `c` y `r` sean 1D y 2D respectivamente (evitar rank mismatch).
- `np.delete` en el bucle es ineficiente; considera usar máscaras booleanas si k es grande.
- Si `candidates` está vacío o k > len(candidates), retorna parcial (sin error, pero potencialmente incorrecto).

**Impacto en eSS/VNS:** Maximiza exploración global mediante diversidad métrica; crucial para evitar clustering prematuro en la búsqueda.

---

### OWN generate_diverse_population (L2)
**Dónde se usa:** `ess/population.py:generate_diverse_population`

**Origen:** CHANGE-004A (unificación de la generación inicial; sustituye a `generate_initial_population`, eliminada).

**Qué hace:** Genera la población inicial diversa dentro de `[x_L, x_U]` mediante muestreo estratificado por frecuencia: los primeros bloques cubren el rango en cuartos y el resto se muestrea con probabilidad inversa a la frecuencia de uso de cada subintervalo (equivalente conceptual a `ssm_diverse`).

**Por qué se usa aquí:** Es la **única** política de generación inicial de eSS tras CHANGE-004A; mejora la cobertura del espacio frente al muestreo uniforme puro.

**Inputs & shapes:**
- `n_points` (int): número de puntos.
- `x_L, x_U` (ndarray): `(dim,)` — bounds.
- `rng` (np.random.Generator, opcional) o `seed` (int, opcional).

**Outputs & shapes:**
- `population` (ndarray): `(n_points, dim)` dentro de bounds.

**Ejemplo mínimo:**
```python
pop = generate_diverse_population(20, [-5, -5], [5, 5], seed=42)
# pop.shape == (20, 2)
```

**Pitfalls:**
- Si `rng=None` y `seed=None`, el resultado no es reproducible.
- Usa 4 subintervalos por dimensión (`freq` de shape `(dim, 4)`); no maneja variables enteras/categóricas.
- La diversificación es independiente por dimensión (no acopla dimensiones entre sí).

**Impacto en eSS/VNS:** Inicialización estándar de eSS; base de la fase previa al RefSet.

---

## 3. Combinación / Generación de soluciones
*(Aún no implementado; placeholder para futuras expansiones como `ssm_combination`, `ssm_crossover`.)*

---

## 4. Evaluación / Bounds / Penalizaciones

### OWN evaluate (L1)
**Dónde se usa:** `ess/utils.py:evaluate`

**Origen:** AI-CHANGE-001

**Qué hace:** Wrapper simple que evalúa una función objetivo f en un punto x.

**Por qué se usa aquí:** Abstractiza la evaluación, permitiendo reemplazar con versiones que registren, penalicen o cacheen sin cambiar el kernel.

**Inputs & shapes:**
- `f` (callable): función objetivo, f(x) → float.
- `x` (ndarray): `(dim,)` — punto de evaluación.

**Outputs & shapes:**
- `float` — valor de f(x).

**Ejemplo mínimo:**
```python
def sphere(x): return np.sum(x**2)
result = evaluate(sphere, np.array([1, 2]))
# result == 5.0
```

**Pitfalls:**
- No valida que x esté dentro de bounds; confía en que el llamador lo garantiza.
- No maneja excepciones (NaN, Inf, timeout); el kernel debe gestionar.
- Wrapper trivial; podría eliminarse si no hay extensiones futuras (registro, caché).

**Impacto en eSS/VNS:** Trivial en versión mínima; importante si futuras versiones añaden penalización, registro o caché.

---

### OWN project_bounds (L1)
**Dónde se usa:** `ess/utils.py:project_bounds`

**Origen:** AI-CHANGE-001 (importado de `problems/utils.py`)

**Qué hace:** Proyecta un punto x al hipercubo [x_L, x_U] clipeando cada componente.

**Por qué se usa aquí:** Asegura factibilidad respecto a bounds; uso futuro en búsqueda local y manejo de constraints.

**Inputs & shapes:**
- `x` (ndarray): `(dim,)`
- `x_L, x_U` (ndarray): `(dim,)`

**Outputs & shapes:**
- `x_proj` (ndarray): `(dim,)` con `x_L[i] <= x_proj[i] <= x_U[i]` para todo i.

**Ejemplo mínimo:**
```python
x = np.array([-10, 5, 12])
x_L = np.array([0, -5, 10])
x_U = np.array([10, 10, 11])
x_proj = project_bounds(x, x_L, x_U)
# x_proj == [0, 5, 11]
```

**Pitfalls:**
- Asume `x_L <= x_U` para todo i; comportamiento indefinido si no se cumple.
- `np.clip` requiere arrays del mismo shape o broadcastables; convertir a arrays explícitamente.
- No en uso aún en `ess_kernel_min` (muestreo ya está dentro de bounds); será crítico si se añade búsqueda local sin garantías.

**Impacto en eSS/VNS:** Factibilidad de bounds; esencial para algoritmos de búsqueda local.

---

## 5. RNG / Muestreo / Inicialización

### OWN proyecto (seed coordination) (L1)
**Dónde se usa:** `ess/main.py:ess_kernel_min`, `ess/population.py:generate_diverse_population`, `tests/test_refset.py`

**Origen:** AI-CHANGE-002 (seed consistency refactor)

**Qué hace:** Coordina el uso de semilla (`seed`) desde un diccionario `opts` compartido, asegurando reproducibilidad en todas las funciones que generan números aleatorios.

**Por qué se usa aquí:** Durante la fase de verificación y confirmación, una seed fija asegura que cambios en el código que mejoran resultados son reales, no aleatorios.

**Patrón:**
```python
seed = opts.get('seed', 42) if opts else 42
rng = np.random.default_rng(seed)
```

**Pitfalls:**
- Si múltiples funciones inicializan RNG con la **misma semilla**, generan **la misma secuencia**; cuidado con solapamientos.
- `opts.get('seed', 42)` requiere que `opts` sea un dict; pasar `None` activa default (42).
- Para futuras expansiones: considerar pasar `rng` compartido (ya inicializado) en lugar de seed, o usar `rng.spawn()` para independencia.

**Impacto en eSS/VNS:** Reproducibilidad durante desarrollo; crítico para validar mejoras y comparar con MATLAB.

---

## 6. Externas clave (NumPy/SciPy)

### EXT np.random.default_rng (L2)
**Dónde se usa:** `ess/main.py:ess_kernel_min`, `ess/population.py:generate_diverse_population`, `tests/test_refset.py`

**Origen:** AI-CHANGE-001

**Qué hace:** Crea un generador de números aleatorios moderno (PCG64) inicializado con una semilla.

**Por qué se usa aquí:** Reemplazo recomendado de `np.random.seed()` global; permite múltiples RNG independientes y es thread-safe para futuras paralelizaciones.

**Inputs & shapes:**
- `seed` (int, opcional): semilla. Si None, usa estado OS (no reproducible).

**Outputs & shapes:**
- `rng` (np.random.Generator): objeto con métodos como `.uniform()`, `.normal()`, etc.

**Ejemplo mínimo:**
```python
rng = np.random.default_rng(42)
x = rng.uniform(0, 1, size=5)  # [0.77, 0.02, 0.48, ...]
```

**Pitfalls:**
- Documentación menciona que "si seed=None, usa estado OS"; es no-reproducible. Siempre pasa seed explícitamente.
- Diferentes versiones de NumPy pueden generar secuencias diferentes; especifica NumPy version si reproducibilidad entre máquinas es crítica.
- Broadcasting: `rng.uniform(x_L, x_U, size=shape)` requiere `x_L` y `x_U` arrays o escalares broadcasteables.

**Impacto en eSS/VNS:** Infraestructura de reproducibilidad; base para todas las operaciones estocásticas.

---

### EXT rng.uniform (L1)
**Dónde se usa:** `ess/main.py:ess_kernel_min`, `ess/population.py:generate_diverse_population`

**Origen:** AI-CHANGE-001

**Qué hace:** Genera muestras uniformes en [low, high) a partir de un RNG.

**Por qué se usa aquí:** Muestreo uniforme aleatorio en el espacio de búsqueda; operación fundamental en eSS.

**Inputs & shapes:**
- `low, high` (ndarray o escalar): límites inferiores y superiores.
- `size` (tuple, opcional): forma del array de salida.

**Outputs & shapes:**
- ndarray de forma `size`, elementos uniformes en [low, high).

**Ejemplo mínimo:**
```python
rng = np.random.default_rng(42)
sample = rng.uniform([0, -5], [1, 5], size=(10, 2))  # (10, 2) en [0, 1] x [-5, 5]
```

**Pitfalls:**
- Extremo superior `high` es **excluido** [low, high); cuidado si necesitas máximo exacto.
- Broadcasting: si `low` y `high` son arrays, `size` debe coincidir o ser compatible (broadcasting rules).
- Dimensiones: `size=(n, d)` genera n puntos de dimensión d; no confundir con (d,) que es un punto.

**Impacto en eSS/VNS:** Muestreo del espacio de búsqueda; determina la exploración inicial.

---

### EXT np.argsort (L1)
**Dónde se usa:** `ess/refset.py:create_refset`

**Origen:** AI-CHANGE-001

**Qué hace:** Retorna índices que ordenarían un array; aquí se usa para ranking por fitness.

**Por qué se usa aquí:** Selecciona los mejores individuos (RefSet1) sin copiar; eficiente y idiomático en NumPy.

**Inputs & shapes:**
- `a` (ndarray): `(n,)` — array de fitness.

**Outputs & shapes:**
- `idx` (ndarray): `(n,)` — índices tales que `a[idx]` está ordenado ascendente.

**Ejemplo mínimo:**
```python
f_pop = np.array([5.0, 2.0, 8.0, 1.0])
idx = np.argsort(f_pop)  # [3, 1, 0, 2]
# Mejores: f_pop[[3, 1]] = [1.0, 2.0]
```

**Pitfalls:**
- Retorna índices en orden **ascendente** (menores fitness primero); para máximos, usa `-a` o `[::-1]`.
- NaN y Inf son ordenados al final; cuidado si hay valores inválidos.
- Estable: índices iguales mantienen orden original (importante si hay ties).

**Impacto en eSS/VNS:** Ranking eficiente de soluciones; base para selección por calidad.

---

### EXT np.linalg.norm (L2)
**Dónde se usa:** `ess/refset.py:select_most_diverse`

**Origen:** AI-CHANGE-001

**Qué hace:** Calcula la norma (magnitud) de un vector; aquí se usa para distancia euclidiana.

**Por qué se usa aquí:** Métrica de diversidad; mide distancia en el espacio de búsqueda.

**Inputs & shapes:**
- `x` (ndarray): vector `(dim,)` o matriz `(n, dim)`.
- `axis` (int, opcional): eje a lo largo del cual sumar (para broadcasting).

**Outputs & shapes:**
- `float` (si `x` es 1D) o ndarray (si `x` es 2D con axis).

**Ejemplo mínimo:**
```python
c = np.array([3, 4])
r = np.array([1, 1])
dist = np.linalg.norm(c - r)  # sqrt((3-1)^2 + (4-1)^2) = 3.6...
```

**Pitfalls:**
- Subtracción `c - r` requiere shapes compatibles; `(dim,)` - `(dim,)` es OK, pero `(dim,)` - `(m, dim)` necesita broadcasting explícito.
- Por defecto, calcula L2 (euclidiana); parámetro `ord` cambia a L1, Inf, etc.
- En `select_most_diverse`, se usa en list comprehension (no vectorizado); considera vectorización futura con `axis` si m >> 1000.

**Impacto en eSS/VNS:** Métrica de diversidad; crucial para balance calidad–diversidad en RefSet.

---

### EXT np.vstack (L1)
**Dónde se usa:** `ess/refset.py:create_refset`

**Origen:** AI-CHANGE-001

**Qué hace:** Apila arrays verticalmente (fila a fila).

**Por qué se usa aquí:** Combina RefSet1 y RefSet2 en un array 2D.

**Inputs & shapes:**
- `(ref1: (r1, dim), ref2: (r2, dim))`

**Outputs & shapes:**
- `(r1 + r2, dim)`

**Ejemplo mínimo:**
```python
ref1 = np.array([[1, 2], [3, 4]])
ref2 = np.array([[5, 6]])
combined = np.vstack([ref1, ref2])  # (3, 2)
```

**Pitfalls:**
- Ambos arrays deben tener el mismo número de columnas; error si shapes incompatibles.
- `vstack` crea copia; no modifica originales.

**Impacto en eSS/VNS:** Trivial; solo composición de RefSet.

---

### EXT np.concatenate (L1)
**Dónde se usa:** `ess/refset.py:create_refset`

**Origen:** AI-CHANGE-001

**Qué hace:** Concatena arrays a lo largo de un eje existente.

**Por qué se usa aquí:** Combina valores de fitness y índices.

**Inputs & shapes:**
- `([f1: (r1,), f2: (r2,)])` → `(r1 + r2,)`

**Outputs & shapes:**
- ndarray concatenado.

**Ejemplo mínimo:**
```python
f1 = np.array([1.0, 2.0])
f2 = np.array([3.0])
f_all = np.concatenate([f1, f2])  # [1.0, 2.0, 3.0]
```

**Pitfalls:**
- Solo 1D o ejes existentes; para agregar dimensión, usa `vstack` o `expand_dims`.

**Impacto en eSS/VNS:** Trivial; composición de traza.

---

### EXT np.delete (L2)
**Dónde se usa:** `ess/refset.py:select_most_diverse`

**Origen:** AI-CHANGE-001

**Qué hace:** Elimina elemento(s) en índice(s) específicos.

**Por qué se usa aquí:** Remueve candidato seleccionado del pool en bucle de diversidad.

**Inputs & shapes:**
- `arr` (ndarray): `(m, dim)`
- `obj` (int o array): índice(s) a eliminar.
- `axis` (int): eje a lo largo del cual eliminar.

**Outputs & shapes:**
- ndarray con elemento(s) removido(s).

**Ejemplo mínimo:**
```python
arr = np.array([1, 2, 3, 4])
result = np.delete(arr, 2)  # [1, 2, 4]
```

**Pitfalls:**
- Crea copia; ineficiente en bucles si m es grande. Considera máscaras booleanas (`arr[mask]`) para grandes operaciones.
- `axis` es crítico en 2D: `axis=0` elimina filas, `axis=1` columnas.
- Índices fuera de rango generan error.

**Impacto en eSS/VNS:** Eficiencia O(n) por eliminación; cuello de botella si se llamó muchas veces en bucle.

---

## Actualizaciones

### AI-CHANGE-002
- Introducido pattern de coordinación de seed en `opts` (ver sección 5).
- Actualizado `generate_initial_population` para aceptar `opts` opcional.
- Tests en `test_refset.py` modificados para aceptar `opts` opcional y usar seed coordinado.
- Aclaración arquitectónica: `generate_initial_population` es versión **mínima** para tests; `generate_diverse_population` es versión **estándar** de eSS (aún no integrada en kernel).

### CHANGE-004A / CHANGE-004B-prep (sincronización de ubicaciones)
- `create_refset` y `select_most_diverse` se movieron de `ess/main.py` a `ess/refset.py`.
- `generate_initial_population` fue **eliminada**; `generate_diverse_population` (`ess/population.py`) es la única política de generación inicial.
- `ess/control.py` y `ess/SAVE.py` fueron eliminados.
- Este ledger se saneó en CHANGE-004B-prep para reflejar el estado real del código.

---

## Notas Finales
- Este ledger se actualiza con cada AI-CHANGE-XXX.
- Para defensa del TFG, usa este documento para explicar decisiones de implementación, trade-offs y relaciones entre componentes.
- Las funciones EXT se documentan solo si son nuevas o críticas; no se listan todas las operaciones NumPy triviales.