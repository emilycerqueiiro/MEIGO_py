# SCOPE.md — Alcance del Proyecto TFG “MEIGOpy y Viabilidad de Integración con SELDOMpy”

## 1. Descripción general

Este Trabajo de Fin de Grado tiene como objetivo la **reimplementación en Python del entorno MEIGO**, originalmente desarrollado en Matlab y R, que combina metaheurísticas cooperativas y ajuste bayesiano para la optimización global de modelos no lineales.  
Además, se analizará la **viabilidad técnica de integración con SELDOMpy**, una herramienta en Python para modelado dinámico e identificación de sistemas biológicos.

---

## 2. Componentes principales a implementar

### 2.1 CeSS (Cooperative Enhanced Scatter Search)
- Implementación en Python del algoritmo CeSS.
- Basado en múltiples instancias cooperativas del método eSS.
- Intercambio periódico de información entre RefSets.
- Validación mediante funciones benchmark (Rosenbrock, Rastrigin, Griewank).

### 2.2 cVNS (Cooperative Variable Neighborhood Search)
- Implementación en Python del algoritmo cVNS.
- Búsqueda cooperativa a través de múltiples vecindarios variables.
- Comparativa directa de rendimiento y convergencia frente a CeSS.

### 2.3 BayesFit (versión R de MEIGO)
- Reimplementación del módulo **BayesFit** originalmente desarrollado en R.  
- Adaptación conceptual e implementación en Python utilizando librerías estadísticas modernas (`PyMC`, `emcee`, `scikit-optimize`).
- Enfoque: prior gaussiano, likelihood basada en mínimos cuadrados, y posterior estimada mediante muestreo o aproximación Laplace.
- Integración funcional con CeSS/cVNS para aprovechar soluciones iniciales en la estimación bayesiana.

### 2.4 Integración con SELDOMpy (viabilidad)
- Análisis de la estructura interna de SELDOMpy (modelos, funciones objetivo, gestión de parámetros).
- Diseño de un **prototipo de función puente** `seldom_to_meigo()` que permita traducir modelos SELDOMpy en problemas compatibles con MEIGOpy.
- Informe técnico de viabilidad: beneficios, limitaciones y pasos futuros.

---

## 3. Motivación y justificación

- **Científica:** ampliar la accesibilidad y reproducibilidad del entorno MEIGO trasladándolo a Python, lenguaje estándar en ciencia computacional.
- **Técnica:** unificar optimización global (CeSS/cVNS) y ajuste bayesiano (BayesFit) bajo un framework abierto y extensible.
- **Aplicativa:** facilitar la integración con SELDOMpy, permitiendo cerrar el ciclo de modelado → simulación → optimización → validación.
- **Formativa:** demostrar competencias en ingeniería inversa, optimización global, inferencia bayesiana y desarrollo científico en Python.

---

## 4. Alcance **incluido**

 Implementación modular en Python de:
- CeSS y cVNS funcionales y validados.  
- BayesFit basado en la versión R (enfoque simplificado, prior y likelihood definidos).  
 Integración funcional entre los tres algoritmos bajo una interfaz unificada (`MEIGOpy(problem, opts, algorithm)`).
 Validación en funciones benchmark estándar.
 Informe de **viabilidad técnica** de integración con SELDOMpy (prototipo básico).

---

## 5. Alcance **no incluido**

❌ Replicación exacta bit a bit del comportamiento de MEIGO Matlab/R.  
❌ Integración completa y operativa con SELDOMpy (solo análisis de viabilidad y prototipo).  
❌ Validación experimental con datos biológicos reales.  
❌ Implementación de variantes avanzadas (CeSS híbrido con PSO/GA, Adaptive BayesFit completo).

---

## 6. Cronograma resumido

| Semana | Fase | Objetivo principal |
|---------|------|--------------------|
| 1–2 | Análisis y arquitectura | Entender MEIGO y preparar entorno MEIGOpy |
| 3–4 | Implementación CeSS y eSS base | Núcleo cooperativo funcionando |
| 5 | Implementación BayesFit | Adaptación desde versión R + validación |
| 6 | Integración modular MEIGOpy | CeSS, cVNS, BayesFit unificados |
| 7 | Comparativa Matlab vs Python | Validar equivalencia funcional |
| 8 | Integración SELDOMpy | Prototipo y análisis de viabilidad |
| 9–10 | Redacción y defensa | Documentación técnica y presentación final |

---

## 7. Entregables finales

- Código fuente modular (`src/meigopy/`) con documentación (`README.md`, docstrings).  
- Ejemplos de ejecución reproducibles (`examples/`).  
- Informe técnico de comparación Matlab vs Python.  
- Informe de viabilidad de integración con SELDOMpy.  
- Memoria TFG completa (capítulos 4–12 según índice acordado).  
- Presentación de defensa (diapositivas y guion técnico).  

---

## 8. Criterios de éxito

| Criterio | Descripción | Métrica de validación |
|-----------|--------------|-----------------------|
| **Equivalencia funcional** | CeSS/cVNS reproducen convergencia del MEIGO original | Error relativo < 10% en benchmarks |
| **Operatividad BayesFit** | Genera parámetros y posterior coherente con prior y likelihood | Posteriors bien formados; comparación con MEIGO-R |
| **Integración modular** | CeSS, cVNS y BayesFit pueden ejecutarse desde la misma interfaz | `MEIGOpy(problem, opts, algorithm)` funcional |
| **Interoperabilidad SELDOMpy** | Prototipo capaz de traducir problemas básicos | Ejemplo `seldom_to_meigo()` ejecutable |
| **Calidad técnica** | CI activa, tests unitarios, documentación clara | 100% de tests unitarios pasan, repo limpio |

---

## 🧩 9. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|----------|------------|
| BayesFit (R → Python) más complejo de lo esperado | Alto | Implementar versión simplificada (priors normales + Laplace) y documentar límites |
| Integración paralela CeSS/cVNS genera sobrecoste computacional | Medio | Limitar número de procesos en validaciones iniciales |
| Falta de acceso a código interno de BayesFit original | Alto | Basarse en documentación + replicar comportamiento probabilístico aproximado |
| Plazo de 3 meses ajustado | Alto | Priorizar CeSS/cVNS funcionales + BayesFit simplificado + análisis SELDOMpy conceptual |

---

## 🏁 10. Resultado esperado (versión 0.1)

Un entorno **MEIGOpy funcional y modular** en Python, compuesto por:
- CeSS (Cooperative eSS)
- cVNS (Cooperative VNS)
- BayesFit (versión simplificada en Python)
- Prototipo de comunicación con SELDOMpy  

Validado sobre funciones benchmark y acompañado por documentación, análisis comparativo y defensa técnica rigurosa.

---
