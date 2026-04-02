# HNSW - Material auxiliar del video

Este repositorio acompaña a mi video de YouTube sobre **HNSW (Hierarchical Navigable Small World)**.

La idea no es ofrecer una implementación lista para producción (ya hay mil librerías con implementaciones eficientes hechas en C, C++ o Rust), sino un material sencillo para **entender la intuición** detrás de estas estructuras de búsqueda aproximada por vecinos más cercanos.

## Qué incluye

- `nsw.py`: una versión simplificada de un grafo **NSW**, útil para ver la idea base de navegación sobre un grafo.
- `hnsw.py`: una versión simplificada de **HNSW**, añadiendo la parte jerárquica sobre esa misma intuición.

Ambos scripts están pensados con fines **didácticos**. Simplifican varios detalles del algoritmo real para que sea más fácil seguir la lógica general.

Aunque el canal y la explicacion del video estén en español, el código del repositorio está escrito en inglés. Lo hago asé porque programar en inglés es una práctica estándar en la industria y ayuda a que nombres, comentarios y estructuras sean más universales.


## Para qué sirve este repo

Sirve como apoyo al video para quien quiera:

- revisar el ejemplo con calma después de verlo,
- experimentar cambiando parámetros,
- comparar la idea de `NSW` frente a `HNSW`,
- tener una referencia simple de cómo se construye este tipo de estructura.


## Cómo ejecutarlo

Este proyecto usa Python y dependencias como `numpy`, `networkx`, `matplotlib` y `scikit-learn`.

Si quieres lanzar los ejemplos:

```bash
uv run python nsw.py
uv run python hnsw.py
```

## Nota importante

El objetivo de este repositorio es **explicar** HNSW de forma accesible. Si buscas una implementación optimizada y fiel al algoritmo original para uso real, este repo no pretende sustituir librerías especializadas.
