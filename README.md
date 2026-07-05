# Sistema de prediccion e inteligencia deportiva — Mundial 2026

Proyecto académico que integra MLP/LSTM/GRU con calibración bayesiana y un modelo dinámico Dixon–Coles, simulación Monte Carlo del formato de 48 selecciones y un dashboard Streamlit en español.

## Instalación local

Requiere Python 3.13, Linux y, opcionalmente, una GPU NVIDIA.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si TensorFlow no detecta la GPU, el proyecto funciona en CPU. En Linux, la instalación recomendada por TensorFlow es `tensorflow[and-cuda]`; no instale manualmente otra versión de CUDA dentro del entorno.

## Datos y entrenamiento

1. Descargue su token desde Kaggle y defina `KAGGLE_API_TOKEN=...` en `.env`, o guárdelo como `~/.kaggle/access_token`. También se aceptan credenciales legacy en `~/.kaggle/kaggle.json`.
2. Ejecute el pipeline completo:

```bash
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/train_models.py
```

`build_dataset.py --as-of-date YYYY-MM-DD` permite reconstruir un corte histórico sin fuga. El entrenamiento completo ejecuta tres ventanas temporales, ADVI de 50,000 iteraciones y una auditoría NUTS; puede tardar considerablemente.

Para comprobar la integración con solo dos épocas por modelo:

```bash
python scripts/train_models.py --quick
```

La prueba rápida limita ADVI a 200 pasos y omite NUTS. Para una ejecución personalizada use `--bayes-steps`; `--skip-nuts` queda reservado para diagnóstico, no para el artefacto final.

Los datos crudos, procesados, credenciales y modelos están excluidos de Git. `data/raw/manifest.json` registra fuente, fecha, tamaño y SHA-256 de cada archivo.

## Dashboard

```bash
streamlit run app.py
```

Sin artefactos v2 entrenados, la aplicación arranca en un modo demostración Elo claramente identificado. Después de entrenar, carga el backbone Keras, calibración posterior, Dixon–Coles y manifiesto versionado.

### Señal opcional de Polymarket

```bash
python scripts/fetch_polymarket_history.py
python scripts/generate_polymarket_snapshot.py
python scripts/generate_market_oof.py
python scripts/evaluate_market_blend.py
```

Gamma representa el 1-X-2 como tres contratos binarios; el cliente toma el token `Yes` de victoria A, empate y victoria B, los normaliza y guarda snapshots locales. La combinación log-lineal solo se activa cuando `market_blend.json` fue promovido con probabilidades OOF generadas antes de cada partido. Si faltan esas columnas o la mezcla no mejora log-loss y Brier, el dashboard conserva DL + Bayes y lo indica explícitamente. Nunca se consulta la red durante Monte Carlo.

### Estado vivo del torneo

La actualización tiene dos pasos y nunca reemplaza producción directamente:

```bash
python scripts/fetch_tournament_state.py --url "$MUNDIAL_FIFA_SOURCE_URL"
python scripts/approve_tournament_state.py
```

El primero escribe `artifacts/tournament_state.candidate.json` y valida los 104 partidos, dependencias, ganadores y monotonicidad contra el snapshot aprobado. El segundo crea un backup, promueve atómicamente `tournament_state.json` y refresca Polymarket conservando solo cruces pendientes de 90 minutos. Si FIFA falla o cambia el esquema, producción no se toca y el dashboard muestra que sirve el snapshot anterior.

La simulación aplica resultados reales a los grupos, bloquea cruces oficiales y solo sortea partidos pendientes. Una misma semilla, estado y snapshot de mercado producen el mismo resultado; los eliminados permanecen en 0%.

Al cerrar una fase, el dataset puede reconstruirse con los resultados aprobados (las features se calculan con `shift(1)`, antes del partido):

```bash
python scripts/build_dataset.py --tournament-state artifacts/tournament_state.json --as-of-date 2026-06-27
python scripts/train_models.py --phase groups --terminal-fold-start 2026-06-11 \
  --terminal-fold-end 2026-06-27 --artifacts-dir artifacts/candidate-groups
python scripts/gate_phase_model.py --phase groups --data-cutoff 2026-06-27 \
  --candidate-version wc26-groups-v1 --candidate-metrics candidate_metrics.json \
  --incumbent-metrics incumbent_metrics.json
```

El gate solo promueve si mejoran log-loss y Brier sin degradar ECE más de un punto porcentual; el manifiesto registra fase, corte, métricas y versión de rollback. Un rechazo no interrumpe las actualizaciones de estado ni mercados.

## Notebook y pruebas

El notebook editable es `notebooks/Proyecto_Mundial_2026.ipynb` y la entrega con resultados reales está en `notebooks/Proyecto_Mundial_2026_Ejecutado.ipynb`. En Colab use `requirements-colab.txt`; TensorFlow ya viene incluido en el runtime con GPU.

```bash
pytest -q
jupyter lab
```

## Decisiones y limitaciones

- El test bloqueado es el Mundial 2022; ningún partido de ese torneo entra en entrenamiento o validación.
- El backbone se selecciona por log-loss en tres cortes expansivos; Brier desempata y macro F1 se conserva para la rúbrica.
- Las probabilidades de resultado proceden del DL calibrado y Dixon–Coles distribuye esa masa entre marcadores coherentes.
- Cada torneo usa una muestra posterior común; los porcentajes incluyen intervalos Monte Carlo del 95%.
- FC 24 funciona como proxy de las plantillas 2026. Las habilidades no aplicables a porteros ignoran valores ausentes.
- Cuando una selección no tiene plantilla completa se usa la mediana de la edición y se activa `players_imputed`.
- El desempate implementa puntos, diferencia de gol, goles y enfrentamiento directo. Fair play no está disponible y se reemplaza por un sorteo reproducible.
- La ronda de 32 conserva los cruces y grupos de terceros permitidos por el calendario FIFA 2026.
- Polymarket utiliza exclusivamente mercados de 90 minutos; cruces sin cotización válida vuelven automáticamente al modelo base.

## Guion sugerido para el video de dos minutos

- 0:00–0:20: objetivo y modelo seleccionado.
- 0:20–0:40: predictor entre dos selecciones.
- 0:40–1:10: editar un grupo y mostrar la tabla recalculada.
- 1:10–1:35: recorrer el bracket y forzar un ganador.
- 1:35–1:55: mostrar el top 10 y campeones históricos.
- 1:55–2:00: cerrar con métricas sobre Mundial 2022.
