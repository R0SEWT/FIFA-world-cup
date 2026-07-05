# Handoff — Sistema de Predicción e Inteligencia Deportiva Mundial 2026

## 1. Estado actual

El proyecto evolucionó a un predictor híbrido: MLP/LSTM/GRU siguen siendo el backbone exigido por la rúbrica y una capa estadística añade calibración bayesiana, distribución Dixon–Coles de marcadores e incertidumbre posterior para Monte Carlo.

- Rama: `main`.
- Último commit: `f9a365d` (`Add Handoff document for World Cup 2026 prediction system`), publicado en `origin/main`.
- Estado verificado el 04-07-2026; el árbol estaba limpio antes de esta actualización del handoff.
- Python 3.13, TensorFlow 2.21 y PyMC 6.0.1.
- Tests: `16 passed`.
- Dashboard: verificado mediante `streamlit.testing`; cero excepciones.
- Pipeline final: ejecutado de extremo a extremo; genera artefactos v2 con corte `2026-06-27`.
- Auditoría NUTS real: `R-hat máximo = 1.0`, `ESS bulk mínimo = 1408`, 4 cadenas.
- Entrenamiento final de 50,000 iteraciones: **completado**; el modelo seleccionado es `mlp_sgd` y la versión de producción se entrenó durante 19 épocas.
- El notebook editable está actualizado; `Proyecto_Mundial_2026_Ejecutado.ipynb` es anterior a los artefactos finales y todavía debe regenerarse.

## 2. Comandos de reproducción

Instalación y datos:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
python scripts/build_dataset.py
```

Entrenamiento final —50,000 pasos ADVI, 64 muestras posteriores y auditoría NUTS completa—:

```bash
python scripts/train_models.py
```

La forma explícita equivalente es:

```bash
python scripts/train_models.py --bayes-steps 50000
```

Prueba de integración rápida —2 épocas, 200 pasos ADVI y sin NUTS—:

```bash
python scripts/train_models.py --quick
```

Reconstrucción histórica sin usar partidos posteriores al corte:

```bash
python scripts/build_dataset.py --as-of-date YYYY-MM-DD
```

La credencial Kaggle se lee desde `KAGGLE_API_TOKEN` en `.env`. Datos, credenciales y artefactos continúan ignorados por Git; en una copia nueva deben regenerarse.

## 3. Arquitectura actual

- `src/mundial/data.py`: joins as-of, rolling temporal, H2H, plantillas, ventanas y corte `as_of_date`.
- `src/mundial/models.py`: MLP de cuatro capas y encoders LSTM/GRU con tres cabezas multitarea.
- `src/mundial/statistical.py`: temperature scaling bayesiano, Dixon–Coles jerárquico dinámico, ADVI y auditoría NUTS.
- `src/mundial/training.py`: tres folds expansivos, selección por log-loss, test bloqueado, gate de calibración y reentrenamiento de producción.
- `src/mundial/inference.py`: artefactos v2, H2H real, simetría A/B y matrices de marcador 13×13.
- `src/mundial/simulation.py`: muestreo directo de marcadores, muestra posterior común por torneo, prórroga, penaltis e intervalos Monte Carlo.
- `app.py`: predictor, grupos, bracket y campeones con opciones de 100 a 20,000 simulaciones e IC 95%.

Distribución final por partido:

```text
P(marcador)
  = P_DL_calibrada(resultado)
  × P_Dixon-Coles(marcador | resultado)
```

El DL conserva exactamente la masa 1-X-2; Dixon–Coles distribuye esa masa entre marcadores coherentes. El último bucket de cada eje representa 12 o más goles para no perder masa de probabilidad.

Flujo de artefactos:

```text
data/raw/*
  → data/processed/matches.parquet + sequences.npz
  → artifacts/selected_model.keras
  → artifacts/inference_bundle.joblib
  → artifacts/calibration_posterior.joblib
  → artifacts/dixon_coles_posterior.joblib
  → artifacts/artifact_manifest.json
  → Streamlit / simulador
```

El cargador exige artefactos versión 2. Si falta alguno o la versión no coincide, entra en modo demostración de forma explícita.

## 4. Datos y protocolo de evaluación

Dataset actual:

- 11,043 partidos y 294 selecciones.
- 24 features estáticas y dos ventanas `(10, 5)`.
- Train histórico: 4,942 partidos.
- Validación histórica: 2,324 partidos.
- Test bloqueado: 64 partidos del Mundial 2022.
- Estado/producción: 3,713 partidos posteriores.

La selección ya no mira el Mundial 2022. Se usan tres folds expansivos:

1. Validación 2018–2019, entrenamiento anterior a 2018.
2. Validación 2020–2021, entrenamiento anterior a 2020.
3. Validación 2022 hasta el 19-11, entrenamiento anterior a 2022.

El backbone se selecciona por menor log-loss promedio; Brier desempata. Después se evalúa una sola vez sobre Mundial 2022 y se reentrena una versión de producción con todos los partidos disponibles hasta el corte.

## 5. Métricas finales

Estas cifras corresponden a la corrida final de 50,000 pasos ADVI con auditoría NUTS. El modelo seleccionado por log-loss medio fue MLP + SGD.

Backtesting temporal:

| Modelo | Log-loss ↓ | Brier ↓ | Macro F1 |
|---|---:|---:|---:|
| MLP + SGD | **0.8932** | **0.5258** | 0.4337 |
| GRU | 0.8970 | 0.5279 | **0.4711** |
| MLP + Adam | 0.8983 | 0.5286 | 0.4532 |
| LSTM | 0.8992 | 0.5296 | 0.4610 |

Mundial 2022, modelo servido por la corrida final:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.5312 |
| Macro F1 | 0.3928 |
| Log-loss | 1.0048 |
| Brier | 0.5951 |
| ECE | 0.0288 |
| MAE goles A híbrido | 1.0859 |
| MAE goles B híbrido | 0.8753 |

La calibración candidata empeoró log-loss (1.0088 frente a 1.0048), Brier (0.5966 frente a 0.5951) y ECE (0.0817 frente a 0.0288); el gate la rechazó y el artefacto servido usa calibración identidad. Los cambios relativos finales del ELBO fueron 0.000529 para calibración, 0.000740 para Dixon–Coles de evaluación y 0.000540 para Dixon–Coles de producción.

## 6. Cambios importantes respecto al sistema anterior

- El estado exportado incluye el último partido completado; ya no queda retrasado un encuentro.
- H2H real se exporta y usa en inferencia; ya no se reemplaza por ceros.
- Los partidos neutrales se aumentan intercambiando A/B y la inferencia promedia ambas orientaciones. La simetría se verificó con error cero.
- Resultado y marcador ya no se sortean por separado ni se corrigen artificialmente.
- En cada torneo se elige una única muestra posterior para todos sus partidos, conservando incertidumbre paramétrica coherente.
- Las eliminatorias simulan 90 minutos, prórroga con tasas escaladas a un tercio y penaltis si persiste el empate.
- Los intervalos de campeón usan Wilson al 95% y representan error Monte Carlo; no son intervalos deportivos totales.
- El último desempate de grupo sigue siendo aleatorio y reproducible porque no hay datos de fair play.

## 7. Limitaciones y riesgos pendientes

1. **Notebook ejecutado.** Debe ejecutarse otra vez para que tablas, curvas y métricas coincidan con `artifacts/metrics.json`.
2. **Empates como argmax.** El backbone final no elige empates como clase máxima en el test bloqueado, aunque sí asigna masa probabilística y el simulador la utiliza.
3. **Tamaño del test.** Mundial 2022 contiene solo 64 partidos; diferencias pequeñas necesitan bootstrap o intervalos para interpretarse.
4. **Ranking reciente.** Debe comprobarse que las 48 selecciones tengan el snapshot FIFA más reciente disponible.
5. **Plantillas.** FC 24 sigue siendo proxy; no se usan convocatorias oficiales 2026.
6. **Rendimiento.** La incertidumbre posterior aumenta el tiempo del simulador. Se verificó funcionalidad con 2,000 corridas, pero falta un benchmark y posible vectorización antes de usar 10,000/20,000 de forma interactiva.
7. **Cobertura de selecciones.** Aún conviene añadir una prueba de integración que exija ranking, plantilla y secuencia propios para las 48 selecciones.
8. **Promoción de Polymarket.** El cliente y los snapshots funcionan contra Gamma/CLOB reales, pero el peso permanece en cero hasta exportar al menos 30 predicciones OOF temporales para calibración y 20 para evaluación. El gate bloquea deliberadamente el predictor de producción sobre partidos que pudo haber visto.

## 8. Próximos pasos exactos

1. Regenerar `notebooks/Proyecto_Mundial_2026_Ejecutado.ipynb` desde el notebook editable usando los artefactos finales.
2. Comprobar que las tablas y curvas del notebook coincidan con `artifacts/metrics.json`.
3. Verificar manualmente las vistas interactivas del dashboard antes de entregar; la prueba automatizada del 04-07-2026 arrancó la aplicación con artefactos v2 y cero excepciones.
4. Confirmar el snapshot FIFA más reciente y la cobertura propia de ranking, plantilla y secuencia para las 48 selecciones.
5. Ejecutar un benchmark de 10,000/20,000 simulaciones y vectorizar si la latencia interactiva no es aceptable.
6. Grabar el video y preparar la defensa: leakage temporal, BPTT, log-loss/Brier, temperature scaling, Dixon–Coles y diferencia entre incertidumbre posterior y Monte Carlo.
7. Exportar probabilidades OOF expansivas para los partidos con mercado, ejecutar `scripts/evaluate_market_blend.py` y promover Polymarket solo si mejora log-loss y Brier.

## 9. Comprobaciones rápidas

```bash
# Artefactos y métricas
python -m json.tool artifacts/artifact_manifest.json
python -m json.tool artifacts/metrics.json

# Tests
pytest -q

# Dashboard
streamlit run app.py

# GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si aparece “Modo demostración”, comprobar los cinco artefactos v2 enumerados en la sección 3, no solo el `.keras` y el bundle.
