# Calibración isotónica

`models/deep/calibration.py` ajusta las probabilidades crudas de
SmallUNet (P10) contra las frecuencias observadas en el set de
validación, usando regresión isotónica (scikit-learn) -- el enfoque
estándar en la literatura de calibración de redes profundas (no
paramétrico, monótono: corrige cualquier forma de sobreconfianza o
subconfianza sistemática sin asumir una curva particular, a diferencia
de Platt scaling).

**Aviso de honestidad (CLAUDE.md):** Herramienta de investigación. No
usar para decisiones operativas de combate de incendios sin validación
de CONAF/SENAPRED. Los números de esta página son de una corrida real
del pipeline de calibración sobre un checkpoint de FIXTURE sintético
(`pyrocast-calibrate run --fixture`, lo que `make calibrate` ejecuta) --
no de un modelo entrenado sobre NDWS o eventos reales de Chile, que
esta sesión no tuvo forma de descargar ni entrenar (ver
`docs/public-dataset.md`, `docs/model-card.md`). Sirven para probar que
el pipeline funciona de punta a punta, no como una medición de qué tan
bien calibrado está el U-Net real.

## Enfoque

1. Se corre el checkpoint sobre cada muestra del set de validación
   (`models.deep.train.NDWSPretrainDataset`/`ChileFinetuneDataset` --
   el mismo contrato `(entrada, máscara objetivo)` que ya usa el
   entrenamiento), aplicando `sigmoid` a los logits para obtener
   probabilidades crudas por celda.
2. Se ajusta `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip",
   y_min=0.0, y_max=1.0)` contra esos pares (probabilidad cruda,
   resultado real 0/1) -- **sobre el mismo set de validación**, sin un
   split de calibración separado (ver `docs/decisions.md`).
3. Se calculan Brier score y ECE (`models/evaluation/metrics.py`, P8)
   antes y después de calibrar, sobre el mismo set de validación.

## Tabla antes/después (checkpoint de fixture, `make calibrate`)

Corrida real, `uv run --package models pyrocast-calibrate run --fixture`,
en el mismo hardware sin GPU de `docs/model-card.md` (~9.5s de punta a
punta, incluyendo entrenar el checkpoint de fixture):

| Métrica | Antes | Después |
|---|---|---|
| Brier score | 0.2731 | 0.1644 |
| ECE | 0.3249 | 0.0000 |

Menor es mejor en ambas métricas. `n_samples` evaluadas: 1024 (4
muestras de validación sintéticas × 16×16 celdas cada una).

**El ECE de 0.0000 después de calibrar es real, no un error, y
demuestra en vivo el riesgo ya documentado en `docs/decisions.md`**:
al ajustar y evaluar la regresión isotónica sobre el MISMO set de
validación (sin un split de calibración separado), el ECE "después"
es estructuralmente cercano a 0 -- **esto NO es un artefacto de pocas
muestras que mejoraría con más datos**: verificado en la revisión final
del 2026-09-29 con hasta 100.000 muestras sintéticas, el ECE en la
propia muestra de ajuste sigue siendo de orden `1e-17` (machine-zero).
El valor predicho por la regresión isotónica para cada nivel de
probabilidad observado ES, por construcción, el promedio de los
targets de ese mismo nivel en el set de ajuste -- exactamente lo que
ECE con bins mide. Esto significa que **la columna "ECE después" de
esta tabla es una prueba de que el pipeline corre de punta a punta, no
una medición de qué tan bien calibrado queda el modelo en datos
nuevos** -- para eso hace falta evaluar sobre un set separado del que
se usó para ajustar (ver `models/tests/test_calibration.py::test_fit_isotonic_calibrator_improves_ece_on_a_known_miscalibration`,
que sí mide una mejora real usando un split fit/eval, y
`docs/limitations.md`).

## El modelo calibrado implementa `FireSpreadModel` (P8)

`CalibratedUNet` (`models/deep/calibration.py`) implementa
`shared.model_protocol.FireSpreadModel` -- el mismo Protocol que
`CellularAutomatonModel` (P7) ya implementa, con el mismo convenio de
día 0 (el estado conocido del propio evento, no una predicción real;
ver `models/cellular_automata/model.py`) **y prediciendo de forma
autorregresiva** para los días siguientes: cada día se predice
reemplazando el canal `fire_mask` de entrada por la PROPIA predicción
calibrada del modelo para el día anterior (nunca por el `fire_mask`
real observado), igual que `CellularAutomatonModel` evoluciona su
propio estado de fuego desde el día 0. Sin esto, `CalibratedUNet`
sería un pronóstico de un solo paso con acceso a verdad de terreno que
el autómata celular nunca tiene -- corregido en la revisión final del
2026-09-29, ver `docs/decisions.md`. Esto significa que
`models/evaluation/backtest.py` puede correr contra un `CalibratedUNet`
exactamente igual que contra el autómata celular, sin ningún caso
especial por modelo.

## El calibrador viaja con su checkpoint, nunca con otro

Cada calibrador guardado (`<checkpoint>.calibrator.pt` por defecto)
lleva una huella (sha256) del archivo de checkpoint exacto contra el
que se ajustó. Cargarlo contra un checkpoint distinto -- incluso uno
reentrenado con los mismos hiperparámetros, que produce pesos
distintos por la inicialización aleatoria y el orden de los datos --
levanta `IncompatibleCalibratorError` de inmediato, en la construcción
de `CalibratedUNet`, no en el primer `predict()`.

## Limitaciones

Ver `docs/limitations.md`. No se micro-gestionan acá para evitar que
ambas fuentes se desincronicen.
