# Ficha del modelo: SmallUNet

> **Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.**

`models/deep/unet.py` + `models/deep/train.py` implementan el segundo
modelo del proyecto (P9-P11) -- una U-Net 2D pequeña, complementaria al
autómata celular de P7 (`docs/cellular-automata.md`), entrenada primero
sobre un dataset público de EE.UU. (preentrenamiento, P9,
`docs/public-dataset.md`) y luego afinada sobre eventos reales de Chile
(fine-tuning, P6).

**Aviso de honestidad (CLAUDE.md):** Herramienta de investigación. No
usar para decisiones operativas de combate de incendios sin validación
de CONAF/SENAPRED. Los tiempos de entrenamiento de este documento son
del hardware de desarrollo real (ver abajo), no de un servidor con GPU
-- cualquiera que reproduzca este proyecto en una laptop similar debe
esperar tiempos del mismo orden.

## Hardware de desarrollo

Sin GPU CUDA. MacBook Air, Apple M1, 8 GB RAM.
`torch.cuda.is_available()` -> `False`.
`torch.backends.mps.is_available()` -> `True` (acelerador Metal de
Apple) -- el código elige `cuda` -> `mps` -> `cpu` en ese orden en
tiempo de ejecución, pero CADA decisión de tamaño de este documento
(ancho del modelo, batch size, resolución) se tomó asumiendo que solo
hay CPU disponible: 8 GB de memoria unificada compartida con el resto
del sistema, y MPS en un M1 base es un acelerador modesto, no una GPU
de datacenter.

## Arquitectura: SmallUNet

U-Net 2D, GroupNorm (no BatchNorm -- ver docs/decisions.md, entrena con
batch_size=1 por defecto). Profundidad configurable, 3 niveles por
defecto:

| Etapa | Canales |
|---|---|
| Entrada | 11 (`features.dataset.assemble.CHANNEL_ORDER`) |
| `in_conv` | 16 |
| Down 1 | 32 |
| Down 2 | 64 |
| Down 3 (bottleneck) | 128 |
| Up (simétrico) | 64 -> 32 -> 16 |
| Salida (1x1 conv) | 1 (logit) |

Acepta cualquier alto/ancho (no solo potencias de 2) -- el camino de
subida rellena cada mapa subido al tamaño exacto de su conexión de
salto antes de concatenar (verificado con una entrada de 37x29, ver
`models/tests/test_unet.py`).

Salida: LOGITS, no probabilidades -- aplicar `sigmoid` explícitamente
para obtener probabilidad de fuego por celda (ver
`shared/model_protocol.py`: un futuro wrapper `predict()` que
implemente `FireSpreadModel` haría esa conversión; no está construido
en este plan, ver `docs/limitations.md`).

## Función de pérdida: Focal Loss

`alpha=0.8, gamma=2.0` (defaults del paper original, Lin et al. 2017).
Elegida sobre BCE ponderada porque además de corregir el desbalance de
FRECUENCIA de clase (la clase "fuego" es rara), el término de focusing
`(1-p_t)^gamma` baja la pérdida de cualquier predicción ya confiada y
correcta -- la mayoría de los píxeles de una máscara de fuego real son
negativos triviales que el modelo aprende a predecir rápido; BCE
ponderada seguiría acumulando pérdida sobre ellos, diluyendo el
gradiente útil de los píxeles difíciles (el borde del frente de fuego,
los pocos positivos reales). Ver `models/deep/losses.py` para la
fórmula exacta y `models/tests/test_losses.py` para la propiedad
verificada (`test_focal_loss_down_weights_easy_negatives_more_than_plain_bce`).

## Fases de entrenamiento

### 1. Preentrenamiento (`pyrocast-train pretrain`)

Sobre shards TFRecord de Next Day Wildfire Spread ya descargados
manualmente (`docs/public-dataset.md`) -- desde cero, sin checkpoint
previo. `batch_size=1` por defecto (ver "Tamaño de batch" abajo).

```
pyrocast-train pretrain --shard-dir <ruta a los .tfrecord(.gz)>
```

**Uso de memoria, medido, no estimado**: cada muestra NDWS de 64x64
pesa ~214 KB una vez cargada en `PublicDatasetSample` (medido con
`tracemalloc` sobre 500 muestras sintéticas de esa forma). Los 18.545
chips oficiales completos pesan **~4.1 GB** solo en muestras -- antes
del modelo, el optimizador, las activaciones y matplotlib, en una
máquina de 8 GB sin GPU (ver "Hardware de desarrollo" arriba).
`pretrain` acepta `--max-samples N` para topear cuántas muestras se
cargan en memoria por split (train y val, cada uno hasta N) -- sin
tope por defecto, para no imponer un límite arbitrario a quien tenga
más RAM.

### 2. Fine-tuning (`pyrocast-train finetune`)

Sobre el split train/val de eventos de Chile (`features/dataset/`,
`docs/dataset-card.md`), partiendo de un checkpoint de preentrenamiento.

```
pyrocast-train finetune --pretrained-checkpoint runs/pretrain/best.pt
```

**Decisión: no se congela ninguna capa.** Se usa un learning rate más
bajo (`lr_preentrenamiento * 0.1` por defecto) en vez de congelar el
encoder. Razón: NDWS es 1 km de resolución sobre EE.UU. continental;
los tensores propios de PyroCast son 250 m sobre Chile, con una fuente
de DEM distinta y una distribución de viento/humedad/vegetación
distinta. Congelar el encoder asumiría que sus filtros de bajo nivel
transfieren directamente a través de ese cambio de resolución y
geografía -- una suposición más fuerte que la que este proyecto está
dispuesto a hacer sin evidencia empírica en ningún sentido. Dejar que
toda la red se adapte, con pasos más pequeños, es la opción más segura
dado el desajuste de dominio conocido. Esta es una decisión de diseño,
no la única válida -- ver `docs/decisions.md` y `docs/limitations.md`
(no está validada empíricamente contra congelar el encoder).

## Tamaño de batch

`batch_size=1` por defecto, en ambas fases. Los eventos reales de Chile
tienen el tamaño de su propio bbox (ninguno es igual a otro) -- batir
más de un evento a la vez requeriría relleno/recorte a un tamaño común,
no construido en este plan (YAGNI: el número esperado de eventos reales
de Chile es pequeño, batch_size=1 no es un problema de velocidad
significativo ahí). Configurable por CLI para quien tenga más datos o
más RAM.

## Tiempo de entrenamiento

`pyrocast-train smoke-test` (datos sintéticos, `base_channels=8,
depth=2`, 16x16, 4 muestras de train + 2 de val, 1 época) tardó
**~8.4 segundos** (medido con `time`, llamada directa al CLI, máquina
en reposo) en este hardware. Vía `make train` (incluye el overhead de
resolución de `uv`), entre **~15 y ~20 segundos** en corridas aisladas
-- hasta ~51 segundos medido una vez con la máquina bajo carga
(ejecutando la suite de tests completa + mypy justo antes en la misma
sesión de shell). Ambos números son reales, medidos en esta sesión, no
estimados -- se reporta el rango en vez de solo el mejor caso.

Para una corrida real de preentrenamiento sobre NDWS (18.545 chips
oficiales de 64x64, `base_channels=16, depth=3`) **no se midió en esta
sesión** -- este entorno no tiene acceso de red para descargar NDWS.
Estimado por orden de magnitud a partir del tiempo del smoke test
(escalando por tamaño de imagen —64x64 tiene 16x más píxeles que 16x16—
y por la cantidad de muestras —miles, no 6—, sin benchmarking real):
del orden de horas por época en CPU/MPS, no minutos -- entrenar la
cantidad de épocas que `patience` normalmente permite probablemente
toma **muchas horas a un día** en este hardware. Esta es una
estimación, no una medición -- CLAUDE.md exige no ocultar esta
incertidumbre.

## Reproducibilidad

Cada checkpoint (`models/deep/checkpoint.py::TrainingConfig`) guarda:
arquitectura completa (`in_channels, base_channels, depth`),
hiperparámetros de entrenamiento (`lr, batch_size, seed, focal_alpha,
focal_gamma, max_epochs, patience`), fase, y las rutas de datos
exactas usadas. Cargar un checkpoint reconstruye el modelo desde su
PROPIA configuración, nunca desde los defaults del comando que lo
carga -- `finetune` no necesita (ni debe) adivinar la arquitectura de
un checkpoint de `pretrain` (verificado en
`models/tests/test_checkpoint.py`).

## Limitaciones

Ver `docs/limitations.md` para la lista completa. No se micro-gestionan
acá para evitar que ambas fuentes se desincronicen.
