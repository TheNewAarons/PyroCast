# Dataset público de preentrenamiento: Next Day Wildfire Spread

> **Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.**

`models/deep/public_dataset.py` carga este dataset externo como fuente
de preentrenamiento para el futuro U-Net (P9-P11) — el dataset propio
de PyroCast (P6, Chile) es demasiado pequeño por sí solo para entrenar
una red profunda desde cero.

**Aviso de honestidad (CLAUDE.md):** este documento describe un dataset
de OTRO PAÍS (Estados Unidos continental) usado únicamente para
preentrenamiento. Ningún resultado reportado sobre incendios chilenos
debe mezclar métricas de este dataset con las de `docs/results.md` sin
decirlo explícitamente.

## Qué es

"Next Day Wildfire Spread" (NDWS), Huot, Hu, Goyal, Sankar, Ihme y
Chen, "Next Day Wildfire Spread: A Machine Learning Data Set to Predict
Wildfire Spreading from Remote-Sensing Data", IEEE Transactions on
Geoscience and Remote Sensing / ICDM 2022 (preprint:
[arXiv:2112.02447](https://arxiv.org/abs/2112.02447)). ~18.545 recortes
("chips") de 64×64 píxeles a 1 km de resolución sobre incendios reales
de EE.UU. continental (2012-2020), cada uno con 12 variables de entrada
del día `t` y la máscara de fuego del día `t+1` como etiqueta.

## Cómo obtenerlo (no redistribuido en este repo)

1. Crear una cuenta gratuita en [Kaggle](https://www.kaggle.com/).
2. Ir a [kaggle.com/datasets/fantineh/next-day-wildfire-spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
   y aceptar los términos del dataset (Kaggle exige esto antes de
   permitir la descarga — no hay forma de automatizarlo, ni debería
   automatizarse).
3. Descargar con la Kaggle CLI (`pip install kaggle`, configurar
   `~/.kaggle/kaggle.json` con tu API token de Kaggle):
   ```
   kaggle datasets download -d fantineh/next-day-wildfire-spread -p <dónde-quieras>
   unzip <dónde-quieras>/next-day-wildfire-spread.zip -d <dónde-quieras>
   ```
4. Apuntar `models/deep/public_dataset.py` a los archivos `.tfrecord`
   (o `.tfrecord.gz`) descomprimidos — el loader recibe una lista de
   rutas o un patrón glob explícito, nunca asume un nombre de archivo
   ni construye una URL por su cuenta.

**Nota de incertidumbre, honesta a propósito**: el nombre EXACTO de los
archivos dentro del zip de Kaggle no se pudo confirmar con
certeza total al escribir este documento (la página de Kaggle requiere
una sesión de navegador autenticada que no se pudo inspeccionar
directamente) — puede variar según cuándo se descargue. El formato
TFRecord en sí (confirmado contra el código de exportación oficial de
los autores) y el esquema de features (confirmado contra su
`constants.py`) son estables independientemente del nombre de archivo
exacto. Si el archivo descargado NO es TFRecord (algunas re-subidas de
terceros en Kaggle ofrecen versiones pre-convertidas a `.npy`/`.npz`),
`models/deep/public_dataset.py` no lo soporta — habría que agregar un
loader separado para ese formato, no asumirlo aquí.

**Código de exportación original** (Google Earth Engine, para
reconstruir o actualizar el dataset desde cero, no necesario para usar
el dataset ya publicado): [google-research/google-research/simulation_research/next_day_wildfire_spread](https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread)

## Licencia

**CC BY 4.0** (Creative Commons Attribution 4.0). Permite uso,
modificación y redistribución con atribución — pero este proyecto NO
redistribuye el dataset (ver sección anterior): cada usuario lo
descarga por su cuenta directamente desde Kaggle.

## Formato

TFRecord, posiblemente comprimido con gzip (`.tfrecord.gz` — soportado:
`models/deep/tfrecord_reader.py` detecta la compresión por los magic
bytes del archivo, no por su nombre, así que funciona sin importar cómo
el usuario haya nombrado el archivo descargado). El formato es el
contenedor binario de TensorFlow — cada archivo es una secuencia de
registros `longitud + CRC32C enmascarado + datos + CRC32C enmascarado`,
donde `datos` es un mensaje protobuf `tf.train.Example` serializado.
`models/deep/tfrecord_reader.py` implementa un lector de este formato
SIN depender de `tensorflow` ni del paquete `tfrecord` de PyPI (ver
`docs/decisions.md`) — ver ese módulo para el detalle exacto del
parseo.

**Qué tan verificado está el lector, en tres niveles honestos** (no
todo "verificado" significa lo mismo):
1. **Contra la especificación pública del formato**: el polinomio
   CRC32C y la fórmula de máscara de TFRecord están verificados contra
   el vector de control estándar de la especificación (no solo contra
   el propio encoder de test, que duplica la misma implementación y
   por eso no podría detectar un polinomio o una máscara incorrectos
   por sí solo).
2. **Autoconsistente por round-trip**: el anidamiento protobuf
   (`Example` → `Features` → `Feature` → `FloatList`) se verifica
   codificando y decodificando con un encoder propio en los tests.
3. **Sin probar**: un archivo TFRecord real descargado de Kaggle — este
   entorno no tiene acceso de red para obtener uno. Ver
   `docs/limitations.md`.

Cada `Example` trae 13 features, cada una un `float_list` empaquetado
de 4096 valores (64×64 aplanado, reconstruido en ese orden por
`tfrecord_reader.py`):

| Feature NDWS | Significado | Fuente | Unidad |
|---|---|---|---|
| `elevation` | Elevación | SRTM | metros |
| `pdsi` | Índice de sequía Palmer | GRIDMET | índice adimensional |
| `NDVI` | Vegetación (escalado ×10000) | VIIRS VNP13A1 | adimensional (÷10000 → [-1,1]) |
| `pr` | Precipitación | GRIDMET | mm |
| `sph` | Humedad específica | GRIDMET | kg/kg |
| `th` | Dirección del viento (desde dónde sopla) | GRIDMET | grados |
| `tmmn` | Temperatura mínima diaria | GRIDMET | Kelvin |
| `tmmx` | Temperatura máxima diaria | GRIDMET | Kelvin |
| `vs` | Velocidad del viento | GRIDMET | m/s |
| `erc` | Energy Release Component (peligro de incendio) | NFDRS | índice (BTU/pie²) |
| `population` | Densidad poblacional | GPWv4 | personas/km² |
| `PrevFireMask` | Máscara de fuego día `t` | MOD14A1 V6 | {-1, 0, 1} |
| `FireMask` (**etiqueta**, no una de las 12 de entrada) | Máscara de fuego día `t+1` | MOD14A1 V6 | {-1, 0, 1} |

Estadísticas de normalización oficiales (min/max de recorte, media,
desviación estándar por canal) están publicadas en el `constants.py`
de los autores, pero **no se aplican en este adaptador** — ver
`docs/decisions.md`: los tensores propios de PyroCast (P6) guardan
valores físicos crudos, sin normalizar, y este adaptador iguala esa
convención en vez de la de NDWS.

## Split oficial del paper (no el que usa este proyecto)

14.979 train / 1.877 val / 1.689 test, separado por semanas
(2012-2020) en proporción 8:1:1. `models/deep/public_dataset.py::split_public_dataset`
NO reutiliza este split — genera su propio split determinista
train/val sobre los archivos que el usuario efectivamente tenga en
disco, separado por completo del split de eventos de Chile
(`features/dataset/split.py`) para que el fine-tuning posterior no
mezcle ambos splits de forma confusa.

## Cómo difiere del esquema de PyroCast (P6)

`features/dataset/assemble.py::CHANNEL_ORDER` tiene 11 canales:
`elevation, slope_deg, aspect_deg, wind_u, wind_v, temperature,
relative_humidity, precipitation, ndvi, fuel_type, fire_mask`. NDWS no
tiene el mismo esquema — la tabla completa de mapeo (qué se copia
directo, qué se deriva, qué canal de PyroCast queda relleno sin
equivalente, y las fórmulas exactas) está documentada en el docstring
de `models/deep/public_dataset.py`, no duplicada aquí para evitar que
ambas fuentes se desincronicen. Resumen:

- **Copiado directo** (misma magnitud, misma unidad): `elevation`.
- **Derivado con una fórmula de PyroCast ya existente**: `slope_deg`,
  `aspect_deg` (Horn 1981, desde la elevación de NDWS).
- **Derivado con conversión de unidad/convención**: `wind_u`/`wind_v`
  (dirección+velocidad → componentes), `temperature` (promedio
  min/max), `relative_humidity` (humedad específica → relativa,
  aproximada), `precipitation` (mm → m), `ndvi` (÷10000).
- **Sin equivalente, relleno explícito**: `fuel_type` (NDWS no tiene
  ningún canal de cobertura de suelo/vegetación categórica — se rellena
  con el código "desconocido" de PyroCast, código 99, nunca inventado
  desde NDVI).
- **Sin canal correspondiente en el tensor de PyroCast en absoluto**:
  la etiqueta `FireMask` (día `t+1`) de NDWS — P6 no tiene concepto de
  "etiqueta" dentro del tensor; se devuelve por separado.
- **De NDWS, sin uso en absoluto**: `pdsi`, `erc`, `population` — no
  tienen equivalente y no se usan para derivar nada (documentado, no
  descartado en silencio).
