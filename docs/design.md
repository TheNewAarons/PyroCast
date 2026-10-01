# Diseño de la interfaz web

> **Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.**

Rediseño de `serving/web/` (plantilla `templates/index.html`, estilos `static/app.css`, JS de presentación `static/app.js`). No cambió la API ni los modelos. Stack intacto: Jinja2 + Leaflet por CDN, sin build step.

## Concepto: consola de monitoreo satelital

Un mapa a pantalla completa y un único panel HUD translúcido a la izquierda. Mucho espacio negativo, líneas de 1 px, esquinas casi rectas. **El ámbar de fuego solo aparece donde hay fuego o un estado activo**: la rampa de probabilidad, las detecciones de FIRMS, el botón de acción y el control de días. Todo lo demás es gris sobre casi negro azulado. Lo memorable es el propio mapa: celdas ámbar sobre una base oscura y apagada; el resto del diseño se mantiene callado.

Decisiones que siguen la guía de la skill `frontend-design` (evitar lo genérico):
- El aviso de investigación es una **banda fija arriba**, parte del layout (no un modal ni un elemento ocultable), con un marcador de rombo neutro. No usa el ámbar: no es un dato de fuego.
- Los errores no usan un segundo color: caja de borde claro con una etiqueta `ERROR` y el código del backend, en el mismo gris-blanco del texto fuerte.
- Las etiquetas en mayúsculas con espaciado amplio se reservan para **metadatos** (secciones del panel); el resto del texto va en minúsculas normales.
- Sin numeración decorativa ("01/02/03"), sin tarjetas idénticas, sin degradados, sin sombras, sin animaciones continuas.
- La lectura de coordenadas bajo el cursor es funcional (la consola siempre dice dónde estás), no decoración.

## Tokens

Todos en `:root` de `static/app.css` (un solo archivo). El JS lee la rampa y el acento desde ahí (`getComputedStyle`), así que las celdas del mapa y la leyenda comparten una única fuente.

| Token | Valor | Uso |
|---|---|---|
| `--bg` | `#070a0f` | fondo (casi negro azulado) |
| `--surface-solid` / `--surface-alpha` | `#090d13` / 0.9 | paneles translúcidos (con `backdrop-filter: blur(14px)`) |
| `--line` | `#1c2733` | separadores decorativos de 1 px |
| `--line-ui` | `#6d7f95` | borde de controles |
| `--text` / `--text-strong` / `--text-dim` | `#d5dce5` / `#f2f5f9` / `#93a0b1` | texto, énfasis, ayudas |
| `--accent` | `#ff8a1f` | fuego y estados activos |
| `--accent-ink` | `#130a02` | texto sobre el acento |
| `--ramp-1` a `--ramp-5` | `#b45410` `#d9691a` `#f4801f` `#ffa245` `#ffd9a8` | probabilidad (5 a 20 %, 20 a 40, 40 a 60, 60 a 80, 80 a 100) |
| `--font-sans` | Jost (400, 500) | títulos y texto |
| `--font-mono` | Martian Mono (300 a 500) | datos, coordenadas, métricas, leyenda |
| `--fs-xs` / `-sm` / `-md` / `-xl` | 11 / 13 / 15 / 24 px | escala tipográfica |
| `--tracking-label` | 0.16em | etiquetas de metadatos |
| `--sp-1` a `--sp-6` | 4, 8, 12, 16, 24, 32 px | espaciado |
| `--radius` | 2px | esquinas |
| `--panel-width` | 340px | ancho del HUD |
| `--t-fast` / `--t-med` | 140ms / 260ms | transiciones |
| `--map-dim` | 0.62 | brillo de las teselas base |

Mapa base: **Esri World Dark Gray Base** (sin clave), atenuado con `brightness(0.62)`. CARTO Dark Matter se descartó: desde 2026 exige clave de API (sin ella sirve teselas con marca de agua). Atribución visible en el mapa; detalle de licencia en `docs/data-sources.md`.

## Accesibilidad (verificada)

- **Contraste WCAG AA por cálculo**: `serving/tests/test_design_tokens.py` calcula las razones desde los tokens reales. Los paneles son translúcidos, así que se evalúan contra el peor fondo posible (el panel compuesto sobre blanco puro, `#22252b`). Resultados mínimos: `--text-dim` sobre ese peor fondo 5.8:1 (AA texto: 4.5), `--text` 11.1:1, `--text-strong` 14.0:1; botón (`--accent-ink` sobre `--accent`) 8.3:1; `--line-ui` sobre el panel 4.8:1 y sobre el peor fondo 3.7:1 (AA elementos de interfaz: 3); cada clase de la rampa >= 3:1 contra la base de mapa atenuada (`#252526`). El test falla si un cambio de tokens rompe AA.
- **Foco visible con teclado**: `:focus-visible` con contorno de 2 px de `--focus` (`#f2f5f9`, 17.8:1 sobre el panel); el deslizador de días responde a las flechas, Inicio y Fin (verificado en la captura 4).
- **`prefers-reduced-motion`**: se anulan todas las animaciones y transiciones.
- **La escala no depende solo del color**: la leyenda muestra el rango numérico de cada clase ("80 a 100 %"), cada celda tiene un tooltip con su porcentaje, y el control de días muestra el día en texto ("3/3 2025-11-13").
- **Responsivo**: a 390 px el HUD es un panel inferior colapsable (botón Panel, `aria-expanded`), los campos de fecha y horizonte van en dos columnas, y no hay scroll horizontal (comprobado con el navegador: `scrollWidth <= innerWidth`).
- Lectores de pantalla: regiones etiquetadas, `role="alert"` para errores, `role="status"` para el estado, `aria-live` en la lectura del punto.

## Microinteracciones

Hover y foco en controles (140 ms), aparición del panel de resultado y del banner de error (260 ms, 4 px), y transición de color/opacidad de las celdas al mover el deslizador de días (la misma duración). Nada se anima solo.

## Skills usadas

- **`frontend-design`** (oficial de Anthropic, marketplace `claude-plugins-official`, instalada con `claude plugin install frontend-design@claude-plugins-official`). Es la única skill instalada para este trabajo (no se instaló ninguna de terceros: la oficial cubre dirección visual, tipografía, restricción y autocrítica, y no hacía falta más). Se leyó su `SKILL.md` completo y se aplicó: planificar tokens antes de construir, contrastar con los valores por defecto "de IA" (el fondo casi negro con un único acento brillante es uno de ellos: aquí es una decisión explícita de la propuesta, así que se sigue, pero se evitaron las etiquetas en mayúsculas sobre cada encabezado, la numeración decorativa y los separadores con puntos medios), gastar la audacia en un solo elemento (el mapa de probabilidad), y verificar con capturas. Nota: el plugin recién instalado no se carga en la misma sesión; se aplicó leyendo su archivo.

## Verificación

Capturas con Chrome real (Playwright con el Chrome del sistema) en `docs/screenshots/`: `escritorio-1-inicio`, `-3-prediccion-dia1`, `-4-prediccion-dia3-foco` (con foco de teclado), `-5-error` (fecha sin clima: el banner muestra el mensaje y el código del backend y conserva la predicción anterior), `movil-1-inicio` y `movil-3-panel-colapsado`. El flujo completo (elegir punto, predecir con datos reales, mover el deslizador, error del backend) se ejecutó en el navegador; la consola solo registró el 422 esperado del caso de error.
