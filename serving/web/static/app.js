// Mapa de PyroCast: JS plano + Leaflet (CDN), sin build step.
// Todo el estado vive en el objeto `state`. El texto que viene del backend
// se escribe siempre con textContent (nunca como HTML).
(function () {
  "use strict";

  var root = document.getElementById("app");
  var cfg = {
    predictUrl: root.dataset.predictUrl,
    activeFiresUrl: root.dataset.activeFiresUrl,
    bbox: root.dataset.studyBbox.split(",").map(Number), // [oeste, sur, este, norte]
    defaultDate: root.dataset.defaultDate,
    defaultHorizon: Number(root.dataset.defaultHorizon),
  };

  // La escala de color vive en app.css (:root, --ramp-1..5): una sola fuente para
  // las celdas del mapa y la leyenda. Por debajo de MIN_PROB la celda no se pinta.
  var css = getComputedStyle(document.documentElement);
  function token(name, fallback) {
    return (css.getPropertyValue(name) || "").trim() || fallback;
  }
  var MIN_PROB = 0.05;
  var BINS = [
    { min: 0.8, color: token("--ramp-5", "#ffd9a8"), label: "80 a 100 %" },
    { min: 0.6, color: token("--ramp-4", "#ffa245"), label: "60 a 80 %" },
    { min: 0.4, color: token("--ramp-3", "#f4801f"), label: "40 a 60 %" },
    { min: 0.2, color: token("--ramp-2", "#d9691a"), label: "20 a 40 %" },
    { min: MIN_PROB, color: token("--ramp-1", "#b45410"), label: "5 a 20 %" },
  ];
  var ACCENT = token("--accent", "#ff8a1f");
  var INK = token("--text-strong", "#f2f5f9");
  var LINE = token("--line-ui", "#6d7f95");

  var state = {
    point: null,        // {lat, lon}
    result: null,       // última respuesta de /predict
    dayIndex: 0,        // 0-based sobre result.days
    pointMarker: null,
    predictionLayer: null,
    firesLayer: null,
  };

  var el = {
    date: document.getElementById("date"),
    horizon: document.getElementById("horizon"),
    button: document.getElementById("predict-button"),
    pointText: document.getElementById("point-text"),
    banner: document.getElementById("error-banner"),
    status: document.getElementById("status"),
    firesStatus: document.getElementById("fires-status"),
    result: document.getElementById("result"),
    slider: document.getElementById("day-slider"),
    dayLabel: document.getElementById("day-label"),
    modelNote: document.getElementById("model-note"),
    warnings: document.getElementById("warnings"),
    legend: document.getElementById("legend-list"),
  };

  // ---------- mapa ----------
  var west = cfg.bbox[0], south = cfg.bbox[1], east = cfg.bbox[2], north = cfg.bbox[3];
  var map = L.map("map", {
    maxBounds: [[south - 2, west - 2], [north + 2, east + 2]], zoomControl: false,
  });
  L.control.zoom({ position: "topright" }).addTo(map);
  // Base oscura y desaturada de Esri (sin clave); atribución exigida por el proveedor.
  L.tileLayer(
    "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    {
      maxNativeZoom: 16, maxZoom: 17,
      attribution: "Tiles © Esri, HERE, Garmin, © OpenStreetMap contributors, " +
        "and the GIS user community",
    }
  ).addTo(map);
  var studyArea = L.rectangle([[south, west], [north, east]], {
    color: LINE, weight: 1, fill: false, dashArray: "4 6", interactive: false,
  }).addTo(map);
  map.fitBounds(studyArea.getBounds());

  // lectura de coordenadas bajo el cursor (consola)
  var cursor = document.getElementById("cursor-readout");
  map.on("mousemove", function (ev) {
    cursor.textContent = "LAT " + ev.latlng.lat.toFixed(4) + "  LON " + ev.latlng.lng.toFixed(4);
  });
  map.on("mouseout", function () { cursor.textContent = "LAT --  LON --"; });

  // panel colapsable (útil sobre todo en móvil)
  var hud = document.getElementById("hud");
  var hudToggle = document.getElementById("hud-toggle");
  function setHudCollapsed(collapsed) {
    hud.dataset.collapsed = collapsed ? "true" : "false";
    hudToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  }
  hudToggle.addEventListener("click", function () {
    setHudCollapsed(hud.dataset.collapsed !== "true");
  });
  setHudCollapsed(false);

  // ---------- leyenda ----------
  BINS.forEach(function (bin) {
    var li = document.createElement("li");
    var swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = bin.color;
    li.appendChild(swatch);
    li.appendChild(document.createTextNode(bin.label));
    el.legend.appendChild(li);
  });
  var none = document.createElement("li");
  var noneSwatch = document.createElement("span");
  noneSwatch.className = "swatch";
  none.appendChild(noneSwatch);
  none.appendChild(document.createTextNode("menos de 5 %, sin color"));
  el.legend.appendChild(none);

  // ---------- errores ----------
  function showError(title, detail, code) {
    el.banner.replaceChildren();
    var strong = document.createElement("strong");
    strong.textContent = title;
    el.banner.appendChild(strong);
    if (detail) {
      var p = document.createElement("p");
      p.textContent = detail;
      el.banner.appendChild(p);
    }
    if (code) {
      var c = document.createElement("span");
      c.className = "code";
      c.textContent = "código: " + code;
      el.banner.appendChild(c);
    }
    el.banner.hidden = false;
  }

  function clearError() {
    el.banner.hidden = true;
    el.banner.replaceChildren();
  }

  // Interpreta ambos formatos de error: el propio ({error: {code, message,
  // details}}) y el de validación estándar de FastAPI ({detail: [...]}).
  async function describeFailure(response) {
    var body = null;
    try { body = await response.json(); } catch (e) { /* cuerpo no JSON */ }
    if (body && body.error) {
      return { title: "La predicción no se pudo calcular", detail: body.error.message,
               code: body.error.code };
    }
    if (body && Array.isArray(body.detail)) {
      var msgs = body.detail.map(function (d) {
        return (d.loc || []).slice(1).join(".") + ": " + d.msg;
      });
      return { title: "Datos de entrada inválidos", detail: msgs.join("; "),
               code: "validation_error" };
    }
    return { title: "Error inesperado del servidor",
             detail: "HTTP " + response.status + " " + response.statusText +
                     ". No se muestra ninguna predicción.", code: "http_" + response.status };
  }

  // ---------- incendios activos ----------
  async function loadActiveFires() {
    try {
      var response = await fetch(cfg.activeFiresUrl + "?days=2");
      if (!response.ok) {
        var f = await describeFailure(response);
        el.firesStatus.textContent = "Incendios activos no disponibles: " + f.detail;
        return;
      }
      var data = await response.json();
      state.firesLayer = L.geoJSON(data, {
        pointToLayer: function (feature, latlng) {
          return L.circleMarker(latlng, {
            radius: 5, color: ACCENT, weight: 2, fillColor: "#000", fillOpacity: 0.6,
          });
        },
        onEachFeature: function (feature, layer) {
          layer.bindTooltip("Detección " + feature.properties.detected_at);
        },
      }).addTo(map);
      el.firesStatus.textContent = data.features.length +
        " detección(es) activa(s) en los últimos 2 días (" + data.source +
        (data.cached ? ", en caché" : "") + ").";
    } catch (e) {
      el.firesStatus.textContent = "Incendios activos no disponibles: no se pudo contactar la API.";
    }
  }

  // ---------- selección de punto ----------
  map.on("click", function (ev) {
    state.point = { lat: Number(ev.latlng.lat.toFixed(5)), lon: Number(ev.latlng.lng.toFixed(5)) };
    if (state.pointMarker) { state.pointMarker.remove(); }
    state.pointMarker = L.circleMarker(ev.latlng, {
      radius: 7, color: INK, weight: 1.5, fillColor: INK, fillOpacity: 0.25,
    }).addTo(map);
    el.pointText.textContent = "Latitud " + state.point.lat + ", longitud " + state.point.lon;
    el.button.disabled = false;
  });

  // ---------- predicción ----------
  function cellStyle(feature) {
    var p = feature.properties.probability_by_day[state.dayIndex];
    for (var i = 0; i < BINS.length; i++) {
      if (p >= BINS[i].min) {
        return { stroke: false, fillColor: BINS[i].color, fillOpacity: 0.72, className: "pc-cell" };
      }
    }
    return { stroke: false, fillOpacity: 0, className: "pc-cell" };
  }

  function showDay(index) {
    state.dayIndex = index;
    var d = state.result.days[index];
    el.dayLabel.textContent = d.day + "/" + state.result.days.length + "  " + d.date;
    if (state.predictionLayer) { state.predictionLayer.setStyle(cellStyle); }
  }

  function renderResult(data) {
    state.result = data;
    if (state.predictionLayer) { state.predictionLayer.remove(); }
    state.predictionLayer = L.geoJSON(data, {
      style: cellStyle,
      onEachFeature: function (feature, layer) {
        layer.on("mouseover", function () {
          var p = feature.properties.probability_by_day[state.dayIndex];
          layer.bindTooltip((p * 100).toFixed(1) + " %", { sticky: true }).openTooltip();
        });
      },
    }).addTo(map);

    el.slider.min = 1;
    el.slider.max = data.days.length;
    el.slider.value = 1;
    showDay(0);

    var b = data.grid.bbox_wgs84; // [oeste, sur, este, norte]
    map.fitBounds([[b[1], b[0]], [b[3], b[2]]], { maxZoom: 14 });

    el.modelNote.textContent = "Modelo: " + data.model.name +
      (data.model.calibrated ? " (calibrado)." : " (sin calibrar: las probabilidades son un " +
        "puntaje relativo, no una probabilidad calibrada).") +
      (data.cached ? " Resultado en caché." : "");
    el.warnings.replaceChildren();
    data.warnings.forEach(function (w) {
      var li = document.createElement("li");
      li.textContent = w;
      el.warnings.appendChild(li);
    });
    el.result.hidden = false;
  }

  async function predict() {
    if (!state.point) { return; }
    clearError();
    el.button.disabled = true;
    el.status.textContent = "Calculando…";
    var payload = {
      lat: state.point.lat, lon: state.point.lon, date: el.date.value,
      horizon_days: Number(el.horizon.value),
    };
    try {
      var response = await fetch(cfg.predictUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        var f = await describeFailure(response);
        showError(f.title, f.detail, f.code);
        el.status.textContent = "";
        return; // se conserva la predicción anterior, no se inventa una nueva
      }
      renderResult(await response.json());
      el.status.textContent = "Listo.";
    } catch (e) {
      showError("No se pudo contactar la API", "Revisa que el servidor esté en marcha.",
                "network_error");
      el.status.textContent = "";
    } finally {
      el.button.disabled = false;
    }
  }

  el.button.addEventListener("click", predict);
  el.slider.addEventListener("input", function () { showDay(Number(el.slider.value) - 1); });
  if (!el.date.value && cfg.defaultDate) { el.date.value = cfg.defaultDate; }
  loadActiveFires();
})();
