"""Clave de cache para un producto ERA5-Land diario ya agregado: hash de
(rango de fechas, variables solicitadas) — bbox no forma parte de la
clave (se asume el bbox de estudio del proyecto, fijo por defecto; ver
docs/decisions.md)."""
import datetime as dt
import hashlib


def cache_key_for(start: dt.date, end: dt.date, variables: tuple[str, ...]) -> str:
    payload = f"{start.isoformat()}|{end.isoformat()}|{','.join(sorted(variables))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
