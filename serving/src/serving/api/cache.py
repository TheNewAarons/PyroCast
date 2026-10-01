"""Caches en memoria, thread-safe (los endpoints son síncronos y corren
en el threadpool de Starlette)."""
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable


class LRUCache[V]:
    """Caché LRU con TTL opcional. `ttl_seconds=None` -> sin expiración
    (las predicciones dependen solo de rasters ya procesados en disco)."""

    def __init__(
        self,
        max_size: int,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[Hashable, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> V | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if self._ttl is not None and self._clock() - stored_at > self._ttl:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: Hashable, value: V) -> None:
        with self._lock:
            self._items[key] = (self._clock(), value)
            self._items.move_to_end(key)
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)
