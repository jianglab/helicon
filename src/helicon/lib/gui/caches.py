"""Small bounded caches shared by the display widgets."""

from __future__ import annotations

from collections import OrderedDict


def _default_sizeof(value) -> int:
    """Best-effort size in bytes of a cached value.

    NumPy arrays report ``nbytes``; Qt images/pixmaps report
    ``width * height * depth / 8``; anything else counts as 0 bytes (only the
    entry count limits it).
    """
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    try:
        return int(value.width()) * int(value.height()) * int(value.depth()) // 8
    except Exception:
        return 0


class LRUCache:
    """Dictionary-like cache that evicts the least recently used entries.

    Parameters
    ----------
    max_items : int, optional
        Maximum number of entries kept.
    max_bytes : int, optional
        Maximum total size of the entries, as measured by ``sizeof``. The
        most recent entry is always kept even if it alone exceeds the limit.
    sizeof : callable, optional
        ``sizeof(value) -> int`` returning the size of a value in bytes.
    """

    def __init__(
        self, max_items: int = 4096, max_bytes: int = 256 * 2**20, sizeof=None
    ):
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self._sizeof = sizeof or _default_sizeof
        self._data: OrderedDict = OrderedDict()
        self._sizes: dict = {}
        self.nbytes = 0

    def __contains__(self, key) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key):
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def get(self, key, default=None):
        """Return the cached value for ``key`` (marking it recent) or ``default``."""
        if key in self._data:
            return self[key]
        return default

    def __setitem__(self, key, value) -> None:
        if key in self._data:
            self.nbytes -= self._sizes.pop(key)
            del self._data[key]
        size = max(0, int(self._sizeof(value)))
        self._data[key] = value
        self._sizes[key] = size
        self.nbytes += size
        self._evict()

    def __delitem__(self, key) -> None:
        del self._data[key]
        self.nbytes -= self._sizes.pop(key)

    def _evict(self) -> None:
        """Drop the oldest entries until both limits are met."""
        while len(self._data) > 1 and (
            len(self._data) > self.max_items or self.nbytes > self.max_bytes
        ):
            key, _ = self._data.popitem(last=False)
            self.nbytes -= self._sizes.pop(key)

    def keys(self):
        """Return the cached keys, oldest first."""
        return self._data.keys()

    def clear(self) -> None:
        """Remove every entry."""
        self._data.clear()
        self._sizes.clear()
        self.nbytes = 0
