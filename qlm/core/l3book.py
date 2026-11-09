"""L3 (market-by-order) book.

This is the layer that L2 physically cannot give you. L2 aggregates every
order at a price into one number, so a 400-lot level could be one institution
or eighty retail orders -- and those two states behave completely differently
when the level gets hit. L3 keeps individual order IDs, which buys us three
things institutions actually use:

  1. Queue position      -- where our passive order sits, hence fill probability.
  2. Iceberg detection   -- a level that refills from the SAME price with a
                            near-identical clip size after each match is a
                            reserve/iceberg order, i.e. a large parent working.
  3. Order-size profile  -- child-order clip sizes from execution algos cluster
                            tightly; retail does not. Clip homogeneity is a
                            direct fingerprint of a VWAP/POV algo at work.
"""
from __future__ import annotations

from collections import defaultdict, deque

from .events import L3Event


class _Level:
    """One price level, preserving arrival order (the actual FIFO queue)."""
    __slots__ = ("price", "side", "orders", "total")

    def __init__(self, price: float, side: int):
        self.price = price
        self.side = side
        self.orders: deque[str] = deque()
        self.total = 0.0


class OrderBookL3:
    """Market-by-order book with iceberg/replenishment instrumentation."""

    def __init__(self, symbol: str, refill_window_ns: int = 500_000_000):
        self.symbol = symbol
        self.refill_window_ns = refill_window_ns

        self.orders: dict[str, tuple[int, float, float]] = {}   # id -> (side, px, size)
        self.levels: dict[tuple[int, float], _Level] = {}
        self.sequence = 0
        self.ts_ns = 0
        self.ready = False

        # --- iceberg instrumentation -------------------------------------
        # For each price we remember the last time it was consumed by a match
        # and the clip size that was consumed. A fresh `open` at that price for
        # a similar size inside refill_window_ns is a replenishment.
        self._last_match: dict[tuple[int, float], tuple[int, float]] = {}
        self.refills: dict[tuple[int, float], int] = defaultdict(int)
        self.refill_volume: dict[tuple[int, float], float] = defaultdict(float)
        self._clip_sizes: deque[float] = deque(maxlen=512)

    # ---------------------------------------------------------------- apply
    def apply(self, e: L3Event) -> None:
        self.ts_ns = e.ts_ns
        self.sequence = e.sequence or self.sequence
        a = e.action

        if a == "open":
            self._open(e)
        elif a == "match":
            self._match(e)
        elif a == "done":
            self._done(e)
        elif a == "change":
            self._change(e)

    def _key(self, side: int, price: float) -> tuple[int, float]:
        return (side, round(price, 10))

    def _open(self, e: L3Event) -> None:
        size = e.remaining or e.size
        if size <= 0:
            return
        k = self._key(e.side, e.price)
        lvl = self.levels.get(k)
        if lvl is None:
            lvl = self.levels[k] = _Level(e.price, e.side)
        lvl.orders.append(e.order_id)
        lvl.total += size
        self.orders[e.order_id] = (e.side, e.price, size)
        self._clip_sizes.append(size)

        # Replenishment test: did this price just get eaten, and is this new
        # order roughly the same clip? That is the iceberg signature.
        prev = self._last_match.get(k)
        if prev is not None:
            dt = e.ts_ns - prev[0]
            if 0 <= dt <= self.refill_window_ns and prev[1] > 0:
                ratio = size / prev[1]
                if 0.5 <= ratio <= 2.0:
                    self.refills[k] += 1
                    self.refill_volume[k] += size

    def _match(self, e: L3Event) -> None:
        oid = e.order_id                       # maker order id
        rec = self.orders.get(oid)
        if rec is None:
            return
        side, px, size = rec
        traded = min(e.size, size)
        left = size - traded
        k = self._key(side, px)
        lvl = self.levels.get(k)
        if lvl is not None:
            lvl.total -= traded
        self._last_match[k] = (e.ts_ns, traded)
        if left <= 1e-12:
            self._remove(oid)
        else:
            self.orders[oid] = (side, px, left)

    def _done(self, e: L3Event) -> None:
        self._remove(e.order_id)

    def _change(self, e: L3Event) -> None:
        rec = self.orders.get(e.order_id)
        if rec is None:
            return
        side, px, old = rec
        new = e.remaining or e.size
        k = self._key(side, px)
        lvl = self.levels.get(k)
        if lvl is not None:
            lvl.total += (new - old)
        self.orders[e.order_id] = (side, px, new)

    def _remove(self, oid: str) -> None:
        rec = self.orders.pop(oid, None)
        if rec is None:
            return
        side, px, size = rec
        k = self._key(side, px)
        lvl = self.levels.get(k)
        if lvl is None:
            return
        lvl.total -= size
        try:
            lvl.orders.remove(oid)
        except ValueError:
            pass
        if lvl.total <= 1e-12 or not lvl.orders:
            self.levels.pop(k, None)

    # -------------------------------------------------------------- queries
    def best(self, side: int) -> tuple[float, float]:
        """(price, size) of the best level on `side`."""
        prices = [k[1] for k in self.levels if k[0] == side]
        if not prices:
            return (float("nan"), 0.0)
        px = max(prices) if side > 0 else min(prices)
        return (px, self.levels[self._key(side, px)].total)

    @property
    def mid(self) -> float:
        b, _ = self.best(1)
        a, _ = self.best(-1)
        return 0.5 * (b + a)

    def order_count(self, side: int, price: float) -> int:
        lvl = self.levels.get(self._key(side, price))
        return len(lvl.orders) if lvl else 0

    def queue_ahead(self, oid: str) -> float:
        """Size resting ahead of our order -- how much must trade before we fill."""
        rec = self.orders.get(oid)
        if rec is None:
            return 0.0
        side, px, _ = rec
        lvl = self.levels.get(self._key(side, px))
        if lvl is None:
            return 0.0
        ahead = 0.0
        for other in lvl.orders:
            if other == oid:
                break
            r = self.orders.get(other)
            if r:
                ahead += r[2]
        return ahead

    def iceberg_score(self, side: int, price: float) -> float:
        """0..1 confidence that a hidden/reserve order works this price.

        Built purely from observed refill count, which is the frequentist
        detection family from the GLOBEX literature: you cannot see hidden
        size, but you can count how often a level rises from the dead.
        """
        n = self.refills.get(self._key(side, price), 0)
        return 0.0 if n <= 0 else min(1.0, n / 5.0)

    def clip_homogeneity(self) -> float:
        """0..1. High means recent order sizes cluster tightly -- the
        fingerprint of an execution algo slicing a parent order into
        uniform children, rather than heterogeneous natural flow."""
        if len(self._clip_sizes) < 30:
            return 0.0
        xs = sorted(self._clip_sizes)
        n = len(xs)
        med = xs[n // 2]
        if med <= 0:
            return 0.0
        near = sum(1 for x in xs if 0.8 * med <= x <= 1.25 * med)
        return near / n

    def total_refill_pressure(self, side: int) -> float:
        """Aggregate replenished notional on one side -- persistent hidden
        demand (side=+1) or supply (side=-1)."""
        return sum(v for k, v in self.refill_volume.items() if k[0] == side)
