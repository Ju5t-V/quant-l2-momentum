"""L2 (market-by-price) limit order book.

Design note on performance: diff-depth streams mutate *sizes* on existing
price levels far more often than they add or remove levels. So we keep the
sorted price arrays cached and only re-sort when the key set actually changes.
That turns the common case from O(n log n) into O(1).
"""
from __future__ import annotations

import numpy as np

from .events import BookDelta


class OrderBookL2:
    __slots__ = ("symbol", "_bids", "_asks", "_bid_px", "_ask_px",
                 "_dirty_b", "_dirty_a", "last_id", "ts_ns", "ready")

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._bid_px: np.ndarray | None = None   # descending
        self._ask_px: np.ndarray | None = None   # ascending
        self._dirty_b = True
        self._dirty_a = True
        self.last_id = 0
        self.ts_ns = 0
        self.ready = False

    # ---------------------------------------------------------------- apply
    def apply(self, d: BookDelta) -> None:
        if d.is_snapshot:
            self._bids.clear()
            self._asks.clear()
            self._dirty_b = self._dirty_a = True
            self.ready = True

        b, a = self._bids, self._asks
        for px, sz in d.bids:
            if sz <= 0.0:
                if b.pop(px, None) is not None:
                    self._dirty_b = True
            else:
                if px not in b:
                    self._dirty_b = True
                b[px] = sz
        for px, sz in d.asks:
            if sz <= 0.0:
                if a.pop(px, None) is not None:
                    self._dirty_a = True
            else:
                if px not in a:
                    self._dirty_a = True
                a[px] = sz

        self.last_id = d.final_id or self.last_id
        self.ts_ns = d.ts_ns

    # ------------------------------------------------------------- accessors
    def _bp(self) -> np.ndarray:
        if self._dirty_b or self._bid_px is None:
            self._bid_px = np.array(sorted(self._bids, reverse=True), dtype=np.float64)
            self._dirty_b = False
        return self._bid_px

    def _ap(self) -> np.ndarray:
        if self._dirty_a or self._ask_px is None:
            self._ask_px = np.array(sorted(self._asks), dtype=np.float64)
            self._dirty_a = False
        return self._ask_px

    @property
    def best_bid(self) -> float:
        p = self._bp()
        return float(p[0]) if p.size else float("nan")

    @property
    def best_ask(self) -> float:
        p = self._ap()
        return float(p[0]) if p.size else float("nan")

    @property
    def best_bid_size(self) -> float:
        p = self._bp()
        return self._bids[p[0]] if p.size else 0.0

    @property
    def best_ask_size(self) -> float:
        p = self._ap()
        return self._asks[p[0]] if p.size else 0.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    def is_valid(self) -> bool:
        bb, ba = self.best_bid, self.best_ask
        return bb == bb and ba == ba and ba > bb      # NaN-safe

    def depth(self, n: int = 10):
        """Return (bid_px, bid_sz, ask_px, ask_sz) for the top n levels."""
        bp = self._bp()[:n]
        ap = self._ap()[:n]
        bs = np.fromiter((self._bids[p] for p in bp), dtype=np.float64, count=bp.size)
        asz = np.fromiter((self._asks[p] for p in ap), dtype=np.float64, count=ap.size)
        return bp, bs, ap, asz

    # -------------------------------------------------------------- metrics
    def imbalance(self, n: int = 5) -> float:
        """Depth imbalance in [-1, 1]. +1 means all bid, -1 means all ask."""
        _, bs, _, asz = self.depth(n)
        tb, ta = bs.sum(), asz.sum()
        tot = tb + ta
        return 0.0 if tot <= 0 else float((tb - ta) / tot)

    def microprice(self) -> float:
        """Stoikov first-order microprice: mid adjusted by queue imbalance.

        A better short-horizon predictor of the future mid than the mid itself,
        because the side with the smaller queue gets consumed first. Each side
        is weighted by the OPPOSITE queue: a thin ask pulls price up.
        """
        bs, asz = self.best_bid_size, self.best_ask_size
        tot = bs + asz
        if tot <= 0:
            return self.mid
        return (self.best_bid * asz + self.best_ask * bs) / tot

    def notional_within(self, bps: float, side: int) -> float:
        """Resting notional within bps of mid on one side.

        This is the quantity that actually matters for whether a sweep can get
        through. Level counts are meaningless when tick sizes differ.
        """
        m = self.mid
        if m != m:
            return 0.0
        lim = m * bps / 10_000.0
        if side > 0:
            bp, bs, _, _ = self.depth(200)
            mask = bp >= (m - lim)
            return float((bp[mask] * bs[mask]).sum())
        _, _, ap, asz = self.depth(200)
        mask = ap <= (m + lim)
        return float((ap[mask] * asz[mask]).sum())

    def sweep_cost_bps(self, notional: float, side: int) -> float:
        """Cost in bps of taking `notional` immediately. This is the slippage
        model: we walk the real book rather than assuming a fixed spread."""
        m = self.mid
        if m != m or notional <= 0:
            return 0.0
        bp, bs, ap, asz = self.depth(200)
        px, sz = (ap, asz) if side > 0 else (bp, bs)
        if px.size == 0:
            return 0.0
        remaining, cost = notional, 0.0
        for p, s in zip(px, sz):
            lvl = p * s
            take = min(remaining, lvl)
            cost += take * p
            remaining -= take
            if remaining <= 1e-12:
                break
        filled = notional - remaining
        if filled <= 0:
            return 0.0
        vwap = cost / filled
        # Unfilled remainder is penalised: we could not actually get done.
        pen = (remaining / notional) * 50.0
        return abs(vwap - m) / m * 10_000.0 + pen
