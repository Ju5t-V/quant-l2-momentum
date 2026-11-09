"""Normalized market-data event schema.

Every feed adapter (Binance L2, Coinbase L3, replay, synthetic) emits *these*
types and nothing else.  Strategies and features therefore never learn which
venue they are running against, which is what lets the same code path serve
live trading, recorded replay and backtest with zero train/live skew.

Timestamps are integer nanoseconds since epoch, always exchange time when the
venue provides it (never local receipt time), because local receipt time
carries network jitter that leaks into microstructure features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

BID = 1
ASK = -1

# Aggressor / trade-direction convention used everywhere in this codebase.
BUY_AGGRESSOR = 1     # trade lifted the ask  -> demand
SELL_AGGRESSOR = -1   # trade hit the bid     -> supply
UNKNOWN_AGGRESSOR = 0


@dataclass(slots=True)
class BookDelta:
    """An L2 (market-by-price) update.

    ``bids``/``asks`` are absolute *new sizes* at a price, not increments.
    A size of 0.0 means the level is removed.  This matches Binance diff-depth
    semantics and is the lowest common denominator across venues.
    """
    ts_ns: int
    symbol: str
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    first_id: int = 0        # Binance 'U'
    final_id: int = 0        # Binance 'u'
    is_snapshot: bool = False


@dataclass(slots=True)
class Trade:
    """A public print off the tape."""
    ts_ns: int
    symbol: str
    price: float
    size: float
    aggressor: int = UNKNOWN_AGGRESSOR
    trade_id: int = 0

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass(slots=True)
class L3Event:
    """A market-by-order event: the real Level 3 feed.

    ``action`` is one of open / match / done / change.  This is what makes
    iceberg and queue-position work possible -- L2 cannot express it because
    it aggregates every order at a price into one number.
    """
    ts_ns: int
    symbol: str
    action: str
    order_id: str
    side: int
    price: float
    size: float = 0.0
    remaining: float = 0.0
    sequence: int = 0
    reason: str = ""          # 'filled' | 'canceled' on done


@dataclass(slots=True)
class Signal:
    """What a strategy emits.  Never an order -- the router decides that."""
    ts_ns: int
    symbol: str
    strategy: str
    direction: int            # +1 long, -1 short, 0 flat
    strength: float           # 0..1, used for risk scaling
    reason: str = ""
    features: dict = field(default_factory=dict)
    ttl_ns: int = 30_000_000_000   # signal decays if not acted on


@dataclass(slots=True)
class Fill:
    ts_ns: int
    symbol: str
    strategy: str
    side: int
    price: float
    size: float
    fee: float
    slippage_bps: float = 0.0
    liquidity: str = "taker"
