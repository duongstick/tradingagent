"""Double-entry portfolio ledger.

Every economic event is recorded as a balanced set of debits and credits across
accounts (Cash, Positions, Fees, PnL). The invariant `sum(debits) == sum(credits)` must
hold after every transaction, and `assets == cash + positions_value`. Borrowing the
accountant's discipline makes silent balance drift impossible to hide — a bug shows up
immediately as an unbalanced entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


class LedgerError(Exception):
    """Raised when a transaction would violate the double-entry invariant."""


@dataclass
class Entry:
    timestamp: datetime
    account: str
    debit: float
    credit: float
    memo: str


@dataclass
class Lot:
    symbol: str
    quantity: int
    avg_price: float


@dataclass
class Ledger:
    cash: float
    entries: list[Entry] = field(default_factory=list)
    positions: dict[str, Lot] = field(default_factory=dict)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    def _post(self, ts, legs: list[tuple[str, float, float]], memo: str) -> None:
        debits = sum(d for _, d, _ in legs)
        credits = sum(c for _, _, c in legs)
        if abs(debits - credits) > 1e-6:
            raise LedgerError(
                f"unbalanced transaction: debits={debits:.4f} credits={credits:.4f} ({memo})"
            )
        for acct, d, c in legs:
            self.entries.append(Entry(ts, acct, d, c, memo))

    def buy(self, symbol: str, qty: int, price: float, fee: float, ts: datetime | None = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        cost = qty * price
        if cost + fee > self.cash + 1e-6:
            raise LedgerError(f"insufficient cash for {symbol}: need {cost+fee:.0f}, have {self.cash:.0f}")
        # Cash decreases (credit); Positions + Fees increase (debit).
        self._post(ts, [
            ("Positions", cost, 0.0),
            ("Fees", fee, 0.0),
            ("Cash", 0.0, cost + fee),
        ], f"BUY {qty} {symbol} @ {price:.0f}")
        self.cash -= cost + fee
        self.fees_paid += fee
        lot = self.positions.get(symbol)
        if lot is None:
            self.positions[symbol] = Lot(symbol, qty, price)
        else:
            total = lot.quantity + qty
            lot.avg_price = (lot.avg_price * lot.quantity + price * qty) / total
            lot.quantity = total

    def sell(self, symbol: str, qty: int, price: float, fee: float, ts: datetime | None = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        lot = self.positions.get(symbol)
        if lot is None or lot.quantity < qty:
            raise LedgerError(f"cannot sell {qty} {symbol}: holding {lot.quantity if lot else 0}")
        proceeds = qty * price
        pnl = (price - lot.avg_price) * qty
        self._post(ts, [
            ("Cash", proceeds - fee, 0.0),
            ("Fees", fee, 0.0),
            ("Positions", 0.0, lot.avg_price * qty),
            ("RealizedPnL", 0.0, pnl) if pnl >= 0 else ("RealizedPnL", -pnl, 0.0),
        ], f"SELL {qty} {symbol} @ {price:.0f}")
        self.cash += proceeds - fee
        self.fees_paid += fee
        self.realized_pnl += pnl - fee
        lot.quantity -= qty
        if lot.quantity == 0:
            del self.positions[symbol]

    def positions_value(self, prices: dict[str, float]) -> float:
        return sum(lot.quantity * prices.get(sym, lot.avg_price) for sym, lot in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.positions_value(prices)

    def check_invariant(self) -> bool:
        """Verify the books balance: total debits == total credits."""
        total_d = sum(e.debit for e in self.entries)
        total_c = sum(e.credit for e in self.entries)
        if abs(total_d - total_c) > 1e-3:
            raise LedgerError(f"ledger out of balance: D={total_d:.2f} C={total_c:.2f}")
        return True
