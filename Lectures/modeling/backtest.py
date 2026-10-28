"""Readable classroom reference: monthly cohorts, independent lots, cash and fees.

HW4 students choose their own strategy and write their own backtester.
Adjusted-price units represent total-return exposure rather than actual
historical shares and broker fills.
"""

import numpy as np
import pandas as pd


class HeldPriceGapError(ValueError):
    """A stopped simulation, with dates for an auditable source investigation."""

    def __init__(self, symbol, date, last_quote_date):
        self.symbol = symbol
        self.date = pd.Timestamp(date)
        self.last_quote_date = pd.Timestamp(last_quote_date)
        super().__init__(f"Unresolved held-price gap: {symbol} at {self.date.date()}")


def capped_budgets(scores, holdings, sectors, cash, nav, top_n=20,
                   cohort_fraction=1 / 3, name_cap=.05, sector_cap=.25, fee_bps=5):
    """Plan inclusive dollar budgets from *prior-close* data, before execution.

    holdings is aggregate market value by ticker. Scores and sectors are also
    ticker-indexed Series. Reduce each equal budget as needed, retaining unused
    cash. Update planned exposures after every allocation, not after the loop.
    """
    if not scores.index.is_unique or not holdings.index.is_unique or not sectors.index.is_unique:
        raise ValueError("Aggregate duplicate tickers before allocating")
    if nav <= 0 or cash < 0 or not np.isclose(cash + holdings.sum(), nav):
        raise ValueError("Prior-close holdings and cash must reconcile to NAV")
    if not np.isfinite(scores).all() or not np.isfinite(holdings).all() or holdings.lt(0).any():
        raise ValueError("Scores/holdings must be finite; holdings nonnegative")
    if top_n < 1 or fee_bps < 0 or not all(0 < cap <= 1 for cap in [cohort_fraction, name_cap, sector_cap]):
        raise ValueError("Invalid allocation setting")
    sectors = sectors.fillna("Unknown")
    if not set(scores.index).union(holdings.index).issubset(sectors.index):
        raise ValueError("Every scored/held ticker needs an explicit sector bucket")
    ranked = scores.rename_axis("symbol").rename("score").reset_index().sort_values(
        ["score", "symbol"], ascending=[False, True]).head(top_n)
    fee = fee_bps / 10_000
    budget = min(cash, nav * cohort_fraction)
    equal_budget = budget / len(ranked) if len(ranked) else 0.
    planned_names = holdings.to_dict()
    planned_sectors = holdings.groupby(sectors.reindex(holdings.index)).sum().to_dict()
    orders = {}
    for symbol in ranked.symbol:
        sector = sectors.loc[symbol]
        name_room = max(0., name_cap * nav - planned_names.get(symbol, 0.))
        sector_room = max(0., sector_cap * nav - planned_sectors.get(sector, 0.))
        inclusive = min(equal_budget, budget, name_room * (1 + fee), sector_room * (1 + fee))
        notional = inclusive / (1 + fee)
        if inclusive > 1e-8:
            orders[symbol] = inclusive
            budget -= inclusive
            planned_names[symbol] = planned_names.get(symbol, 0.) + notional
            planned_sectors[sector] = planned_sectors.get(sector, 0.) + notional
    return pd.Series(orders, dtype=float, name="inclusive_budget")


def run_backtest(prices, benchmark, scores, start, end, initial_cash=1_000_000,
                 top_n=3, cohort_fraction=1 / 3, fee_bps=5, holding_days=90):
    """Use yesterday's scores; buy monthly with carried cash; mark every session."""
    if initial_cash <= 0 or top_n < 1 or not 0 < cohort_fraction <= 1 or fee_bps < 0:
        raise ValueError("Invalid capital, selection, budget, or fee parameter")
    if prices.duplicated(["symbol", "date"]).any() or scores.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate price or prediction key")
    if not np.isfinite(scores.score).all():
        raise ValueError("Nonfinite prediction")
    if not (np.isfinite(prices.adj_close) & prices.adj_close.gt(0)).all():
        raise ValueError("Invalid price")
    calendar = pd.DatetimeIndex(benchmark.date).sort_values()
    if calendar.has_duplicates or not (np.isfinite(benchmark.adj_close) & benchmark.adj_close.gt(0)).all():
        raise ValueError("Invalid benchmark")
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if not len(days) or calendar.get_loc(days[0]) == 0:
        raise ValueError("Need test sessions and one preceding signal session")
    quotes = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
    score_groups = {date: group for date, group in scores.groupby("date")}
    fee = fee_bps / 10_000
    cash = float(initial_cash)
    lots, fills, diagnostics = [], [], []
    previous_day = calendar[calendar.get_loc(days[0]) - 1]
    history = [{"date": previous_day, "cash": cash, "holdings_value": 0., "nav": cash,
                "open_lots": 0, "fees": 0., "gross_traded": 0., "stale_marks": 0}]
    last_marks, last_seen = {}, {}
    next_lot_id = 0
    for day in days:
        day_index = calendar.get_loc(day)
        signal_date = calendar[day_index - 1]
        carried_cash = cash  # Today's unknown sale proceeds cannot fund precommitted buys.
        daily_fees, gross_traded, stale_marks = 0., 0., 0
        observed = quotes.loc[day].dropna()
        for symbol, price in observed.items():
            last_marks[symbol], last_seen[symbol] = price, day_index
        for lot in lots.copy():
            symbol = lot["symbol"]
            if day_index - last_seen[symbol] > 5:
                raise HeldPriceGapError(symbol, day, calendar[last_seen[symbol]])
            if symbol not in observed:
                stale_marks += 1
                diagnostics.append({"date": day, "symbol": symbol, "event": "stale_mark"})
            if day >= lot["due_date"] and symbol in observed:
                notional = lot["units"] * observed[symbol]
                cost = notional * fee
                cash += notional - cost
                daily_fees += cost
                gross_traded += notional
                fills.append({"date": day, "signal_date": lot["signal_date"], "lot_id": lot["lot_id"],
                              "symbol": symbol, "side": "SELL", "price": observed[symbol],
                              "units": lot["units"], "notional": notional, "fee": cost,
                              "cash_change": notional - cost, "due_date": lot["due_date"]})
                lots.remove(lot)
        is_entry_day = day == days[0] or day.to_period("M") != signal_date.to_period("M")
        if is_entry_day and signal_date in score_groups:
            # Rankings and total budget are fixed using the previous close's information.
            ranked = score_groups[signal_date].sort_values(["score", "symbol"], ascending=[False, True])
            chosen = ranked.head(top_n)
            budget = min(carried_cash, history[-1]["nav"] * cohort_fraction)
            per_name = budget / len(chosen) if len(chosen) else 0
            for row in chosen.itertuples():
                if row.symbol not in observed:
                    diagnostics.append({"date": day, "symbol": row.symbol, "event": "rejected_missing_quote"})
                    continue  # Do not substitute another name using execution-day information.
                if per_name <= 1e-8:
                    continue
                notional = per_name / (1 + fee)  # Reserve the entry fee inside the cash budget.
                cost = notional * fee
                units = notional / observed[row.symbol]
                cash -= notional + cost
                daily_fees += cost
                gross_traded += notional
                lot = {"lot_id": next_lot_id, "symbol": row.symbol, "units": units,
                       "signal_date": signal_date, "entry_date": day,
                       "entry_price": observed[row.symbol], "due_date": day + pd.Timedelta(days=holding_days)}
                lots.append(lot)
                next_lot_id += 1
                fills.append({"date": day, "signal_date": signal_date, "lot_id": lot["lot_id"],
                              "symbol": row.symbol, "side": "BUY", "price": observed[row.symbol],
                              "units": units, "notional": notional, "fee": cost,
                              "cash_change": -(notional + cost), "due_date": lot["due_date"]})
        holdings = sum(lot["units"] * last_marks[lot["symbol"]] for lot in lots)
        if cash < -1e-7:
            raise AssertionError("Cash became negative")
        history.append({"date": day, "cash": cash, "holdings_value": holdings, "nav": cash + holdings,
                        "open_lots": len(lots), "fees": daily_fees, "gross_traded": gross_traded,
                        "stale_marks": stale_marks})
    history = pd.DataFrame(history).set_index("date")
    b = benchmark.set_index("date").adj_close.reindex(days)
    history["benchmark_nav"] = float(initial_cash)
    history.loc[days, "benchmark_nav"] = initial_cash * b / b.iloc[0]
    # Initial row is pre-entry capital. First day's return includes all entry costs.
    return {"history": history,
            "fills": pd.DataFrame(fills, columns=["date", "signal_date", "lot_id", "symbol", "side", "price",
                                                 "units", "notional", "fee", "cash_change", "due_date"]),
            "open_lots": pd.DataFrame(lots, columns=["lot_id", "symbol", "units", "signal_date",
                                                     "entry_date", "entry_price", "due_date"]),
            "diagnostics": pd.DataFrame(diagnostics, columns=["date", "symbol", "event"])}


def performance(nav, annual_rf=0.02):
    """Daily simple returns, sample volatility, and a calendar-time CAGR."""
    nav = nav.sort_index()
    if len(nav) < 2 or nav.isna().any() or (nav <= 0).any():
        raise ValueError("Need at least two finite, positive NAV observations")
    if not np.isfinite(nav).all():
        raise ValueError("Nonfinite NAV")
    returns = nav.pct_change(fill_method=None).dropna()
    excess = returns - ((1 + annual_rf) ** (1 / 252) - 1)
    volatility = returns.std(ddof=1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return pd.Series({"final_value": nav.iloc[-1], "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
                      "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
                      "max_drawdown": (nav / nav.cummax() - 1).min(),
                      "annualized_volatility": volatility * np.sqrt(252),
                      "Sharpe": np.sqrt(252) * excess.mean() / volatility if volatility > 1e-12 else np.nan})
