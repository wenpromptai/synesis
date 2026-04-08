---
up: []
related: ["[[implied-volatility]]", "[[open-interest]]", "[[moneyness]]", "[[delta]]"]
created: 2026-04-07
type: concept
tags: [options, data]
aliases: [chain, option chain, options chain]
---

# Options Chain

> [!info] Definition
> The complete listing of all available option contracts for a given underlying, organized by expiration date and strike price, showing bids, asks, volumes, open interest, and implied volatility. ^definition

## Structure

Each chain entry contains:

| Field | Description |
|-------|-------------|
| Strike | Exercise price |
| Type | Call or put |
| Expiration | Contract expiry date |
| Bid / Ask | Current market prices |
| Last Price | Most recent trade |
| Volume | Contracts traded today |
| [[open-interest|OI]] | Total outstanding contracts |
| [[implied-volatility|IV]] | Per-contract implied vol |
| [[moneyness|ITM]] | Whether in-the-money |

With `greeks=True`, also: [[delta]], [[gamma]], [[theta]], [[vega]], rho.

## In Practice

The chain is the primary data source for every options strategy:
- **Strike selection:** Filter by [[delta]] or [[moneyness]] level
- **Liquidity check:** [[open-interest]] > threshold, bid-ask spread < threshold
- **IV analysis:** Compare IV across strikes to observe [[volatility-smile]]
- **Expiration selection:** Compare chains across dates for [[calendar-spread]]

## Data Source

> [!info] Synesis Pipeline
> | Method | Returns |
> |--------|---------|
> | `yfinance.get_options_expirations(ticker)` | List of available expiry dates |
> | `yfinance.get_options_chain(ticker, expiration)` | Full chain: calls + puts with all fields |
> | `yfinance.get_options_chain(ticker, expiration, greeks=True)` | Chain + computed Greeks |
> | `yfinance.get_options_snapshot(ticker)` | ATM snapshot with spot price + 30d RV |
> | `massive.get_options_contracts(ticker)` | Contract reference metadata |

---
**See also:** [[implied-volatility]] | [[open-interest]] | [[delta]] | [[moneyness]] | [[volatility-smile]]
