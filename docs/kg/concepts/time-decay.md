---
up: []
related: ["[[theta]]", "[[gamma]]", "[[calendar-spread]]"]
created: 2026-04-07
type: concept
tags: [options, mechanics]
aliases: [theta decay, time value erosion]
---

# Time Decay

> [!info] Definition
> The erosion of an option's extrinsic (time) value as expiration approaches, measured by [[theta]]. The most reliable force in options — time only moves one direction. ^definition

## Decay Curve

```
Extrinsic Value
100% |━━━━━━━━━━━━━━╲
     |                ╲
 75% |                  ╲
     |                    ╲
 50% |                      ╲
     |                        ╲
 25% |                          ╲
     |                            ╲━━
  0% └────────────────────────────────
     90    60    45    30   15   7  0  DTE
```

- **90-45 DTE:** ~25% of value decays (slow)
- **45-15 DTE:** ~40% of value decays (accelerating)
- **15-0 DTE:** ~35% of value decays (rapid)

## Key Properties

1. **Non-linear:** Decay accelerates — the "sweet spot" for selling is 30-45 DTE
2. **ATM heaviest:** [[moneyness|ATM]] options have the most extrinsic value to lose
3. **IV amplifies:** Higher [[implied-volatility|IV]] = more extrinsic value = more daily decay
4. **Weekend pricing:** Markets price weekend decay into Friday close

## In Practice

- **Income strategies** ([[covered-call]], [[iron-condor]], [[short-strangle]]): Time decay IS the profit mechanism
- **[[calendar-spread]]:** Exploits differential decay — front-month decays faster than back-month
- **[[long-straddle]]:** Time decay is the enemy — need the move to outpace daily theta cost
- **[[zero-dte]]:** Extreme decay — ATM 0DTE options lose nearly all extrinsic value by close

---
**See also:** [[theta]] | [[gamma]] | [[moneyness]] | [[calendar-spread]]
