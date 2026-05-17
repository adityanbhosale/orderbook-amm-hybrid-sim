# orderbook-amm-hybrid-sim

## Overview

This repo is a multi-market simulation that compares three venue designs: constant-product AMM, central-limit order book (CLOB), and a hybrid venue (described below), all under the same event-driven Bayesian agent trader populations. The core question underlying this test sweep was **how adding a passive AMM-style liquidity layer to an orderbook changes PnL economics for informed traders (simulated here agentically) and price discovery over time.** The aforementioned Bayesian trading agents trade 3 correlated markets with shared latent factors; venues only differ in how orders placed by these agents match and how passive liquidity is supplied in the market. The design space is modeled as an analog of the public work from Ellipsis Labs and the associated Solana infrastructure.

## Relation to Existing Models:

The initial iteration of Phoenix (aka Phoenix Legacy – repo: https://github.com/Ellipsis-Labs/phoenix-v1) is a purely on-chain CLOB design. The constant product AMM venue in this test sweep is modeled as an analog of Phoenix-v1 – including price-time priority, a descending bid side, ascending ask side, market orders walking the book (with volume-weighted average fill priors), market & limit orders, and posting/taking. Plasma-style engines (which reads as the underlying model for the SolFi DEX) emphasize sandwich-resistant AMM execution. Lastly, Phoenix Perps is described as blending orderbook matching with an AMM-style passive layer. This codebase models that design space, not the private Phoenix Perps implementation (which I did not have access to while building this sweep). The build allows you to swap mechanisms under identical assumptions / market information / agent code and read off differences in convergence, rent extraction for informed traders, and trade intensity.


## What this is not

I want to be clear that this is not a model of Phoenix Perps, not a production venue benchmark, or an exhaustive search over mechanism parameters. The sweep is a single factorial configuration (3 mechanisms, 12 cells on capital and signal axes, 25 seeds per cell, 900 total runs). The results included below should be read as **what a single configuration shows**,  not a definitive ranking of AMM versus CLOB versus hybrid venues. The durable feature of the build is the substrate in ‘environment/’, ‘venues/’, and ‘agents/’, discrete-event scheduling, venue interfaces, and log-space beliefs of which the sweep in ‘simulation/sweep.py’ is one application. If any value in this summary contradicts with your rerun of the sweep, I’d trust the parquet under ‘analysis/results/’ and the notebook 
analysis/sweep_analysis.ipynb’.


## Architecture

The **substrate** (`environment/`) implements continuous-price, log-linear fair-value dynamics across a three-market cluster with correlated structure. Markets are generated from a latent factor model with shared loadings (`environment/information.py`, `environment/cross_market.py`). Signals arrive on an event-driven scheduler (`environment/simulator.py`) with configurable observation delays; trades flow through `MarketEnvironment` (`environment/market_environment.py`), which routes intents, applies margin (`environment/margin.py`), and logs fills. Capital checks use impact-based commitment before execution (`environment/trading_utils.py`).

**Venues** (`venues/`) share an abstract `Venue` in `venues/base.py`: `submit_market_order`, `submit_limit_order`, `cancel_order`, `get_state`, `estimate_impact`, and `tick`. Three implementations sit behind the same interface. `venues/constant_product.py` is a minimal constant-product AMM. `venues/clob.py` is a price-time-priority book with bootstrap seeding via `seed_initial_book` (agent `-1`) so cold start does not deadlock on empty mids. `venues/hybrid.py` composes an inner CLOB with a per-tick passive LP (`LP_AGENT_ID = "-2"`) that refreshes symmetric quotes—an analog of a hybrid orderbook plus passive layer, not a line-by-line port of any live deployment.

**Agents** (`agents/`) are four event-driven Bayesian classes ported from prior work on event-triggered RWAs (see [Prior work](#prior-work)): naive Gaussian (`agents/informed_naive.py`), tail-aware (`agents/informed_tail.py`), aggregated cross-market (`agents/informed_aggregated.py`), and joint-factor (`agents/informed_joint_factor.py`), plus a Poisson-style noise trader (`agents/event_noise.py`). All informed agents operate in log-fair-value space and update beliefs from signals. On AMM venues they trade via market orders only; on CLOB and hybrid they route through maker/taker logic in `agents/belief_utils.py` (`maker_taker_decision`, `consider_clob_hybrid_trade`).

The central design choice is that **`estimate_impact(side, qty)`** is the venue-agnostic sizing primitive. Agents scale positions against estimated volume-weighted average price from `downsize_quantity_for_capital` in `environment/trading_utils.py`, not against mechanism-specific knobs such as LMSR liquidity parameter or raw curve reserves. Limit orders on CLOB and hybrid use a parallel path (`downsize_limit_quantity_for_capital`, `committed_capital_limit` in `environment/margin.py`). Sweep metrics in `metrics/convergence.py`, `metrics/rent.py`, and `metrics/capital.py` consume the trade log emitted by `MarketEnvironment` without venue-specific branches. That keeps populations comparable when you change only the venue class.

## Findings

**This sweep is one parameterization (3 mechanisms × 12 cells × 25 seeds = 900 runs). Read findings as illustrative, not definitive.**

**1. Hybrid mechanism reduces agent-to-agent trade volume.** Across every slice we inspected, hybrid runs average roughly seventeen trades per run versus thirty-two to thirty-five for AMM and CLOB under the same agent mix and budgets. The passive LP in `venues/hybrid.py` posts and refreshes quotes each tick; flow that would otherwise cross the visible book is absorbed at the LP layer. That is the cleanest mechanism-specific effect in the data: hybrid is not “more active,” it is **less directly adversarial between informed agents** because more volume interacts with the programmatic layer first.

**2. Price discovery is largely mechanism-agnostic at this displacement.** With per-seed anchor displacement drawn as Normal(0, 1%) in `simulation/sweep.py` (`ANCHOR_DISPLACEMENT_STD = 0.01`), all three mechanisms land near sub–one-percent log-RMSE against truth in the headline cell—on the order of eighty basis points in relative terms once mids have traded. Convergence ordering (AMM slightly ahead of hybrid, CLOB close behind) is stable when you move capital or signal regime; venue choice does not dramatically reorder discovery accuracy in this configuration. Figure `analysis/figures/02_convergence_curves.png` shows mean relative error paths; `analysis/figures/04_price_trajectories.png` overlays sample mids against dashed fair values for one seed.

**3. Rent efficiency is regime-dependent, not mechanism-dependent.** The stabilized metric in `metrics/rent.py` (`rent_efficiency_stable`, denominator floored at ten dollars) avoids blow-ups when noise PnL is tiny. In the headline slice (diverse mix, mid capital, low signal), CLOB scores highest and hybrid lowest—but that ordering **does not survive** stress slices. At low capital, all rent-efficiency values collapse toward zero with AMM on top. At high signal noise, AMM and hybrid go negative while only CLOB stays near break-even. Figure `analysis/figures/01_rent_efficiency.png` plots stable rent by capital and signal. There is no universal “best mechanism for informed traders” in this run.

**4. Capital does not bind at tested budget levels.** Across mechanisms and capital bands ($100, $1,000, $10,000), `frac_informed_exhausted_before_convergence` from `metrics/capital.py` remains at zero in the reported cells. That is a notable null relative to earlier LS-LMSR work where capital was binding in tail regimes. Figure `analysis/figures/03_capital_saturation.png` documents the flat line. Either budgets are generous relative to trade size and margin haircuts, or the two-percent relative convergence band fires before exhaustion matters—or both.

### Headline table (diverse mix, mid capital, low signal; 25 seeds)

| Mechanism | Convergence | Rent eff (stable) | Trades/run | LP role |
|-----------|-------------|-------------------|------------|---------|
| AMM | 76 bps | +1.35 | 35.2 ± 1.3 | Curve (passive) |
| CLOB | 87 bps | +2.13 | 32.2 ± 4.0 | None |
| Hybrid | 84 bps | +0.74 | 17.2 ± 2.7 | Per-tick refresh |

Convergence is mean `normalized_rmse_log` expressed in basis points of log space (×10⁴). Rent efficiency uses the stable variant from the sweep output. Trade counts are mean ± std across seeds.

Diagnostic slices in `analysis/sweep_analysis.ipynb` show the same trade-volume ordering at low capital and rent reshuffling at high signal without rerunning the sweep. Table A (low capital) compresses rent efficiency toward zero; Table B (high signal) drives AMM and hybrid rent negative while CLOB stays near flat—useful context when interpreting the headline row above as a single cell, not a global winner.

## Limitations

- Single sweep configuration; no LP parameter exploration (`lp_spread_pct`, `lp_base_qty`, decay, refresh frequency are fixed in `venues/hybrid.py`).
- Bootstrap displacement is ad hoc (1% per-seed normal in `simulation/sweep.py`); real launches use venue-specific seeding and disclosure.
- Maker/taker threshold is fixed (`taker_threshold=2.0` in `agents/belief_utils.py`); live strategies adapt posting versus taking with inventory and confidence.
- No sustained adversarial agent; manipulation, if added, is not modeled as a long-horizon game.
- Capital binding from prior LS-LMSR work does not replicate here—budgets may be too large, or convergence may register before exhaustion bites.
- Phoenix Perps’ hybrid implementation is private; `HybridVenue` is a defensible analog (composed CLOB + refreshing LP), not a replica.

## What's next

Natural extensions follow directly from the limitations. An **LP parameter sweep**—spread, base quantity, decay, levels, and refresh cadence in `venues/hybrid.py` and `simulation/sweep.py`—is likely the highest marginal experiment: it targets the one mechanism-specific lever that already shows a strong volume effect. **Adaptive maker/taker thresholds** tied to posterior precision would loosen the fixed `taker_threshold` in `agents/belief_utils.py`. **Sustained adversarial agents** would replace any single-shot manipulation story with inventory and repeated impact. **Tighter capital regimes** (order-one to order-ten dollar budgets) would probe where `metrics/capital.py` starts reporting nonzero exhaustion. Finally, richer **multi-asset correlation** experiments could stress the joint-factor agent in `agents/informed_joint_factor.py` beyond the three-market cluster baked into `InformationConfig`.

## How to run

### Setup

Clone the repository, create a virtual environment, and install development dependencies (includes pytest, pandas, pyarrow, matplotlib, and jupyter for the analysis notebook):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Tests

Five smoke tests cover a single-agent AMM path, multi-agent budgets, CLOB book mechanics, hybrid LP refresh, and maker/taker intents on a seeded CLOB:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

### Full sweep

The factorial harness writes `analysis/results/sweep_summary.parquet` and `analysis/results/sweep_timeseries.parquet`. Expect on the order of twelve minutes for nine hundred runs on a typical laptop:

```bash
PYTHONPATH=. .venv/bin/python -m simulation.sweep
```

To rerun only the CLOB arm after a book-seeding change and merge into existing results:

```bash
PYTHONPATH=. .venv/bin/python -m simulation.sweep clob-only
```

### Notebook

Load the parquet outputs and regenerate figures under `analysis/figures/`:

```bash
cd analysis
MPLBACKEND=Agg PYTHONPATH=.. jupyter notebook sweep_analysis.ipynb
```

Or execute headless:

```bash
cd analysis
MPLBACKEND=Agg PYTHONPATH=.. ../.venv/bin/jupyter nbconvert --to notebook --execute sweep_analysis.ipynb --inplace
```

## Prior work

The four Bayesian informed agents and the information/signal environment are ported from the earlier simulation codebase [lmsr-preclinical-markets](https://github.com/adityabhosale/lmsr-preclinical-markets) (LS-LMSR event-triggered RWA markets); this repo replaces LMSR venues with AMM/CLOB/hybrid implementations while preserving log-space beliefs and event-driven population structure.
