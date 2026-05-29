# Regime-Adaptive Entropic Black-Litterman (RAEBL) Trading Strategy

> **Author:** Md Adnan Khalid · Roll No. 250103060  
> **Type:** Quantitative Algorithmic Trading · University Problem Set  
> **Period:** January 2018 – January 2025 · Monthly Rebalancing · Long-Only

---

## Overview

RAEBL is a six-phase algorithmic trading strategy that adapts the classic Black-Litterman portfolio optimisation framework to the prevailing market regime in real time. Rather than using fixed parameters, three core components are made dynamic:

| Component | Standard BL | RAEBL |
|---|---|---|
| Trust in views `τ` | Fixed ~0.05 | HMM entropy-scaled |
| Equilibrium prior `Π` | Fixed CAPM δ | Regime-conditional blend |
| View uncertainty `Ω` | Proportional to view variance | Hurst × IR × Granger harmonic mean |

**Universe:** AAPL, MSFT, NVDA, JPM, GS, CAT, XOM, PG, JNJ, PFE + SPY (market reference)  
**Capital:** $1,000,000 · **Transaction cost:** 10 bps one-way · **Rebalance:** Monthly

---

## Results

| Metric | RAEBL | Equal-Weight Benchmark |
|---|---|---|
| Annualised Return | 17.09% | 20.47% |
| Annualised Volatility | **19.13%** | 19.90% |
| Sharpe Ratio | 0.7154 | 0.8389 |
| Maximum Drawdown | **−27.93%** | −32.75% |
| Calmar Ratio | 0.6117 | 0.6251 |
| Sortino Ratio | 0.9067 | 1.0314 |
| Total Return | 200.96% | 267.25% |
| Best Month | **16.83%** | 13.99% |
| Worst Month | **−10.03%** | −10.46% |
| Monthly Win Rate | 61.4% | 63.9% |
| Avg Monthly Turnover | 41.0% | 6.2% |
| Total Fees | $55,767 | $8,782 |

RAEBL outperforms on every **risk metric** — lower volatility, shallower drawdown, better worst-month protection. Raw return trails due to the evaluation period being a secular bull market (2018–2024), where passive exposure is structurally advantaged over defensive regime-adaptive strategies.

---

## Architecture

```
Phase 1          Phase 2           Phase 2.5        Phase 3           Phase 4          Phase 5
─────────────    ─────────────     ─────────────    ─────────────     ─────────────    ─────────────
Data             Regime            Parameter        RAEBL             Backtesting      Robustness
Foundation  ──►  Detection   ──►   Tuning      ──►  Optimizer   ──►   Engine      ──►  Validation
─────────────    ─────────────     ─────────────    ─────────────     ─────────────    ─────────────
Kalman filter    HMM (3-state)     Walk-fwd CV      BL posterior      Raw adj close    Walk-forward
Log returns      Hurst exponent    Random search    Entropy τ         10 bps cost      Sensitivity
Realised vol     Granger network   Train-only        Regime Π         Monthly exec     IS/OOS split
Avg pairwise     Signal routing    Score: S̄−0.5σS   Multi-src Ω
correlation
```

---

## Signal Routing Logic

```
             ┌─────────────────────────────────────────────────┐
             │                Signal Router                     │
             │         (HMM Regime × Hurst Classification)     │
             └──────────┬──────────────┬──────────────┬────────┘
                        │              │              │
                   ┌────▼────┐   ┌─────▼─────┐  ┌────▼─────┐
                   │  BULL   │   │   BEAR    │  │  CHOPPY  │
                   └────┬────┘   └─────┬─────┘  └────┬─────┘
                        │              │              │
           ┌────────────┤       ┌──────┴──────┐      │
     Trending?  MeanRev?│  Trending?      else │   Lead-lag
        │           │   │     │               │   only (>0.5)
   Mom+lead-lag  Reversion   60% Mom       Lead-lag (>0.4)
```

**Hurst Classification** (126-day rolling R/S analysis):
- `H > 0.53` → Trending → Momentum signal
- `H < 0.47` → Mean-Reverting → Reversion signal
- `|H − 0.5| ≤ 0.03` → Random Walk → No position

---

## Project Structure

```
.
├── Analysis.ipynb          # Main notebook — all 6 phases
├── methodology.tex         # LaTeX methodology document (3 pages + appendix)
├── README.md
│
└── outputs/                # Generated on notebook run
    ├── phase1_kalman_filter.png
    ├── phase1_feature_dashboard.png
    ├── phase1_correlation_matrices.png
    ├── phase2_hmm_regimes.png
    ├── phase2_hurst_exponent.png
    ├── phase2_causality_network.png
    ├── phase2_signals_confidence.png
    ├── phase3_weights_parameters.png
    ├── phase4_equity_turnover.png
    ├── phase5_robustness.png
    ├── phase6_equity_curve.png
    ├── phase6_drawdown.png
    └── phase6_rolling_sharpe.png
```

---

## Installation

```bash
git clone https://github.com/<your-username>/raebl-strategy.git
cd raebl-strategy

pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy
pandas
matplotlib
seaborn
yfinance
scipy
hmmlearn
statsmodels
networkx
scikit-learn
```

---

## Running the Notebook

Open `Analysis.ipynb` in Jupyter and run all cells in order. Each phase packages its outputs into a dict (`phase1_output`, `phase2_output`, etc.) consumed by the next phase — do not skip cells.

**Estimated runtime:**

| Phase | Task | Time |
|---|---|---|
| 1 | Data download + Kalman filter | ~2 min |
| 2 | HMM (25 restarts) | ~3 min |
| 2 | Hurst exponent (rolling, 10 stocks) | ~4 min |
| 2 | Granger causality (90 pairs) | ~2 min |
| 2.5 | Parameter tuning (80 CV evals) | ~10 min |
| 3–6 | RAEBL loop + backtest + metrics | ~3 min |
| **Total** | | **~25 min** |

---

## Methodology

### Phase 1 — Data Foundation
- **Kalman Filter** applied to each ticker independently. Noise ratio `q/r` estimated on the training window (2018–2022) and held fixed for the test window — prevents look-ahead bias.
- **Features extracted:** log returns (from smoothed prices), 21-day realised volatility, 63-day volume Z-score, 63-day average pairwise correlation.
- **Backtesting uses raw adjusted close** — Kalman-smoothed prices are for signal generation only. Smoothed prices compress variance and would inflate Sharpe artificially.

### Phase 2 — Regime Detection
- **HMM:** 5-feature Gaussian HMM with 25 random restarts; best log-likelihood model chosen. States labelled by volatility + correlation score (not return) because all three states have positive mean return in the 2018–2024 bull market.
- **Hurst Exponent:** 126-day rolling R/S analysis. Band of ±0.03 (tighter than standard ±0.05) to maximise active signal classifications.
- **Granger Causality:** 90 directed pairs tested at `p < 0.05`, max lag 5. Fixed at training time — no re-estimation on test data.

### Phase 2.5 — Parameter Tuning
- Random search over 80 of 6,561 grid combinations.
- 3-fold walk-forward CV inside training window only (never touches test data).
- Scoring: `mean_CV_sharpe − 0.5 × std_CV_sharpe` — penalises instability across folds.

### Phase 3 — RAEBL Portfolio Construction
Three modifications to standard Black-Litterman:
1. **Entropy-scaled τ:** `τ = τ_min + (H(p)/ln3) × (τ_max − τ_min)` where `H(p)` is HMM state entropy
2. **Regime-conditional prior:** `δ_eff = P(Bull)×δ_Bull + P(Bear)×δ_Bear + P(Choppy)×δ_Choppy`
3. **Multi-source Ω:** harmonic mean of Hurst uncertainty, rolling IR, and Granger causality strength

Post-optimisation: 5% momentum tilt toward trailing 63-day winners (regime-scaled).

### Phase 4 — Backtesting
- Signal computed on month-end data; executed on the **first trading day of the next month** (no same-day execution — eliminates look-ahead).
- Transaction cost: 10 bps one-way on all trades.

### Phase 5 — Robustness Validation
- **Walk-forward:** HMM re-trained from scratch on each 3-year rolling window
- **Sensitivity:** 8 parameters perturbed at ±10% and ±30% independently
- **Hard split:** train 2018–2022 / test 2023–2024

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Kalman filter for signal generation only | Smoothed prices reduce noise in signals without distorting P&L |
| Volatility+correlation HMM labelling | All states have positive return in 2018–2024 — return-based labelling would misassign states |
| Hurst band = ±0.03 | Maximises active classifications vs ±0.05 standard; more views fire per rebalance |
| MIN_VIEWS = 1 | Single strong signal triggers BL posterior; avoids 100% fallback to prior weights |
| Equity floors (85/60/50%) | Prevents the optimizer from parking entirely in cash during uncertain regimes |
| Turnover penalty in objective | L2 penalty on weight changes keeps monthly turnover at economically reasonable levels |
| `score = S̄ − 0.5σ_S` | Explicitly favours stable-across-folds parameters over in-sample fit |

---

## Limitations

- **Bull-market bias in evaluation period:** 2018–2024 is structurally unfavourable to defensive strategies; full-cycle evaluation would show the drawdown protection benefit
- **HMM detection lag:** 21-day smoothed inputs mean regime transitions are detected with a lag — fast crashes (e.g. COVID March 2020) may be initially misclassified
- **Hurst warmup:** 126-day window means the first six months of data produce no active Hurst-based signals
- **No short selling:** long-only constraint limits the strategy's ability to profit in confirmed Bear regimes
- **Transaction cost sensitivity:** 41% monthly turnover makes the strategy sensitive to cost assumptions; at 25+ bps the edge would erode significantly

---

## References

1. Gunjan, A., Bhattacharyya, S. *A brief review of portfolio optimization techniques.* Artificial Intelligence Review **56**, 3847–3886 (2023). https://doi.org/10.1007/s10462-022-10273-7

2. Ni, Q., Yin, X., Tian, K. et al. *Particle swarm optimization with dynamic random population topology strategies for a generalized portfolio selection problem.* Natural Computing **16**, 31–44 (2017). https://doi.org/10.1007/s11047-016-9541-x

---

## License

This project is for academic purposes. All data sourced from Yahoo Finance via `yfinance`.
