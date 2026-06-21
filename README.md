<h1 align="center">vnquant</h1>

<p align="center">
  <b>An institutional-grade quantitative research engine for the Vietnam equity market —<br>
  alpha research, factor risk modelling, smart execution, and a grounded RAG assistant,<br>
  wired into one reproducible pipeline.</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-33%20passing-brightgreen.svg">
  <img alt="Lint" src="https://img.shields.io/badge/ruff-clean-success.svg">
  <img alt="Type" src="https://img.shields.io/badge/typed-dataclasses-informational.svg">
  <img alt="Status" src="https://img.shields.io/badge/data-synthetic%20%C2%B7%20reproducible-orange.svg">
</p>

> **Portfolio / showcase edition.** This repository demonstrates the *engineering
> architecture* of a full quantitative trading research stack. It runs entirely on
> **synthetic data** with **textbook example factors** — no proprietary alphas, calibrated
> parameters, trained model weights, broker credentials, or live track records are
> included. Shipping synthetic data is a deliberate choice: the value on display is the
> **machinery, the statistical discipline, and the guardrails**, not a money-making
> strategy. See [LICENSE](LICENSE).

`vnquant` is a compact, end-to-end quant research engine modelled on how an institutional
desk separates concerns: rigorous alpha research with multiple-testing correction, a
structural factor risk model, risk-based portfolio construction, a smart order router,
double-entry trade accounting, a data-leakage audit framework, an event-driven
backtester, and a **retrieval-augmented research assistant** — wired together into one
reproducible pipeline.

It targets the **Vietnam equity market** specifically: the ±7% HOSE daily price band,
100-share lot rounding, and thin-liquidity execution are first-class concerns throughout.

---

## Highlights

- 🧪 **Statistical discipline first.** Benjamini-Hochberg FDR + the Deflated Sharpe Ratio
  correct for data-snooping when many factors are tested — the single most important habit
  separating real quant research from curve-fitting.
- 📉 **Structural factor risk model** `Σ = B·F·Bᵀ + D` with **Ledoit-Wolf shrinkage** for
  small-sample robustness, feeding volatility-targeted portfolio construction.
- 🛡️ **Data-leakage audit framework** encoding real production war stories: look-ahead,
  survivorship, feature-completeness, and model-collapse checkers.
- 🤖 **Grounded RAG assistant** over the engine's own run-facts, with a two-gate
  **hallucination guard** that *abstains* when context is insufficient — offline by
  default, OpenAI-pluggable.
- 🇻🇳 **Vietnam microstructure-aware** execution: ±7% HOSE band, 100-share lots, and a
  square-root market-impact model in a thin market.
- 🔁 **Fully reproducible & tested.** Seeded RNG end-to-end, **33 passing tests**, clean
  `ruff`, double-entry books that provably balance after every run.

---

## Why this exists

Most personal "trading bot" projects are a single indicator wrapped in a loop. The hard
part of real quant work isn't the signal — it's the **discipline** that stops you from
fooling yourself: correcting for data-snooping when you test many factors, building a
covariance matrix that doesn't blow up on small samples, accounting for every cent so
balance drift can't hide, auditing your data so a silent leak doesn't inflate your
backtest, and — when you bolt an LLM on top — refusing to answer when the retrieved
context doesn't support it.

This repo shows that machinery, built correctly, on safe synthetic data.

---

## Framework

The engine is organized as a set of independent stages, each in its own subpackage.

### Data layer (`vnquant/data`)
Generates a realistic OHLCV panel with an embedded **market + sector + idiosyncratic**
factor structure, so the alpha and risk stages have genuine signal to discover. Respects
the HOSE ±7% daily band and 100-share lots.

### Alpha research (`vnquant/alpha`)
- **Operators** — a vectorized, WorldQuant-style vocabulary (`ts_rank`, `cs_zscore`,
  `winsorize`, `ts_corr`, …) for composing cross-sectional signals.
- **Neutralization** — strips market and sector exposure via cross-sectional OLS so a
  factor isn't just re-expressing beta or a sector tilt.
- **Information Coefficient** — Spearman rank IC, ICIR, t-stat, with an explicit `delay`
  to prevent look-ahead.
- **Multiple-testing correction** — **Benjamini-Hochberg FDR** + the **Deflated Sharpe
  Ratio** (Bailey & López de Prado). This is the centerpiece: when you mine N factors,
  some look good by luck, and these methods discount for that.
- **Library** — classic, publicly-documented example factors (momentum, short-term
  reversal, low-volatility, volume-trend). The proprietary mined factors that constitute
  real edge are deliberately *not* shipped.

### Risk model & portfolio construction (`vnquant/risk`)
- **Structural factor risk model** — `Σ = B·F·Bᵀ + D`, with **Ledoit-Wolf shrinkage** on
  the factor covariance for small-sample robustness.
- **Construction** — top-N selection, inverse-volatility sizing, per-name caps, and
  scaling to a target portfolio volatility.

### Execution (`vnquant/execution`)
- **Smart Order Router** — TWAP / VWAP / POV scheduling with a **square-root market-impact
  model**, splitting a parent order into lot-rounded child slices to minimize impact in a
  thin market.
- **Double-entry ledger** — every event posts balanced debits/credits across Cash /
  Positions / Fees / PnL accounts. The invariant `Σ debits == Σ credits` is checked after
  every transaction, so balance bugs surface instantly instead of drifting silently.

### Audit framework (`vnquant/audit`)
Pluggable checkers for the bias classes that quietly destroy backtests: **look-ahead**
(forward-looking targets must be NaN at the tail), **temporal ordering**, **survivorship**,
**feature-completeness** (don't trust a win-rate computed across mislabeled rows), and
**signal collapse** (a near-constant model that should be quarantined before it poisons
training data). These checkers encode real war stories — see [Lessons learned](#lessons-learned).

### Backtest (`vnquant/backtest`)
An **event-driven** simulator that walks the panel one bar at a time, rebuilds the risk
model on a trailing window at each rebalance, constructs target weights, and routes the
implied trades through the ledger with commission and slippage. Event-driven execution
makes look-ahead structurally hard — the engine can only ever see data up to the current bar.

### Research assistant — RAG (`vnquant/assistant`)
A **retrieval-augmented generation** layer that answers natural-language questions about
the engine — *grounded* on its own methodology notes and the **live facts of a pipeline
run** (which factors passed FDR, per-factor IC/p-values, Ledoit-Wolf shrinkage, audit
severity, backtest summary) rather than on a model's parametric memory. The full RAG
machinery is here: word-window **chunking** with overlap, an **embedder** (offline
deterministic hashing embedder, or OpenAI embeddings), a cosine **vector store** +
top-k **retriever**, and a **grounded generator** (offline extractive, or an OpenAI chat
model under a strict "answer only from context" prompt).

The centerpiece is the **hallucination guard**: before generating, the assistant applies
two gates — a semantic cosine floor *and* a lexical content-term overlap with the query —
and **abstains** ("I don't have enough grounded context") when either fails. This is the
generation-edge twin of the audit framework: both exist to stop the system from fooling
its user. It runs **fully offline by default** (no API key), and swaps to a real LLM
backend with one config flag.

```bash
# grounded on a live pipeline run, fully offline:
vnquant ask "How many factors passed FDR and what was the audit severity?"
vnquant ask "Why use the Deflated Sharpe Ratio?" --no-ground   # methodology only
vnquant ask "..." --backend openai                             # real embeddings + LLM
```

---

## Installation and CLI

### Installation

```bash
git clone <your-fork-url> vn-quant-engine
cd vn-quant-engine

# with uv (recommended)
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

# optional: real LLM backend for the research assistant (OpenAI)
uv pip install -e ".[dev,llm]"

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

No API keys or credentials are required — the showcase generates its own data.

### CLI usage

```bash
vnquant --help

vnquant research      # alpha research loop: IC / FDR / Deflated-Sharpe table
vnquant backtest      # event-driven backtest performance summary
vnquant audit         # run the data-leakage audit checks
vnquant pipeline      # full end-to-end run (data → audit → alpha → risk → backtest)
vnquant ask "..."     # RAG research assistant, grounded on a live run (abstains if unsure)

# reproducibility: every command takes a seed
vnquant research --seed 7
```

Example `vnquant research` output:

```
            Alpha Research — IC / Multiple-Testing
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃           Factor ┃ mean IC ┃   ICIR ┃ t-stat ┃ p-value ┃ FDR pass ┃ DSR prob ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│   high_low_range │ -0.0199 │ -0.114 │  -3.09 │   0.002 │      yes │     0.00 │
│     volume_trend │ +0.0112 │ +0.065 │  +1.76 │   0.078 │       no │     0.00 │
│    momentum_12_1 │ -0.0108 │ -0.063 │  -1.69 │   0.091 │       no │     0.00 │
│   low_volatility │ +0.0031 │ +0.018 │  +0.47 │   0.636 │       no │     0.00 │
│ short_term_reve… │ -0.0020 │ -0.011 │  -0.30 │   0.762 │       no │     0.00 │
└──────────────────┴─────────┴────────┴────────┴─────────┴──────────┴──────────┘
```

> **Reading the numbers honestly:** on purely synthetic, near-random data, factors should
> *not* exhibit a durable edge — and they don't. Only one survives FDR (a weak, likely
> spurious finding), and the **Deflated Sharpe probability is ≈ 0 for all of them**, which
> is exactly the point: the multiple-testing layer correctly refuses to certify luck as
> alpha. Likewise the demo `backtest` is unprofitable by construction. This repo
> showcases the *machinery and the guardrails*, not a money-making strategy.

---

## Package usage

```python
from vnquant.default_config import Config
from vnquant.pipeline import run_pipeline

cfg = Config()
cfg.data.seed = 42
out = run_pipeline(cfg)

print(out.audit_severity)                 # OK / WARNING / CRITICAL
print(out.risk_shrinkage)                 # Ledoit-Wolf intensity
print(out.backtest.summary())             # final equity, return, Sharpe, max DD, trades
out.backtest.ledger.check_invariant()     # double-entry books balance
```

Use individual stages directly:

```python
from vnquant.data import generate_panel, to_wide
from vnquant.alpha import research
from vnquant.risk import fit_factor_risk_model, construct_portfolio
from vnquant.execution import route, Ledger

panel, sym2sector = generate_panel()
close = to_wide(panel, "close")
reports = research(close, to_wide(panel, "volume"),
                   to_wide(panel, "high"), to_wide(panel, "low"), sym2sector)
```

---

## Project layout

```
vn-quant-engine/
├── vnquant/
│   ├── data/            # synthetic OHLCV panel generation
│   ├── alpha/           # operators, neutralizer, IC, multiple-testing, research loop
│   ├── risk/            # factor risk model (Ledoit-Wolf), portfolio construction
│   ├── execution/       # smart order router, double-entry ledger
│   ├── audit/           # leakage / look-ahead / survivorship / collapse checkers
│   ├── backtest/        # event-driven simulator
│   ├── assistant/       # RAG: chunking, embeddings, vector store, grounded generator
│   ├── pipeline.py      # end-to-end orchestrator
│   └── default_config.py
├── cli/                 # typer + rich command-line interface
├── tests/               # pytest suite
├── main.py
├── pyproject.toml
└── README.md
```

---

## Reproducibility

Every entry point accepts a `--seed`, and all synthetic data is generated from a seeded
NumPy `default_rng`. Two runs with the same seed produce identical IC tables, risk
estimates, and backtest curves. The test suite asserts the double-entry ledger balances
after a full pipeline run.

```bash
pytest -q          # 33 tests
```

---

## Lessons learned

These checkers and design choices come from debugging a real system. Each is a war story
worth telling in an interview:

- **Silent data leak via `dropna()`.** A blanket `df.dropna()` once deleted the rows where
  a forward-looking target was still NaN — i.e. the most recent sessions — silently
  truncating the panel so inference read month-old prices. Fix: drop NaNs on *feature*
  columns only. Lesson: the dangerous data bugs don't crash; they inflate. → encoded in
  `audit.check_lookahead_target`.
- **Hallucinated win-rate from feature incompleteness.** Legacy rows missing a key feature
  got a bogus default label, producing a fake 34% win-rate; stratifying by completeness
  revealed 62% vs 27%. Lesson: never trust an aggregate metric before stratifying by data
  quality. → `audit.check_feature_completeness`.
- **Model collapse poisoning the training set.** A collapsed model emitted near-constant
  output (std < 0.02). Quarantining it at the *output* wasn't enough — it kept writing
  garbage into the store that fed future training. Lesson: cut a broken model off at the
  data-write boundary, not just the prediction boundary. → `audit.check_constant_signal`.
- **Multiple-testing is not optional.** Mining many factors and reporting the best one's
  Sharpe is the classic overfitting trap. BH-FDR + Deflated Sharpe are cheap insurance. →
  `alpha.multiple_testing`.
- **An LLM that always answers will eventually lie.** Bolting RAG onto a quant engine is
  easy; making it *refuse* when retrieval is weak is the hard, important part. The
  assistant gates generation on both a semantic score and a lexical-overlap check, and
  abstains otherwise — the generation-edge twin of the data audit. → `assistant.rag`.

---

## Skills demonstrated

A quick map from this codebase to what an **AI / ML Engineer** role in banking and
financial services typically looks for:

| Area | What's in the repo | Where |
|------|--------------------|-------|
| **Applied ML / statistics** | Spearman IC, ICIR, t-stats; Benjamini-Hochberg FDR; Deflated Sharpe Ratio; cross-sectional OLS neutralization | `vnquant/alpha` |
| **Risk modelling** | Structural factor covariance `Σ = B·F·Bᵀ + D`, Ledoit-Wolf shrinkage, PSD guarantees | `vnquant/risk` |
| **GenAI / RAG** | Chunking, embeddings, vector store, retrieval, grounded generation, **hallucination guard with abstention** | `vnquant/assistant` |
| **MLOps / data quality** | Leakage / look-ahead / survivorship / feature-completeness / model-collapse auditing | `vnquant/audit` |
| **Software engineering** | Typed dataclasses, clean package boundaries, dependency injection (pluggable backends), Typer CLI, 33 pytest tests, clean `ruff` | repo-wide |
| **Domain knowledge** | Vietnam market microstructure, transaction-cost & market-impact modelling, double-entry accounting invariants | `vnquant/execution` |
| **Reproducibility** | Seeded RNG everywhere; identical IC tables, risk estimates, and equity curves across runs | `--seed` flag |

> **A note for reviewers:** this is a clean-room showcase derived from a larger private
> trading system. The architecture, statistical methods, audit checkers, and the RAG
> assistant are real engineering; the data and factors are deliberately synthetic and
> public so nothing proprietary is exposed.

---

## Disclaimer

This is an engineering showcase, not investment advice. It ships synthetic data and
textbook factors only. Trading financial instruments carries risk of loss.
