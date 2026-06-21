"""Knowledge corpus builder.

Turns the engine's own artifacts into a retrievable corpus, so the assistant answers from
GROUNDED facts rather than the model's parametric memory:

1. Static methodology notes — the *why* behind FDR, Deflated Sharpe, Ledoit-Wolf, the SOR
   impact model, the double-entry ledger, and each audit checker. These are the war
   stories an interviewer would probe.
2. Dynamic run facts — rendered from a live ``PipelineOutput``: which factors passed FDR,
   IC/ICIR/p-values per factor, the Ledoit-Wolf shrinkage intensity, the audit severity,
   and the backtest summary. Re-grounding on each run keeps answers honest about *this*
   run rather than a generic story.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    """A source document for the index."""

    doc_id: str
    text: str
    source: str  # human-readable citation label


METHODOLOGY_DOCS: list[Document] = [
    Document(
        doc_id="method.multiple_testing",
        source="methodology: multiple-testing",
        text=(
            "Multiple-testing correction is the centerpiece of vnquant's alpha research. "
            "When you mine N candidate factors, some will look profitable purely by luck. "
            "Benjamini-Hochberg FDR controls the expected proportion of false discoveries "
            "among the factors you declare significant, at a configurable level fdr_alpha. "
            "The Deflated Sharpe Ratio (Bailey and Lopez de Prado) goes further: it "
            "discounts an observed Sharpe for the number of trials, the track-record "
            "length, and the skew and kurtosis of returns, returning the probability that "
            "the true Sharpe is positive. Reporting the best factor's raw Sharpe without "
            "these corrections is the classic data-snooping overfitting trap."
        ),
    ),
    Document(
        doc_id="method.information_coefficient",
        source="methodology: information coefficient",
        text=(
            "The Information Coefficient (IC) measures a factor's predictive power as the "
            "Spearman rank correlation between the factor value and forward returns. "
            "vnquant reports mean IC, ICIR (mean IC divided by its standard deviation, an "
            "information-ratio-like stability measure), and a t-stat with p-value. A delay "
            "parameter shifts the signal so today's factor predicts strictly future "
            "returns, preventing look-ahead bias."
        ),
    ),
    Document(
        doc_id="method.neutralization",
        source="methodology: neutralization",
        text=(
            "Neutralization strips market and sector exposure from a raw factor via "
            "cross-sectional OLS, so the factor is not merely re-expressing market beta or "
            "a sector tilt. After market-neutralization each cross-section sums to "
            "approximately zero. This isolates the idiosyncratic signal the factor claims "
            "to capture."
        ),
    ),
    Document(
        doc_id="method.risk_model",
        source="methodology: factor risk model",
        text=(
            "The structural factor risk model decomposes the covariance matrix as "
            "Sigma = B F B^T + D, where B is factor loadings, F is the factor covariance, "
            "and D is idiosyncratic variance. Ledoit-Wolf shrinkage pulls the sample factor "
            "covariance toward a structured target to stay robust on small samples; the "
            "shrinkage intensity lies in [0, 1] and is reported as a health signal. A high "
            "shrinkage value means the raw sample covariance was noisy and was heavily "
            "regularized."
        ),
    ),
    Document(
        doc_id="method.construction",
        source="methodology: portfolio construction",
        text=(
            "Portfolio construction selects the top-N names by score, sizes them by inverse "
            "volatility, applies a per-name weight cap, and scales the book to a target "
            "annualized volatility. Long-only weights are produced for the backtest."
        ),
    ),
    Document(
        doc_id="method.execution",
        source="methodology: smart order router",
        text=(
            "The Smart Order Router schedules a parent order via TWAP, VWAP, or POV, "
            "splitting it into lot-rounded child slices to minimize market impact in a thin "
            "market. It uses a square-root market-impact model: expected impact grows with "
            "the square root of the order size relative to average daily volume. Vietnam "
            "microstructure is first-class: the HOSE plus/minus 7 percent daily price band "
            "and 100-share lot rounding."
        ),
    ),
    Document(
        doc_id="method.ledger",
        source="methodology: double-entry ledger",
        text=(
            "Every trade event posts balanced debits and credits across Cash, Positions, "
            "Fees, and PnL accounts. The invariant that total debits equal total credits is "
            "checked after every transaction, so a balance bug surfaces instantly instead "
            "of drifting silently. This double-entry accounting is how the engine accounts "
            "for every cent of commission and slippage."
        ),
    ),
    Document(
        doc_id="audit.lookahead",
        source="audit war story: look-ahead via dropna",
        text=(
            "A forward-looking target must have NaNs at the tail because future prices do "
            "not exist yet. War story: a blanket df.dropna() once deleted exactly those "
            "tail rows, silently truncating the most recent sessions so inference read "
            "stale, month-old prices. The dangerous data bugs do not crash; they inflate. "
            "Fix: drop NaNs on feature columns only and keep target NaNs at the tail. "
            "Encoded in check_lookahead_target."
        ),
    ),
    Document(
        doc_id="audit.feature_completeness",
        source="audit war story: feature completeness",
        text=(
            "Rows with partial features must not be silently mislabeled. War story: legacy "
            "rows missing a key feature received a bogus default classification, producing "
            "a hallucinated 34 percent win-rate; stratifying by feature completeness "
            "revealed 62 percent for complete rows versus 27 percent for incomplete ones. "
            "Never trust an aggregate metric before stratifying by data quality. Encoded in "
            "check_feature_completeness."
        ),
    ),
    Document(
        doc_id="audit.signal_collapse",
        source="audit war story: model collapse",
        text=(
            "Detect model or signal collapse when output becomes near-constant (standard "
            "deviation below a threshold such as 0.02). War story: a collapsed model kept "
            "writing near-identical garbage into the store that fed future training. Cut a "
            "broken model off at the data-write boundary, not just the prediction boundary. "
            "Encoded in check_constant_signal."
        ),
    ),
    Document(
        doc_id="audit.temporal_survivorship",
        source="audit: temporal order and survivorship",
        text=(
            "The time index must be strictly increasing and free of duplicates, because "
            "out-of-order data breaks causality. The backtest universe should include "
            "delisted names, not just survivors, to avoid survivorship bias that flatters "
            "returns. Encoded in check_temporal_order and check_survivorship."
        ),
    ),
]


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def build_run_documents(out) -> list[Document]:
    """Render a live ``PipelineOutput`` into grounded, retrievable run-fact documents."""
    docs: list[Document] = []

    # Audit severity.
    docs.append(
        Document(
            doc_id="run.audit",
            source="run result: audit",
            text=(
                f"For this pipeline run the overall data-leakage audit severity is "
                f"{out.audit_severity.value}. Severity OK means no structural data problems "
                f"were detected; WARNING and CRITICAL indicate issues to investigate before "
                f"trusting results."
            ),
        )
    )

    # Risk shrinkage.
    docs.append(
        Document(
            doc_id="run.risk",
            source="run result: risk model",
            text=(
                f"For this run the Ledoit-Wolf shrinkage intensity of the factor covariance "
                f"is {_fmt(out.risk_shrinkage, 3)} on a 0 to 1 scale. Higher means the "
                f"sample covariance was noisier and was regularized more heavily toward the "
                f"structured target."
            ),
        )
    )

    # Per-factor IC + multiple-testing outcome.
    passed = [n for n, r in out.factor_reports.items() if r.survives_fdr]
    passed_txt = ", ".join(passed) if passed else "none"
    docs.append(
        Document(
            doc_id="run.factors.summary",
            source="run result: alpha research summary",
            text=(
                f"For this run, {len(passed)} of {len(out.factor_reports)} factors survived "
                f"Benjamini-Hochberg FDR correction. Factors passing FDR: {passed_txt}. On "
                f"purely synthetic near-random data, factors should not show a durable edge, "
                f"and the Deflated Sharpe probability is expected to be near zero for all of "
                f"them, which demonstrates the multiple-testing layer refusing to certify "
                f"luck as alpha."
            ),
        )
    )
    for name, rep in out.factor_reports.items():
        ic = rep.ic
        docs.append(
            Document(
                doc_id=f"run.factor.{name}",
                source=f"run result: factor {name}",
                text=(
                    f"Factor {name}: mean IC {_fmt(ic.mean_ic)}, ICIR {_fmt(ic.icir, 3)}, "
                    f"t-stat {_fmt(ic.t_stat, 2)}, p-value {_fmt(ic.p_value, 3)}. "
                    f"Survives FDR: {'yes' if rep.survives_fdr else 'no'}. "
                    f"Deflated Sharpe probability: {_fmt(rep.deflated_sr_prob, 2)}."
                ),
            )
        )

    # Backtest summary.
    s = out.backtest.summary()
    docs.append(
        Document(
            doc_id="run.backtest",
            source="run result: backtest summary",
            text=(
                f"Backtest summary for this run: final equity {s['final_equity']}, total "
                f"return {s['total_return_pct']} percent, Sharpe {s['sharpe']}, max drawdown "
                f"{s['max_drawdown_pct']} percent, over {s['trades']} trades. The showcase "
                f"backtest is unprofitable by construction because it trades textbook "
                f"factors on synthetic near-random data; it demonstrates the machinery and "
                f"guardrails, not a money-making strategy."
            ),
        )
    )
    return docs


def build_corpus(out=None) -> list[Document]:
    """Full corpus: static methodology plus dynamic run facts when an output is given."""
    docs = list(METHODOLOGY_DOCS)
    if out is not None:
        docs.extend(build_run_documents(out))
    return docs
