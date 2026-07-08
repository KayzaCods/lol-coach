"""Learn HEF weights from pairwise preferences (preference learning).

Idea (neuro-evolutionary preference learning, Yannakakis et al.): instead of
asking the user for an absolute score, ask which of two comparable decisions
was played better. From many such judgments we fit the HEF weights ω so that
HEF(winner) > HEF(loser) as often as possible.

Model: Bradley-Terry / RankNet with a linear scorer over the HEF terms.
    P(A ≻ B) = σ( w · (terms_A − terms_B) )
Fit w by maximizing the likelihood of the observed preferences (gradient
descent), constrained to w ≥ 0, then normalized to sum 1 so the result is a
drop-in replacement for the HEF weight dict. No external dependencies.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

# Same dimensions/order as features._HEF_WEIGHTS.
DIMS = ["power", "info", "action_fit", "wave"]
DEFAULT_WEIGHTS = {"power": 0.35, "info": 0.30, "action_fit": 0.25, "wave": 0.10}
MIN_PAIRS = 8  # below this, learned weights are reported but not trustworthy


REG_LAMBDA = 0.08  # pull toward DEFAULT_WEIGHTS: with little/conflicting evidence the
                   # solution stays near the prior instead of collapsing dims to 0


def _delta(terms_hi: dict, terms_lo: dict) -> list[float]:
    """terms(winner) − terms(loser); a dim missing on EITHER side votes 0 for that
    dim (no signal) — the old 0.5 imputation turned "unknown" into fake signal."""
    out = []
    for d in DIMS:
        hi, lo = terms_hi.get(d), terms_lo.get(d)
        out.append((hi - lo) if (hi is not None and lo is not None) else 0.0)
    return out


def _rows_from_pairs(pairs) -> tuple[list, int]:
    """Normalize (terms_a, terms_b, winner[, source]) into (delta_row, source).

    The winner's terms go first so every usable row should score z > 0.
    Rows with no signal at all (all-zero delta) are excluded from fit AND
    accuracy — degenerate (empty/identical terms) judgments are pure noise."""
    rows = []
    n_no_signal = 0
    for item in pairs:
        ta, tb, win = item[0], item[1], item[2]
        source = item[3] if len(item) > 3 else "pref"
        if win == "a":
            row = _delta(ta, tb)
        elif win == "b":
            row = _delta(tb, ta)
        else:
            continue  # 'tie' contributes no ordering signal
        if all(v == 0.0 for v in row):
            n_no_signal += 1
            continue
        rows.append((row, source))
    return rows, n_no_signal


def _row_weights(rows: list) -> list[float]:
    """Per-row fit weights, mean 1.0.

    With BOTH feedback sources present ('pref' = Calibrar pairwise, 'dec' =
    per-decision best-option marks) each source contributes EQUAL total mass:
    dec rows only carry action_fit signal (state is held fixed by construction)
    and typically outnumber pref rows several times over — unweighted they
    would drown the full-state judgments. With a single source: plain 1.0."""
    sources = sorted({s for _, s in rows})
    if len(sources) <= 1:
        return [1.0] * len(rows)
    n_by = {s: sum(1 for _, x in rows if x == s) for s in sources}
    return [len(rows) / (len(sources) * n_by[s]) for _, s in rows]


def _fit(rows: list, weights: list[float], iters: int, lr: float) -> list[float]:
    """Weighted Bradley-Terry MAP fit. Optimize around the prior: start AT the
    prior (scaled to sum=4 like the old init) and regularize toward it."""
    w0 = [DEFAULT_WEIGHTS[d] * 4.0 for d in DIMS]
    w = list(w0)
    total = sum(weights) or 1.0
    for _ in range(iters):
        grad = [0.0] * len(DIMS)
        for (row, _s), wt in zip(rows, weights):
            z = sum(w[j] * row[j] for j in range(len(DIMS)))
            p = 1.0 / (1.0 + math.exp(-z))
            for j in range(len(DIMS)):
                grad[j] += wt * (p - 1.0) * row[j]      # d(−log σ(z))/dw_j
        w = [max(0.0, w[j] - lr * (grad[j] / total + REG_LAMBDA * (w[j] - w0[j])))
             for j in range(len(DIMS))]
    return w


def _acc(weight_vec: list[float], rows: list, source: str | None = None):
    """Fraction of rows the weights order correctly. Score only rows where the
    weights produce a signal: z of exactly 0 (e.g. action-only pairs under
    state-only weights) is "can't judge", not "wrong" — counting those as
    misses is what made accuracy unreadable."""
    ok = scored = 0
    for row, s in rows:
        if source is not None and s != source:
            continue
        z = sum(weight_vec[j] * row[j] for j in range(len(DIMS)))
        if z == 0:
            continue
        scored += 1
        if z > 0:
            ok += 1
    return round(ok / scored, 3) if scored else None


CV_ITERS = 300   # CV refits converge fast thanks to the prior anchor.
CV_SEED = 13     # deterministic folds: same data -> same reported accuracy


def _cv_accuracy(rows: list, lr: float) -> tuple[float | None, int]:
    """Cross-validated accuracy: leave-one-out up to 40 rows, 10-fold beyond.

    In-sample accuracy flatters small samples (the fit saw every row it is
    graded on); the guardrail in apply_weights must trust THIS number instead.
    Returns (accuracy, n_scored)."""
    n = len(rows)
    if n < 3:
        return None, 0
    idx = list(range(n))
    random.Random(CV_SEED).shuffle(idx)
    k = n if n <= 40 else 10
    folds = [idx[i::k] for i in range(k)]
    ok = scored = 0
    for fold in folds:
        test = set(fold)
        train = [rows[i] for i in range(n) if i not in test]
        if not train:
            continue
        w = _fit(train, _row_weights(train), CV_ITERS, lr)
        for i in fold:
            row, _s = rows[i]
            z = sum(w[j] * row[j] for j in range(len(DIMS)))
            if z == 0:
                continue
            scored += 1
            if z > 0:
                ok += 1
    return (round(ok / scored, 3) if scored else None), scored


def learn_weights(pairs: list, iters: int = 800, lr: float = 0.5) -> dict | None:
    """pairs: list of (terms_a, terms_b, winner in {'a','b','tie'}) or
    (terms_a, terms_b, winner, source) with source in {'pref', 'dec'}.

    MAP estimate: Bradley-Terry likelihood + L2 prior centered on DEFAULT_WEIGHTS
    (so dims only leave the prior under real evidence; no more info=0 artifacts).
    Sources are balanced (see _row_weights) and accuracy is reported three ways:
    in-sample, per source (measures whether the two sources actually disagree),
    and cross-validated (the honest number — used by the apply guardrail).
    Returns {weights, n, n_no_signal, n_by_source, accuracy, accuracy_by_source,
    accuracy_cv, cv_scored, accuracy_default, enough} or None.
    """
    rows, n_no_signal = _rows_from_pairs(pairs)
    if not rows:
        return None

    w = _fit(rows, _row_weights(rows), iters, lr)
    s = sum(w) or 1.0
    weights = {DIMS[j]: round(w[j] / s, 3) for j in range(len(DIMS))}

    w_def = [DEFAULT_WEIGHTS[d] for d in DIMS]
    sources = sorted({src for _, src in rows})
    acc_by_source = {src: _acc(w, rows, src) for src in sources} if len(sources) > 1 else {}
    acc_cv, cv_scored = _cv_accuracy(rows, lr)

    return {
        "weights": weights,
        "n": len(rows),
        "n_no_signal": n_no_signal,
        "n_by_source": {src: sum(1 for _, x in rows if x == src) for src in sources},
        "accuracy": _acc(w, rows),
        "accuracy_by_source": acc_by_source,
        "accuracy_cv": acc_cv,
        "cv_scored": cv_scored,
        # The default weights involve no fitting, so their in-sample accuracy IS
        # their out-of-sample accuracy — directly comparable with accuracy_cv.
        "accuracy_default": _acc(w_def, rows),
        "enough": len(rows) >= MIN_PAIRS,
    }


# ----------------------------------------------------- persisted active weights
def weights_path(db_path: str | Path) -> Path:
    """hef_weights.json lives next to the SQLite DB."""
    return Path(db_path).parent / "hef_weights.json"


def load_weights(db_path: str | Path) -> dict | None:
    p = weights_path(db_path)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if all(k in d for k in DIMS):
                return {k: float(d[k]) for k in DIMS}
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    return None


def save_weights(db_path: str | Path, weights: dict) -> None:
    weights_path(db_path).write_text(
        json.dumps({k: round(float(weights[k]), 3) for k in DIMS}, indent=2),
        encoding="utf-8",
    )
