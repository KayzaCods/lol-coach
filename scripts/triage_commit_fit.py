"""Triage: where _commit_fit (the action_fit term) contradicts the player.

Measured 2026-06-10: of the valid per-decision marks (user_best_option, content-
fresh by fingerprint), _commit_fit ranks the player's chosen-best action ABOVE
the taken one only ~40% of the time — the formula disagrees with the player's
explicit corrections more often than it agrees. This script turns that number
into something actionable:

  1. CASES: every valid mark where best != taken, classified agree/disagree,
     with the state the formula saw (power_index, info term, setup, objective
     value) and the fit it assigned to each action. Disagreements are the
     review queue: each one is either a formula bug or a mark worth rethinking.
  2. TRANSITIONS: which action pairs (taken -> best) concentrate the
     disagreement — tells WHICH branch of _commit_fit is miscalibrated.
  3. SENSITIVITY: sweep the formula's constants (info/power blend, objective
     scaling, branch baselines) and score every combination on BOTH metrics:
     agreement with the marks (dec) AND Calibrar pairwise accuracy under
     default weights (pref) — a knob that helps dec by hurting pref is
     overfitting one source. Reports the Pareto-best single-knob moves and
     the best joint combo. NOTHING is changed automatically: the output is
     evidence for a deliberate edit of features._commit_fit.

Reads the DB only. Writes data/commit_fit_triage.md next to the DB.

    .venv\\Scripts\\python.exe scripts\\triage_commit_fit.py
    .venv\\Scripts\\python.exe scripts\\triage_commit_fit.py --db ruta\\a\\copia.db
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lol_coach.config import load_config
from lol_coach.decisions.features import (
    _AGGRESSIVE_ACTIONS,
    _DEFENSIVE_ACTIONS,
    _info_term,
    action_fingerprint,
    hef_score,
    power_index,
)
from lol_coach.preference import DEFAULT_WEIGHTS, DIMS

from dashboard import _NOISE_NOTE, _pref_ids  # same gates as the learner


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ------------------------------------------------------- parametric commit_fit
# Mirror of features._commit_fit with its constants exposed. Keep in sync by
# hand; the triage asserts the default params reproduce the live function.
P_DEFAULT = {
    "alpha": 0.3,        # info share in setup = alpha*it + (1-alpha)*pi
                         # (recalibrated 2026-06-11 from the player's marks; was 0.6)
    "agg_base": 0.7,     # aggressive fit = setup * (agg_base + agg_slope*ov)
    "agg_slope": 0.6,
    "ign_pen": 0.4,      # ignore_objective: 1-setup - ign_pen*ov*setup
    "vis_base": 0.6,     # check_map/vision_setup: vis_base + (1-vis_base)*(1-setup)
}


def commit_fit_p(action_id, pi, it, objectives, p) -> float | None:
    if action_id is None:
        return None
    if it is not None and pi is not None:
        setup = p["alpha"] * it + (1 - p["alpha"]) * pi
    elif it is not None:
        setup = it
    elif pi is not None:
        setup = pi
    else:
        return None
    ov = (objectives or {}).get("objective_value")
    if action_id in _AGGRESSIVE_ACTIONS:
        fit = setup * (p["agg_base"] + p["agg_slope"] * ov) if isinstance(ov, (int, float)) else setup
        return _clamp01(fit)
    if action_id == "ignore_objective":
        base = 1 - setup
        if isinstance(ov, (int, float)):
            base -= p["ign_pen"] * ov * setup
        return _clamp01(base)
    if action_id in _DEFENSIVE_ACTIONS:
        return _clamp01(1 - setup)
    if action_id in ("check_map", "vision_setup"):
        return _clamp01(p["vis_base"] + (1 - p["vis_base"]) * (1 - setup))
    return 0.5


# ------------------------------------------------------------------ data load
def _valid_dec_cases(conn) -> tuple[list[dict], dict]:
    """Marks that pass the SAME gates as the learner (_decision_feedback_pairs),
    kept rich for review. Returns (cases, excluded_counts)."""
    cases, excl = [], collections.Counter()
    for r in conn.execute(
        "SELECT d.id, d.match_id, d.detector_id, d.game_time_ms, d.moment, "
        "d.context_json, d.options_json, d.user_feedback, d.user_feedback_note, "
        "d.user_best_option, d.user_feedback_at_utc, d.created_at_utc, "
        "d.user_feedback_fingerprint, m.game_start_ms "
        "FROM decisions d JOIN matches m ON m.match_id = d.match_id "
        "WHERE d.user_best_option IS NOT NULL"
    ):
        if _NOISE_NOTE.search(r["user_feedback_note"] or ""):
            excl["ruido"] += 1
            continue
        if r["user_feedback"] == "equivalent":
            excl["equivalent"] += 1
            continue
        ctx = json.loads(r["context_json"])
        action = ctx.get("action") or {}
        fp = r["user_feedback_fingerprint"]
        if fp:
            if fp != action_fingerprint(r["detector_id"], action, json.loads(r["options_json"])):
                excl["stale_contenido"] += 1
                continue
        elif (r["user_feedback_at_utc"] and r["created_at_utc"]
              and r["user_feedback_at_utc"] < r["created_at_utc"]):
            excl["stale_timestamp"] += 1
            continue
        avail = action.get("available_actions") or []
        taken_aid = action.get("action_id")
        bo = r["user_best_option"]
        best_aid = avail[bo].get("action_id") if (avail and 0 <= bo < len(avail)) else None
        if not best_aid or not taken_aid:
            excl["sin_accion"] += 1
            continue
        if best_aid == taken_aid:
            excl["misma_accion"] += 1
            continue
        sf = ctx.get("state_features") or {}
        pi = power_index(sf.get("power") or {})
        it = _info_term(sf.get("info_risk") or {})
        cases.append({
            "id": r["id"], "match": r["match_id"], "detector": r["detector_id"],
            "t_ms": r["game_time_ms"], "moment": r["moment"] or "",
            "note": (r["user_feedback_note"] or "").strip(),
            "taken": taken_aid, "best": best_aid,
            "pi": pi, "it": it, "objectives": sf.get("objectives") or {},
        })
    return cases, dict(excl)


def _pref_rows(conn) -> list[dict]:
    """Clean Calibrar pairs with what's needed to re-score action_fit under
    modified params: per side, the full current terms + (pi, it, objectives,
    taken action_id) so action_fit can be recomputed."""
    def side(did):
        r = conn.execute("SELECT detector_id, context_json FROM decisions WHERE id=?", (did,)).fetchone()
        if not r:
            return None
        ctx = json.loads(r["context_json"])
        sf = ctx.get("state_features") or {}
        aid = (ctx.get("action") or {}).get("action_id")
        terms = hef_score(sf, aid).get("terms") or {}
        if not terms:
            return None
        return {
            "terms": terms, "aid": aid,
            "pi": power_index(sf.get("power") or {}),
            "it": _info_term(sf.get("info_risk") or {}),
            "objectives": sf.get("objectives") or {},
        }

    rows = []
    for p in conn.execute(
        "SELECT id, decision_a_id, decision_b_id, winner, note, a_key, b_key FROM preferences"
    ):
        if p["winner"] not in ("a", "b"):
            continue
        if _NOISE_NOTE.search(p["note"] or ""):
            continue
        a_id, b_id = _pref_ids(conn, p)
        a, b = side(a_id), side(b_id)
        if not a or not b:
            continue
        rows.append({"win": p["winner"], "a": a, "b": b})
    return rows


# ------------------------------------------------------------------- metrics
def dec_agreement(cases, p) -> tuple[float | None, int, int, int]:
    """(rate, n_agree, n_tie, n_total): does fit(best) beat fit(taken)?"""
    agree = tie = total = 0
    for c in cases:
        fb = commit_fit_p(c["best"], c["pi"], c["it"], c["objectives"], p)
        ft = commit_fit_p(c["taken"], c["pi"], c["it"], c["objectives"], p)
        if fb is None or ft is None:
            continue
        total += 1
        if fb > ft:
            agree += 1
        elif fb == ft:
            tie += 1
    return (round(agree / total, 3) if total else None), agree, tie, total


def pref_accuracy(rows, p) -> float | None:
    """Calibrar accuracy under DEFAULT weights with action_fit recomputed
    using params p (other terms untouched). z==0 rows don't count."""
    ok = scored = 0
    for r in rows:
        ta, tb = dict(r["a"]["terms"]), dict(r["b"]["terms"])
        fa = commit_fit_p(r["a"]["aid"], r["a"]["pi"], r["a"]["it"], r["a"]["objectives"], p)
        fb = commit_fit_p(r["b"]["aid"], r["b"]["pi"], r["b"]["it"], r["b"]["objectives"], p)
        if fa is not None:
            ta["action_fit"] = fa
        if fb is not None:
            tb["action_fit"] = fb
        hi, lo = (ta, tb) if r["win"] == "a" else (tb, ta)
        z = sum(
            DEFAULT_WEIGHTS[d] * (hi[d] - lo[d])
            for d in DIMS if d in hi and d in lo
        )
        if z == 0:
            continue
        scored += 1
        if z > 0:
            ok += 1
    return round(ok / scored, 3) if scored else None


# ------------------------------------------------------------------- report
GRID = {
    "alpha": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "agg_base": [0.5, 0.6, 0.7, 0.8],
    "agg_slope": [0.0, 0.3, 0.6, 0.9],
    "ign_pen": [0.0, 0.2, 0.4, 0.6],
    "vis_base": [0.4, 0.5, 0.6, 0.7],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Triaje de _commit_fit vs marcas del jugador.")
    ap.add_argument("--db", default=None, help="ruta de la DB (por defecto: config.toml)")
    args = ap.parse_args()
    cfg = load_config()
    db_path = Path(args.db) if args.db else Path(cfg["paths"]["sqlite_db"])
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Sanity: the parametric mirror must reproduce the live formula.
    probe = {"pi": 0.62, "it": 0.31, "objectives": {"objective_value": 0.6}}
    from lol_coach.decisions.features import _commit_fit as live_fit
    for aid in ("fight_entry", "disengage", "ignore_objective", "vision_setup", "check_map", "roam"):
        a = live_fit(aid, probe["pi"], probe["it"], probe["objectives"])
        b = commit_fit_p(aid, probe["pi"], probe["it"], probe["objectives"], P_DEFAULT)
        assert a is not None and abs(a - b) < 1e-9, f"espejo desincronizado para {aid}: {a} vs {b}"

    cases, excl = _valid_dec_cases(conn)
    prefs = _pref_rows(conn)
    if not cases:
        print("No hay marcas válidas (¿corriste backfill_feedback_fingerprints.py --apply?).")
        return 1

    base_rate, n_agree, n_tie, n_total = dec_agreement(cases, P_DEFAULT)
    base_pref = pref_accuracy(prefs, P_DEFAULT)

    lines = []
    w = lines.append
    w("# Triaje de _commit_fit vs marcas del jugador")
    w("")
    w(f"Casos válidos (mismas puertas que el learner): **{n_total}** "
      f"(excluidos: {excl or 'ninguno'})")
    w(f"Con la fórmula ACTUAL: acuerdo **{n_agree}/{n_total}** ({base_rate}), empates {n_tie}. "
      f"Accuracy Calibrar (pesos default): **{base_pref}** sobre {len(prefs)} pares.")
    w("")

    # Transitions
    trans = collections.Counter()
    trans_dis = collections.Counter()
    for c in cases:
        fb = commit_fit_p(c["best"], c["pi"], c["it"], c["objectives"], P_DEFAULT)
        ft = commit_fit_p(c["taken"], c["pi"], c["it"], c["objectives"], P_DEFAULT)
        key = f"{c['taken']} -> {c['best']}"
        trans[key] += 1
        if fb is not None and ft is not None and fb <= ft:
            trans_dis[key] += 1
    w("## Dónde se concentra el desacuerdo (tomada -> que debió ser)")
    w("")
    w("| transición | casos | en desacuerdo |")
    w("|---|---|---|")
    for k, n in trans.most_common():
        w(f"| `{k}` | {n} | {trans_dis.get(k, 0)} |")
    w("")

    # Case list (disagreements first)
    def fits(c):
        return (commit_fit_p(c["taken"], c["pi"], c["it"], c["objectives"], P_DEFAULT),
                commit_fit_p(c["best"], c["pi"], c["it"], c["objectives"], P_DEFAULT))

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    w("## Casos en desacuerdo (cola de revisión: ¿bug de fórmula o marca a repensar?)")
    w("")
    w("| dec | detector | momento | tomada | mejor (tuya) | fit tomada | fit mejor | power | info | obj_val | nota |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    n_listed = 0
    for c in sorted(cases, key=lambda c: (c["detector"], c["t_ms"])):
        ft, fb = fits(c)
        if ft is None or fb is None or fb > ft:
            continue
        ov = (c["objectives"] or {}).get("objective_value")
        w(f"| {c['id']} | {c['detector'].replace('_v1','')} | {c['moment'][:38]} | "
          f"`{c['taken']}` | `{c['best']}` | {fmt(ft)} | {fmt(fb)} | "
          f"{fmt(c['pi'])} | {fmt(c['it'])} | {fmt(ov)} | {c['note'][:60]} |")
        n_listed += 1
    w("")
    w(f"({n_listed} casos. El id es el de la decisión en el dashboard.)")
    w("")

    # Sensitivity: single-knob marginals + full grid
    w("## Sensibilidad de parámetros")
    w("")
    w(f"Métrica doble: acuerdo con tus marcas (dec) y accuracy Calibrar (pref). "
      f"Base actual: dec {base_rate} / pref {base_pref}.")
    w("")
    w("### Mover UNA perilla (resto en default)")
    w("")
    w("| perilla | valor | dec | pref |")
    w("|---|---|---|---|")
    for knob, values in GRID.items():
        for v in values:
            p = dict(P_DEFAULT); p[knob] = v
            r, *_ = dec_agreement(cases, p)
            pa = pref_accuracy(prefs, p)
            mark = " ←default" if abs(v - P_DEFAULT[knob]) < 1e-9 else ""
            w(f"| {knob} | {v}{mark} | {r} | {pa} |")
    w("")

    best = []
    from itertools import product
    keys = list(GRID)
    for combo in product(*(GRID[k] for k in keys)):
        p = dict(zip(keys, combo))
        r, *_ = dec_agreement(cases, p)
        pa = pref_accuracy(prefs, p)
        if r is None or pa is None:
            continue
        best.append((r, pa, p))
    # Pareto-ish: best dec subject to pref >= base - 0.03
    floor = (base_pref or 0) - 0.03
    feas = [b for b in best if b[1] >= floor]
    feas.sort(key=lambda b: (-b[0], -b[1]))
    w("### Mejores combinaciones (pref no peor que base-0.03)")
    w("")
    w("| dec | pref | parámetros |")
    w("|---|---|---|")
    for r, pa, p in feas[:8]:
        w(f"| {r} | {pa} | `{p}` |")
    w("")
    w("Nada se aplica solo: esto es evidencia para editar features._commit_fit a mano, "
      "caso por caso contra la tabla de desacuerdos.")

    out = "\n".join(lines) + "\n"
    md_path = db_path.parent / "commit_fit_triage.md"
    md_path.write_text(out, encoding="utf-8")

    print(f"Casos válidos: {n_total} | acuerdo actual: {n_agree}/{n_total} ({base_rate}) | "
          f"empates: {n_tie} | pref base: {base_pref}")
    print("Transiciones con más desacuerdo:")
    for k, n in trans_dis.most_common(5):
        print(f"  {k}: {n}/{trans[k]}")
    if feas:
        r, pa, p = feas[0]
        print(f"Mejor combo (sin degradar pref): dec {r} / pref {pa} -> {p}")
    print(f"\nInforme completo: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
