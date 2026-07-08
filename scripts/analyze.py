"""Safe (re)analysis — the ONE command to (re)run detectors on any match.

Per match it runs the FULL chain in the right order:
  1. run_all_detectors -> persist_decisions   (preserves the user's feedback by
     stable key; never a raw DELETE)
  2. compute_reward.run_for_match             (reward lives inside context_json,
     so it must be recomputed after every re-analysis)
  3. complete_clips (once, after all matches) (clip assignments are wiped by
     re-analysis for every non-death detector)
  4. matches.analyzed_at_utc = now            (the "done" marker sync_ascent uses)

History note: the previous analyze.py did `DELETE FROM decisions` + reinsert,
silently destroying feedback/reward/clips — both production incidents traced to
that pattern. analyze_recent.py was retired at the same time.

Usage:
    .venv\\Scripts\\python.exe scripts\\analyze.py                 # pending only (analyzed_at IS NULL)
    .venv\\Scripts\\python.exe scripts\\analyze.py LA1_12345       # one match
    .venv\\Scripts\\python.exe scripts\\analyze.py --all           # re-analyze everything
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sibling scripts

from lol_coach.analysis import persist_decisions, run_all_detectors
from lol_coach.config import load_config
from lol_coach.db import connect

import complete_clips
import compute_reward


def analyze_match(conn, match_id: str) -> int:
    """Detectors + reward for one match, feedback-preserving; sets the marker.
    Clips are NOT assigned here — run run_clips() after the batch."""
    decisions = run_all_detectors(conn, match_id)
    n, fb_stats = persist_decisions(conn, match_id, decisions)
    if fb_stats["dropped"]:
        keys = ", ".join(f"{det}@{ms}" for det, ms in fb_stats["dropped_keys"])
        print(
            f"   AVISO {match_id}: {fb_stats['dropped']} feedback(s) sin destino tras "
            f"re-análisis (el detector cambió triggers/timestamps): {keys}"
        )
    try:
        compute_reward.run_for_match(conn, match_id)
    except Exception as e:  # reward is best-effort (reference matches lack some data)
        print(f"   (reward fallo en {match_id}: {e!r})")
    conn.execute(
        "UPDATE matches SET analyzed_at_utc=? WHERE match_id=?",
        (datetime.now(timezone.utc).isoformat(), match_id),
    )
    conn.commit()
    return n


def run_clips(conn) -> None:
    nfull = complete_clips.normalize_full_replays(conn)
    res = complete_clips.complete_clips_for_decisions(conn)
    conn.commit()
    print(f"clips: full normalizados={nfull} | asignados {res}")


def main() -> int:
    cfg = load_config()
    conn = connect(cfg["paths"]["sqlite_db"])

    args = [a for a in sys.argv[1:] if a]
    if "--all" in args:
        mids = [r["match_id"] for r in conn.execute(
            "SELECT match_id FROM matches ORDER BY game_start_ms")]
        label = "todas"
    elif args:
        mids = [args[0]]
        label = "1 explícita"
    else:
        mids = [r["match_id"] for r in conn.execute(
            "SELECT match_id FROM matches WHERE analyzed_at_utc IS NULL ORDER BY game_start_ms")]
        label = "pendientes"

    if not mids:
        print("Nada pendiente de analizar (usa --all para reprocesar todo).")
        return 0

    print(f"Analizando {len(mids)} partida(s) ({label})...")
    total = 0
    for mid in mids:
        n = analyze_match(conn, mid)
        total += n
        print(f"   {mid}: {n} decisiones")
    run_clips(conn)
    print(f"LISTO. {len(mids)} partida(s), {total} decisiones. Feedback preservado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
