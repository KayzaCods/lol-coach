"""Cola de feedback de trades: los trades PERDIDOS sin marcar, con sus hechos.

El detector de trades (trade_v1) es el más fiel al criterio del jugador y el más
huérfano de marcas (backlog #5: prioritario para el learner). Marcar decenas a
ciegas en el dashboard es lento; esta cola saca, por partida, cada trade perdido
sin marcar con lo necesario para juzgarlo (minuto, daño recibido tú vs duo enemigo,
estado al cierre, consecuencia) para decidir rápido y marcarlo en Decisiones
(filtro Trade). Re-corre el script para ver lo que queda.

Trades GANADOS quedan ocultos a propósito (solo Calibrar), así que no aparecen aquí.
Solo se miran cuentas propias (master/emerald); la referencia (challenger) se excluye.

Solo lee la DB. Escribe data/trade_feedback_queue.md.

    .venv\\Scripts\\python.exe scripts\\trade_feedback_queue.py
    .venv\\Scripts\\python.exe scripts\\trade_feedback_queue.py --db ruta\\a\\copia.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lol_coach.config import load_config

WON = '%"positive_trade": true%'        # trades ganados: ocultos de Decisiones
OWN = ("master", "emerald")             # cuentas propias; excluye la referencia


def _date(ms) -> str:
    if not ms:
        return "?"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def build(conn: sqlite3.Connection) -> tuple[str, int, int]:
    """(markdown, n_trades, n_matches) de los trades perdidos sin marcar."""
    rows = conn.execute(
        f"""
        SELECT d.id, d.match_id, d.game_time_ms, d.context_json, m.our_cohort,
               m.game_start_ms,
               (SELECT p.champion_name FROM participants p
                 WHERE p.match_id = d.match_id AND p.puuid = m.our_puuid) AS champ
        FROM decisions d JOIN matches m ON m.match_id = d.match_id
        WHERE d.detector_id = 'trade_v1'
          AND m.our_cohort IN ({",".join("?" * len(OWN))})
          AND d.context_json NOT LIKE ?
          AND d.user_feedback IS NULL AND d.user_best_option IS NULL
        ORDER BY m.game_start_ms DESC, d.game_time_ms ASC
        """,
        (*OWN, WON),
    ).fetchall()

    by_match: dict[str, list] = {}
    for r in rows:
        by_match.setdefault(r["match_id"], []).append(r)

    out = ["# Cola de feedback de trades (perdidos sin marcar)", ""]
    out.append(
        f"> {len(rows)} trades perdidos sin marcar en {len(by_match)} partidas (cuentas "
        f"propias). Márcalos en el dashboard: pestaña **Decisiones**, filtro **Trade**. "
        f"Re-corre el script para ver lo que queda."
    )
    out.append("")
    out.append(
        "El criterio no es no tradear: es cambiar daño solo con ventana (maná/cooldowns/"
        "posición); si no, respetar y farmear. `daño` = lo que recibió tu duo vs el duo "
        "enemigo en la ventana (proxy de quién ganó el intercambio)."
    )
    out.append("")

    for mid, items in by_match.items():
        r0 = items[0]
        out.append(
            f"## {r0['champ'] or '?'} — {_date(r0['game_start_ms'])} "
            f"({r0['our_cohort']}) · {len(items)} sin marcar · `{mid}`"
        )
        out.append("")
        out.append("| id | min | daño (tú vs enemigo) | estado al cierre | consecuencia |")
        out.append("|---|---|---|---|---|")
        for r in items:
            ctx = json.loads(r["context_json"])
            dus, den = ctx.get("duo_damage_taken"), ctx.get("enemy_duo_damage_taken")
            hp, mana = ctx.get("our_hp_pct_end"), ctx.get("our_mana_pct_end")
            state = " · ".join(
                x for x in (
                    f"HP {int(hp * 100)}%" if hp is not None else None,
                    f"maná {int(mana * 100)}%" if mana is not None else None,
                ) if x
            ) or "—"
            cons = ctx.get("consequence") or "—"
            span = ctx.get("minute_span") or ctx.get("time_mmss") or "?"
            out.append(f"| {r['id']} | {span} | {dus} vs {den} | {state} | {cons} |")
        out.append("")

    return "\n".join(out), len(rows), len(by_match)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cola de feedback de trades sin marcar")
    ap.add_argument("--db", help="ruta a la DB (por defecto, la de config.toml)")
    args = ap.parse_args()

    db = args.db or load_config()["paths"]["sqlite_db"]
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    text, n, nmatches = build(conn)
    conn.close()

    out_path = Path(db).parent / "trade_feedback_queue.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"{n} trades perdidos sin marcar en {nmatches} partidas.")
    print(f"Escrito: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
