# lol-coach — Architecture

A behavioral coaching engine for League of Legends. It ingests my own ranked
games, detects the *decision points* inside each one, evaluates the choice that
was actually made against the alternatives, and produces a written argument for
what was optimal and why. The output is not a stat sheet or a list of flags — it
is an **argument** per moment: structured context + options with expected value
(EV) + prose. Criteria are refined by human feedback that trains a preference
model over the evaluation weights.

The premise: climbing out of high elo is a *behavioral* problem (consistency),
not a mechanical one. The system's job is to surface the pattern you repeat while
knowing it is wrong. The dominant measured pattern for this account is
**over-extension while ahead**.

---

## Stack (deliberately minimal)

Python standard library + `requests` (Riot's API rejects urllib's default
User-Agent with a 403). The frontend is React over UMD served as a single
`dashboard.html` — **no build step**. Storage is SQLite in WAL mode. No
FastAPI / Vite / ORM, on purpose: the constraint keeps the surface area small
and the whole system inspectable. `pytest` is a dev-only dependency; the runtime
stays stdlib + `requests`.

The dashboard UI and generated coaching prose are in Spanish (the target user is
the author); the code is in English.

---

## Data model: accounts and cohorts

Matches are tagged with a `cohort` (the elo bracket they were played in) so
baselines from very different opponent skill levels are not mixed. A small set
of **reference cohorts** (e.g. a Challenger one-trick) is ingested separately and
kept isolated: stats, patterns, and the learner use only the player's own
accounts unless a reference cohort is explicitly selected. Reference players are
ingested through a dedicated manual script and are never added to `config.toml`,
so the automatic ingest never touches them.

| Account | Cohort | Role |
|---|---|---|
| main account | `master` | primary |
| secondary | `emerald` | secondary |
| reference player | `challenger` | comparison only, never trains the model |

---

## Pipeline

```
CAPTURE
  recorder.py (Live Client API :2999, 1 Hz)      external input recorder
    data/raw/<session>/snapshots.jsonl             video .mp4 + input event log
    (exact HP / mana / gold, own games only)       (keystrokes / clicks per game)
        |                                              |
INGEST  +--------------+--------------------------------+
  auto_ingest.py  (scheduled task, every 30 min, headless)
    -> rotated daily DB backup (7 copies, WAL-safe)
    -> sync_ascent.py:
        0) link recorder sessions  <-> Riot matches
        1) link clips, pull missing matches from Riot
        2) for each PENDING match (analyzed_at_utc IS NULL):
             ingest input events -> input_events
             analyze_match -> detectors -> persist_decisions -> reward -> marker
        3) assign clips to decisions
  ingest_reference.py (manual) -> reference player, isolated cohort
        |
  INGEST GUARD: ranked queues only (420 solo/duo, 440 flex), duration >= 300s.
  Normals / remakes -> skipped_matches (tombstone: never re-fetched).
        |
ANALYSIS  lol_coach/analysis.py
  DETECTOR_FUNCS = [death, trade, tempo, objective_readiness]
        |
DASHBOARD  scripts/dashboard.py (stdlib http.server, :8765) + dashboard.html
  /api/decisions  -> HEF recomputed per request with the active weights
  /api/patterns   -> behavioral-progress series (Wilson CIs, moving windows)
  /api/session    -> "today's session": unmarked moments in an anti-bias mix
  /api/ingest_status -> red banner if the Riot API key has gone dead
        |
FEEDBACK  POST /api/feedback   (per-decision: agree / disagree / note /
          best_option / mark_type)   ·   POST /api/preference  (pairwise)
        |
LEARNER   preference.learn_weights (Bradley-Terry MAP, cross-validated)
          -> hef_weights.json (guardrailed) -> feeds back into /api/decisions
```

Two independent capture layers feed the same DB: a 1 Hz **Live Client poller**
for exact per-second state (HP/mana/gold/position) on the player's own games, and
an external **video + input-event recorder** used for input-based signals and
clip attachment. Riot's Match-V5 timeline backfills everything else and is the
only source for reference players.

---

## Decision detectors

Signature: `analyze_x(conn, match_id) -> list[Decision]`. Each `Decision` carries
`context` (standardized `state_features` + the action taken), `options` (each with
a label, a consequence, and an EV; exactly one is marked as "what you did"), an
`argument` (prose), and optionally a video clip.

Internally every detector separates **I/O from logic**: `_gather_facts(conn, …)
-> *Facts` does all the I/O and returns a flat dataclass (returning `None` when
there is no decision), and `_evaluate(facts) -> Decision` is pure. The
classification / options / argument logic is therefore unit-tested without a
database.

- **death_v1** — every one of the player's own deaths, classified by the real
  action: `calculated_sacrifice` (team took an objective within range — traded
  life for objective), `engage_worth` (own kill/assist in the window),
  `support_lost` (an ally fell first — you walked into a fight already lost),
  `trade_lost`, `disengage_failed` (caught on the way out), `engage_blind`
  (over-extension). A key subtlety: enemy **pre-fight visibility** is measured
  10s *before* the death, not at the moment of death — the kill event itself
  "reveals" the killer, which would otherwise make every gank look "seen".
- **trade_v1 (v2)** — bot-lane damage trades during laning (min 2–14, support/bot
  roles, bottom half of the map). Enemy HP is not observable, so damage taken is
  the proxy; a trade is "bad" if your duo took disproportionately more. Minutes
  where the duo dies are skipped — death_v1 owns that play, so a death is never
  double-counted as a trade. Won trades are persisted but hidden from the lists,
  surfaced only in the pairwise calibration view.
- **tempo_v1** — fights your team took without you. Does not fire if you were
  dead (estimated respawn + grace) or if the fight was physically unreachable.
- **objective_readiness_v1** — each epic objective taken (dragon/herald/baron).
  Scores `setup` = team vision + numbers near the pit + position + resources.
  Handles must-defend states (conceding is correct), uncontested/ dominating
  states (presence optional), natural despawn (not "stolen"), and objectives
  traded for one another. Options avoid outcome bias and never present twin
  choices: if you did the right thing, one option reads "(what you did) — this
  was correct" against a real low-EV contrast.

Retired detectors (`hesitation`, `awareness`, `vision_prep`) remain in the tree
for reference but are not imported: hesitation from current input data is noise,
"look at the map" proved too subjective, and vision prep was absorbed into
objective readiness.

---

## HEF — hierarchical evaluation function

`F = Σ ωᵢ·termᵢ − Π` over the terms present.

| Term | Measures | Default weight |
|---|---|---|
| `power` | local advantage (level / gold / HP / numbers) | 0.35 |
| `info` | information safety (enemies seen, jungle, local threat) | 0.30 |
| `action_fit` | how well the action matches the setup | 0.25 |
| `wave` | wave state — **context-sensitive** | 0.10 |

- **`_commit_fit`** blends `setup = 0.3·info + 0.7·power`. The info term was found
  to *overestimate* safety — recalibrating the blend (0.6→0.3 on info) raised
  agreement with human marks from 0.46 to 0.58. Aggressive actions map to
  `setup`, defensive to `1 − setup`, vision to `0.6 + 0.4·(1 − setup)`; a
  `commit_sin_setup` penalty fires on aggression with low info.
- **`wave`** inverts by context: near your own tower during a *death* means safety
  (high); a wave pushed into the enemy during an *objective* means pressure
  (high). The exact same board reads oppositely depending on the moment.
- Active weights live in `hef_weights.json` (learned, reversible to default). The
  HEF is recomputed per request: changing weights never touches the options, EVs,
  or prose (the detectors generate those) — only the score.

---

## Feedback and learning

Two feedback sources, kept separate and balanced to equal mass:

1. **Pairwise calibration** (`preferences`): "which of A or B was played better?"
   over comparable decisions. Carries the weight of the *state* terms.
2. **Per-decision** (`decisions.user_*`): a verdict (`agree`/`disagree`/
   `equivalent`) + the option you consider correct + a note. Isolates
   `action_fit` by comparing two actions on the *same* state.

**Feedback fingerprints (robustness).** `user_best_option` is an *index*, so a
mark is only valid while the option set keeps its meaning. An
`action_fingerprint = v1|detector|action_ids|t{taken_index}` (a semantic
silhouette, ignoring prose with volatile numbers) is stored with each mark, and
the learner validates by *content* rather than by timestamp. A timestamp gate
once invalidated 192/192 marks on a full re-analysis; the fingerprint gate
revives the ones whose options did not actually change.

**Mark type.** `user_mark_type` records *why* a mark was made — `decision`
(a critique of the criterion, the only type that trains the learner),
`execution` (a mechanics complaint, not a decision), `mixed`, `missing_context`,
and `wrong_moment` (the card points at the wrong instant — the mistake was
earlier). The train/skip gate lives in a single pure function; untyped legacy
marks fall back to a noise-note regex so historical behavior is unchanged.

**Durability across re-analysis.** `persist_decisions` preserves feedback by the
stable key `(detector_id, game_time_ms)` across a delete+reinsert; feedback whose
key does not reappear (because a detector changed its triggers) is *reported* as
dropped rather than silently lost.

**Learner** (`preference.learn_weights`, Bradley-Terry MAP): an L2 prior toward
the default weights prevents collapse (e.g. info → 0); the two sources are
balanced to equal mass; accuracy is reported in-sample, per-source, and
**cross-validated**, and the `apply_weights` guardrail trusts only the
out-of-sample number.

---

## Behavioral progress

The point of the whole system is to answer "am I actually improving?" without
outcome bias. The progress view tracks the dominant pattern
(`overextension_with_lead`: engage-blind / disengage-failed deaths while ahead)
as a proportion over a moving window, with a **Wilson confidence interval**, a
death-density curve, a provisional target, and a verdict computed over blocks of
games. The honest current read is "interval-overlapping — no distinguishable
improvement yet", which is exactly the kind of claim the CIs are there to keep
the system from overstating.

---

## Testing

Characterization tests (they pin current behavior as a safety net for criterion
refactors), run with `pytest`:

- `test_features.py` — pure evaluation: HEF + penalties, `_commit_fit`,
  `_wave_term` (the exact death↔objective inversion), `power_index`,
  `action_fingerprint`.
- `test_preference.py` — the Bradley-Terry learner: source balancing, no-collapse
  under the prior, deterministic cross-validation.
- `test_death.py` / `test_tempo.py` / `test_objective_readiness.py` /
  `test_trade.py` — end-to-end integration of all four detectors over a synthetic
  ten-player match, asserting the key guard of each.
- `test_*_logic.py` — the pure `_evaluate` branches per detector, no database.
- `test_feedback.py`, `test_session.py`, `test_progress*.py` — the feedback gate,
  the session mix, and the progress math.

---

## Safety invariants

1. `config.toml` (the Riot API key) is never committed. Raw gameplay JSON and the
   SQLite DB are never committed. `git` is the safety net.
2. A daily DB backup runs even when the API key is dead — user feedback lives only
   in the DB.
3. Re-analysis goes only through `analyze.py` / `sync_ascent` (`persist_decisions`).
   Never `DELETE FROM decisions` by hand.
4. Fixed order: detectors → reward → clips. Skipping clips orphans them.
5. `analyzed_at_utc` is the "done" marker.
6. Ranked only; normals / remakes are tombstoned.
7. `current_gold` = spendable gold (readiness/power); `total_gold` = stat only.

---

## Entry points

| Command | Purpose |
|---|---|
| `scripts/sync_ascent.py [--all]` | full pipeline (capture + analysis + clips) |
| `scripts/analyze.py [match \| --all]` | re-analyze, preserving feedback |
| `scripts/dashboard.py [--port 8765]` | dashboard server |
| `scripts/ingest_reference.py "Name#TAG" --champion X` | ingest a reference player |
| `scripts/download_game_data.py` | fetch static game data (patch-versioned) |
| `.venv\Scripts\python.exe -m pytest` | run the test suite |
