"""Dataclasses shared by all decision evaluators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Option:
    """One alternative course of action at a decision moment, with scored EV."""
    label: str
    predicted_consequence: str
    ev_score: float  # 0..1, heuristic in Phase 1


@dataclass
class Decision:
    """Output of a decision evaluator. One per detected moment.

    Stored to the `decisions` table and used as the unit shown in reports.
    """
    detector_id: str
    match_id: str
    game_time_ms: int
    moment: str               # short human-readable description
    outcome: str              # what actually happened
    context: dict[str, Any]   # structured facts feeding the argument
    options: list[Option]
    argument: str             # the prose argument
    clip_path: Optional[str] = None
    clip_offset_s: Optional[float] = None

    @property
    def game_time_s(self) -> float:
        return self.game_time_ms / 1000.0

    @property
    def time_mmss(self) -> str:
        s = int(self.game_time_ms // 1000)
        return f"{s // 60}:{s % 60:02d}"
