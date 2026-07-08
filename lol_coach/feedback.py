"""Vocabulario de tipos de marca y el gate de entrenamiento del learner (#12).

Puro stdlib. Cada marca por-decision lleva un TIPO que dice POR QUÉ se hizo; solo
las marcas 'decision' (crítica al criterio de decisión) entrenan al learner HEF.
Los otros tipos capturan quejas de ejecución o apuntan a consumidores futuros
(extractor de contexto, #17/#13). Las marcas hechas antes de que existiera la
columna no tienen tipo y caen al regex de ruido de la nota.
"""
from __future__ import annotations

MARK_TYPES = ("decision", "execution", "mixed", "missing_context", "wrong_moment")
TRAINING_TYPES = frozenset({"decision"})


def trains_learner(mark_type, note, noise_re) -> bool:
    """True sii esta marca debe mover pesos HEF. Tipada: solo 'decision'.
    Sin tipo (marca vieja, mark_type None/''): fallback al regex de ruido."""
    if mark_type:
        return mark_type in TRAINING_TYPES
    return not noise_re.search(note or "")
