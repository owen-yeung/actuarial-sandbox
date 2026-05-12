"""Simplified level-premium term-life projection (mortality + lapse)."""

from term_life.experience import (
    lapse_experience_summary,
    mortality_ae_ratio,
    suggest_k_mort_from_ae,
)
from term_life.projection import (
    Assumptions,
    Portfolio,
    TermLifeProjection,
    project_term_life,
)
from term_life.tables import (
    LapseCurve,
    MortalityTable,
    stylized_base_lapse_by_duration,
    stylized_base_mortality_table,
)

__all__ = [
    "Assumptions",
    "LapseCurve",
    "MortalityTable",
    "Portfolio",
    "TermLifeProjection",
    "lapse_experience_summary",
    "mortality_ae_ratio",
    "project_term_life",
    "stylized_base_lapse_by_duration",
    "stylized_base_mortality_table",
    "suggest_k_mort_from_ae",
]
