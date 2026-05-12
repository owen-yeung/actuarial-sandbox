from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MortalityTable:
    """Age-indexed base annual mortality rates q_x^table (lookup only; no interpolation)."""

    rates: Mapping[int, float]

    def q(self, age: int) -> float:
        if age not in self.rates:
            raise KeyError(f"No base mortality rate for age {age}")
        return float(self.rates[age])


@dataclass(frozen=True)
class LapseCurve:
    """Policy-year (1-indexed) lapse rates ℓ_t^base; missing years use the last defined duration."""

    by_policy_year: Mapping[int, float]

    def rate(self, policy_year: int) -> float:
        if policy_year in self.by_policy_year:
            return float(self.by_policy_year[policy_year])
        if not self.by_policy_year:
            raise ValueError("Empty lapse curve")
        last_t = max(self.by_policy_year)
        return float(self.by_policy_year[last_t])


def stylized_base_mortality_table() -> MortalityTable:
    """
    Illustrative age-specific rates for demos — not an official valuation table.
    Smooth, increasing q_x suitable for ages ~25–75 (term-life sandbox).
    """
    rates: dict[int, float] = {}
    for age in range(18, 101):
        # Simple increasing pattern: roughly 0.1–0.5‰ at young ages to a few % at old ages
        x = age - 18
        q = 0.00008 + 0.0000025 * (x**1.35)
        rates[age] = min(0.45, q)
    return MortalityTable(rates=rates)


def stylized_base_lapse_by_duration(term_years: int) -> LapseCurve:
    """
    High early lapse, then stabilization — illustrative base curve ℓ_t^base.
    Policy years are 1 .. term_years.
    """
    base_early = 0.06
    base_mid = 0.03
    base_tail = 0.02
    by: dict[int, float] = {}
    for t in range(1, term_years + 1):
        if t <= 2:
            by[t] = base_early
        elif t <= 5:
            by[t] = base_mid
        else:
            by[t] = base_tail
    return LapseCurve(by_policy_year=by)
