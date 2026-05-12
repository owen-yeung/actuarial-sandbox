from __future__ import annotations

from dataclasses import dataclass

from term_life.tables import MortalityTable


def mortality_ae_ratio(
    observed_deaths: float,
    exposure: float,
    age: int,
    table: MortalityTable,
) -> float | None:
    """
    A/E = D^obs / D^exp with D^exp = exposure * q_table(age).
    Returns None if expected deaths are zero.
    """
    if exposure <= 0:
        return None
    q = table.q(age)
    expected = exposure * q
    if expected <= 0:
        return None
    return observed_deaths / expected


def suggest_k_mort_from_ae(ae: float, partial_credibility: float = 1.0) -> float:
    """
    Map A/E to a multiplicative mortality scalar, optionally damped toward 1.0.

    k_new = 1 + w * (A/E - 1) with w in [0, 1] as a toy credibility weight.
    """
    w = max(0.0, min(1.0, partial_credibility))
    return max(0.0, 1.0 + w * (ae - 1.0))


@dataclass(frozen=True)
class LapseExperienceRow:
    policy_year: int
    exposure: float
    lapses_observed: float
    crude_rate: float | None


def lapse_experience_summary(
    by_year: dict[int, tuple[float, float]],
) -> tuple[LapseExperienceRow, ...]:
    """
    by_year maps policy_year -> (exposure_at_risk, lapses_observed).
    Crude rate = lapses / exposure when exposure > 0.
    """
    rows: list[LapseExperienceRow] = []
    for t in sorted(by_year):
        exp, laps = by_year[t]
        crude = (laps / exp) if exp > 0 else None
        rows.append(
            LapseExperienceRow(
                policy_year=t,
                exposure=exp,
                lapses_observed=laps,
                crude_rate=crude,
            )
        )
    return tuple(rows)
