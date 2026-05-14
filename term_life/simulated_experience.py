"""
Simulated policy-level experience (Step 1 of the assumption-update workflow).

Generates annual segments with policy attributes, fractional exposure support,
and mutually exclusive death vs lapse outcomes drawn using baseline table rates
scaled by configurable true factors.
"""

from __future__ import annotations

import random

import pandas as pd

from term_life.policy_experience import attained_age_start_of_policy_year
from term_life.tables import LapseCurve, MortalityTable, stylized_base_lapse_by_duration, stylized_base_mortality_table


def generate_policy_level_term_experience(
    *,
    rng: random.Random | None = None,
    n_policies: int = 4_000,
    issue_age: int = 40,
    term_years: int = 20,
    study_policy_years: int = 5,
    true_k_mort: float = 1.07,
    true_k_lapse: float = 0.93,
    mortality_table: MortalityTable | None = None,
    lapse_curve: LapseCurve | None = None,
    sum_assured_choices: tuple[float, ...] = (100_000.0, 250_000.0, 500_000.0),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a cohort of level term policies observed for the first `study_policy_years`
    policy years. Each row is one policy-segment (annual); exposure_years is 1.0 if
    the policy is in-force at the start of the segment, else the segment is omitted.

    Death is drawn first with probability min(1, true_k_mort * q_table), then lapse
    on survivors with probability min(1, true_k_lapse * lapse_base(t)).
    """
    rng = rng or random.Random(seed)
    mort = mortality_table or stylized_base_mortality_table()
    lapse = lapse_curve or stylized_base_lapse_by_duration(term_years)

    rows: list[dict[str, object]] = []
    genders: tuple[str, ...] = ("M", "F")
    smokers: tuple[str, ...] = ("N", "Y")

    for pid in range(n_policies):
        gender = rng.choice(genders)
        smoker = rng.choice(smokers)
        sa = float(rng.choice(sum_assured_choices))
        alive = True
        for t in range(1, study_policy_years + 1):
            if not alive:
                break
            age = attained_age_start_of_policy_year(issue_age, t)
            q = min(1.0, true_k_mort * mort.q(age))
            l_base = lapse.rate(t)
            l = min(1.0, true_k_lapse * l_base)
            u = rng.random()
            died = 0
            lapsed = 0
            if u < q:
                died = 1
                alive = False
            elif u < q + (1.0 - q) * l:
                lapsed = 1
                alive = False
            rows.append(
                {
                    "policy_id": f"POL-{pid:05d}",
                    "issue_age": issue_age,
                    "gender": gender,
                    "smoker": smoker,
                    "sum_assured": sa,
                    "policy_year": t,
                    "exposure_years": 1.0,
                    "died": died,
                    "lapsed": lapsed,
                }
            )
    return pd.DataFrame(rows)


def demo_policy_csv_text() -> str:
    """Small fixed sample for downloads (subset of generator output)."""
    df = generate_policy_level_term_experience(
        n_policies=800,
        study_policy_years=5,
        seed=123,
    )
    return df.to_csv(index=False)
