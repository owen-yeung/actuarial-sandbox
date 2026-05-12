from __future__ import annotations

from dataclasses import dataclass

from term_life.tables import LapseCurve, MortalityTable


@dataclass(frozen=True)
class Portfolio:
    """Starting cohort for a level-premium term block."""

    initial_active: int
    issue_age: int
    sum_assured: float
    annual_premium: float
    annual_expense_per_policy: float = 0.0


@dataclass(frozen=True)
class Assumptions:
    """Mortality / lapse / economic knobs from the spec."""

    term_years: int
    mortality_table: MortalityTable
    lapse_curve: LapseCurve
    k_mort: float = 1.0
    k_lapse: float = 1.0
    discount_rate: float = 0.03

    def q_effective(self, age: int) -> float:
        q = self.k_mort * self.mortality_table.q(age)
        return min(1.0, q)

    def lapse_effective(self, policy_year: int) -> float:
        ell = self.k_lapse * self.lapse_curve.rate(policy_year)
        return min(1.0, ell)


@dataclass(frozen=True)
class TermLifeProjection:
    """Year-by-year projection and discounted aggregates (mid-year discounting)."""

    active_start: tuple[int, ...]
    deaths: tuple[int, ...]
    lapses: tuple[int, ...]
    premium_cf: tuple[float, ...]
    death_benefit_cf: tuple[float, ...]
    expense_cf: tuple[float, ...]
    pv_premiums: float
    pv_death_benefits: float
    pv_expenses: float
    pv_profit: float

    @property
    def term_years(self) -> int:
        return len(self.deaths)


def _discounted_sum(cashflows: list[float], r: float) -> float:
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += cf / ((1.0 + r) ** (t + 0.5))
    return total


def project_term_life(portfolio: Portfolio, assumptions: Assumptions) -> TermLifeProjection:
    """
    Multi-decrement projection (active → death, lapse), annual steps.

    For each policy year t = 0 .. T-1 (0-based loop index):
    - A_t = active at start of year t
    - D_t = A_t * q_{x+t}
    - L_t = (A_t - D_t) * ℓ_{t+1,seg}
    - A_{t+1} = A_t - D_t - L_t
    Premiums and expenses use A_t; death benefits use D_t * SA (outflows negative in PV profit).
    """
    t_end = assumptions.term_years
    if t_end <= 0:
        raise ValueError("term_years must be positive")

    a = [0] * (t_end + 1)
    d = [0] * (t_end + 1)
    lap = [0] * (t_end + 1)
    prem_cf: list[float] = [0.0] * (t_end + 1)
    death_cf: list[float] = [0.0] * (t_end + 1)
    exp_cf: list[float] = [0.0] * (t_end + 1)

    a[0] = portfolio.initial_active

    for t in range(t_end):
        age = portfolio.issue_age + t
        q = assumptions.q_effective(age)
        policy_year = t + 1
        l = assumptions.lapse_effective(policy_year)

        d[t] = int(round(a[t] * q))
        # preserve mass balance when rounding tiny cohorts
        survivors_after_death = max(0, a[t] - d[t])
        lap[t] = int(round(survivors_after_death * l))
        if lap[t] > survivors_after_death:
            lap[t] = survivors_after_death
        a[t + 1] = max(0, a[t] - d[t] - lap[t])

        prem_cf[t] = a[t] * portfolio.annual_premium
        death_cf[t] = -d[t] * portfolio.sum_assured
        exp_cf[t] = -a[t] * portfolio.annual_expense_per_policy

    r = assumptions.discount_rate
    pv_prem = _discounted_sum(prem_cf[:t_end], r)
    pv_ben = _discounted_sum(death_cf[:t_end], r)
    pv_exp = _discounted_sum(exp_cf[:t_end], r)
    pv_profit = pv_prem + pv_ben + pv_exp

    return TermLifeProjection(
        active_start=tuple(a[:-1]) if t_end else (),
        deaths=tuple(d[:t_end]),
        lapses=tuple(lap[:t_end]),
        premium_cf=tuple(prem_cf[:t_end]),
        death_benefit_cf=tuple(death_cf[:t_end]),
        expense_cf=tuple(exp_cf[:t_end]),
        pv_premiums=pv_prem,
        pv_death_benefits=pv_ben,
        pv_expenses=pv_exp,
        pv_profit=pv_profit,
    )
