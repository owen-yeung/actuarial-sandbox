#!/usr/bin/env python3
"""Run the markdown pseudocode example (rounded counts use projection engine)."""

from term_life.projection import Assumptions, Portfolio, project_term_life
from term_life.tables import stylized_base_lapse_by_duration, stylized_base_mortality_table


def main() -> None:
    t = 20
    portfolio = Portfolio(
        initial_active=100_000,
        issue_age=40,
        sum_assured=100_000.0,
        annual_premium=500.0,
        annual_expense_per_policy=50.0,
    )
    base_mort = stylized_base_mortality_table()
    lapse = stylized_base_lapse_by_duration(t)

    assumptions = Assumptions(
        term_years=t,
        mortality_table=base_mort,
        lapse_curve=lapse,
        k_mort=0.95,
        k_lapse=1.0,
        discount_rate=0.03,
    )

    proj = project_term_life(portfolio, assumptions)
    print("Year | Active_start | Deaths | Lapses | Premium CF | Death CF")
    for year in range(proj.term_years):
        print(
            f"{year + 1:4d} | {proj.active_start[year]:12d} | "
            f"{proj.deaths[year]:6d} | {proj.lapses[year]:6d} | "
            f"{proj.premium_cf[year]:11.0f} | {proj.death_benefit_cf[year]:10.0f}"
        )
    print()
    print(f"PV premiums:        {proj.pv_premiums:,.2f}")
    print(f"PV death benefits:  {proj.pv_death_benefits:,.2f}")
    print(f"PV expenses:        {proj.pv_expenses:,.2f}")
    print(f"PV profit:          {proj.pv_profit:,.2f}")

    # Sensitivity: bump mortality scalar
    alt = project_term_life(
        portfolio,
        Assumptions(
            term_years=t,
            mortality_table=base_mort,
            lapse_curve=lapse,
            k_mort=1.05,
            k_lapse=1.0,
            discount_rate=0.03,
        ),
    )
    print()
    print(f"PV profit @ k_mort=0.95: {proj.pv_profit:,.2f}")
    print(f"PV profit @ k_mort=1.05: {alt.pv_profit:,.2f}")


if __name__ == "__main__":
    main()
