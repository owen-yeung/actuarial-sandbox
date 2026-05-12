from __future__ import annotations

import re
from typing import Any

import pandas as pd

from term_life.tables import LapseCurve, MortalityTable, stylized_base_lapse_by_duration, stylized_base_mortality_table


def _norm_col(name: str) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower())


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _norm_col(c) for c in df.columns})


def mortality_table_from_csv(
    df: pd.DataFrame,
    *,
    merge_with_stylized: bool = True,
) -> MortalityTable:
    """
    Expect columns: age (or x), q (or q_x, rate, mortality).

    If merge_with_stylized, start from stylized table and override ages present in CSV.
    """
    d = _rename_columns(df.copy())
    age_aliases = ("age", "x", "attained_age", "issue_age")
    q_aliases = ("q", "q_x", "rate", "mortality", "qx", "q_x_table")

    age_col = next((c for c in d.columns if c in age_aliases), None)
    q_col = next((c for c in d.columns if c in q_aliases), None)
    if age_col is None or q_col is None:
        raise ValueError(
            "Mortality CSV needs age-like column (age, x, attained_age) and "
            "q-like column (q, q_x, rate, mortality)."
        )

    rates: dict[int, float] = {}
    if merge_with_stylized:
        rates.update(dict(stylized_base_mortality_table().rates))
    for _, row in d.iterrows():
        try:
            age = int(float(row[age_col]))
            q = float(row[q_col])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid row in mortality CSV: {row.to_dict()}") from e
        if not (0 <= q <= 1):
            raise ValueError(f"Mortality rate must be in [0,1] at age {age}, got {q}")
        rates[age] = q
    return MortalityTable(rates=rates)


def lapse_curve_from_csv(
    df: pd.DataFrame,
    term_years: int,
    *,
    merge_with_stylized: bool = True,
) -> LapseCurve:
    """
    Expect columns: policy_year (or year, t, duration) and lapse (or rate, l, ell).

    Policy years are 1-indexed. If merge_with_stylized, fill from stylized then override.
    """
    d = _rename_columns(df.copy())
    year_aliases = ("policy_year", "year", "t", "duration", "policy_yr")
    lapse_aliases = ("lapse", "rate", "l", "ell", "lapse_rate", "withdrawal")

    y_col = next((c for c in d.columns if c in year_aliases), None)
    l_col = next((c for c in d.columns if c in lapse_aliases), None)
    if y_col is None or l_col is None:
        raise ValueError(
            "Lapse CSV needs policy-year column (policy_year, year, t, duration) and "
            "lapse rate column (lapse, rate, l, ell)."
        )

    by: dict[int, float] = {}
    if merge_with_stylized:
        by.update(dict(stylized_base_lapse_by_duration(term_years).by_policy_year))
    for _, row in d.iterrows():
        try:
            t = int(float(row[y_col]))
            rate = float(row[l_col])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid row in lapse CSV: {row.to_dict()}") from e
        if t < 1:
            raise ValueError(f"policy_year must be >= 1, got {t}")
        if not (0 <= rate <= 1):
            raise ValueError(f"Lapse rate must be in [0,1] at year {t}, got {rate}")
        by[t] = rate
    if not by:
        raise ValueError("Lapse curve is empty after parsing.")
    return LapseCurve(by_policy_year=by)


def parse_mortality_experience(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows with age, exposure, observed_deaths (flexible column names)."""
    d = _rename_columns(df.copy())
    age_c = next((c for c in d.columns if c in ("age", "x", "attained_age")), None)
    exp_c = next((c for c in d.columns if c in ("exposure", "exposures", "ee", "e")), None)
    death_c = next(
        (c for c in d.columns if c in ("observed_deaths", "deaths", "d", "death", "claims")),
        None,
    )
    if not all([age_c, exp_c, death_c]):
        raise ValueError(
            "Mortality experience CSV needs age, exposure, and observed_deaths "
            "(or deaths, d, claims)."
        )
    out: list[dict[str, Any]] = []
    for _, row in d.iterrows():
        out.append(
            {
                "age": int(float(row[age_c])),
                "exposure": float(row[exp_c]),
                "observed_deaths": float(row[death_c]),
            }
        )
    return out


def parse_lapse_experience(df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Returns policy_year -> (exposure, lapses_observed)."""
    d = _rename_columns(df.copy())
    y_c = next((c for c in d.columns if c in ("policy_year", "year", "t", "duration")), None)
    exp_c = next((c for c in d.columns if c in ("exposure", "exposures", "ee", "e")), None)
    lap_c = next(
        (c for c in d.columns if c in ("lapses", "lapse", "lapses_observed", "withdrawals")),
        None,
    )
    if not all([y_c, exp_c, lap_c]):
        raise ValueError(
            "Lapse experience CSV needs policy_year, exposure, and lapses "
            "(or lapses_observed, withdrawals)."
        )
    by_year: dict[int, tuple[float, float]] = {}
    for _, row in d.iterrows():
        t = int(float(row[y_c]))
        by_year[t] = (float(row[exp_c]), float(row[lap_c]))
    return by_year
