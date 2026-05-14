from __future__ import annotations

import re
import pandas as pd

from term_life.tables import LapseCurve, MortalityTable


def _norm_col(name: str) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower())


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _norm_col(c) for c in df.columns})


def parse_policy_level_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 style policy segments: attributes + exposure + decrement flags.

    Required columns (flexible names):
    - policy_id (or policy, id)
    - issue_age (or issue age)
    - gender
    - smoker (or smoking, smokes)
    - sum_assured (or sa, face, benefit)
    - policy_year (or duration, t, policy duration)
    - exposure_years (or exposure, py, policy_years)
    - died (or death, mortality_event) 0/1
    - lapsed (or lapse, withdrawal) 0/1
    """
    d = _rename_columns(df.copy())
    col_map = {
        "pid": next((c for c in d.columns if c in ("policy_id", "policy", "id", "pol_id")), None),
        "issue_age": next((c for c in d.columns if c in ("issue_age", "issueage", "x0")), None),
        "gender": next((c for c in d.columns if c in ("gender", "sex")), None),
        "smoker": next((c for c in d.columns if c in ("smoker", "smoking", "smokes", "tobacco")), None),
        "sum_assured": next(
            (c for c in d.columns if c in ("sum_assured", "sa", "face", "benefit", "sum_assured_amount")),
            None,
        ),
        "policy_year": next(
            (c for c in d.columns if c in ("policy_year", "duration", "t", "policy_duration", "dur")),
            None,
        ),
        "exposure": next(
            (c for c in d.columns if c in ("exposure_years", "exposure", "py", "policy_years", "years")),
            None,
        ),
        "died": next((c for c in d.columns if c in ("died", "death", "mortality_event", "is_death")), None),
        "lapsed": next((c for c in d.columns if c in ("lapsed", "lapse", "withdrawal", "is_lapse")), None),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(
            "Policy-level CSV is missing required columns. Need: policy id, issue_age, gender, smoker, "
            "sum_assured, policy_year, exposure_years, died, lapsed. "
            f"Could not resolve: {missing}. Got columns: {list(d.columns)}"
        )

    out = pd.DataFrame(
        {
            "policy_id": d[col_map["pid"]].astype(str),
            "issue_age": d[col_map["issue_age"]].astype(float).astype(int),
            "gender": d[col_map["gender"]].astype(str),
            "smoker": d[col_map["smoker"]].astype(str),
            "sum_assured": d[col_map["sum_assured"]].astype(float),
            "policy_year": d[col_map["policy_year"]].astype(float).astype(int),
            "exposure_years": d[col_map["exposure"]].astype(float),
            "died": d[col_map["died"]].astype(int).clip(0, 1),
            "lapsed": d[col_map["lapsed"]].astype(int).clip(0, 1),
        }
    )
    if (out["died"] + out["lapsed"] > 1).any():
        raise ValueError("Each row must have at most one of died=1 or lapsed=1.")
    if (out["exposure_years"] < 0).any():
        raise ValueError("exposure_years must be non-negative.")
    return out


def attained_age_start_of_policy_year(issue_age: int, policy_year: int) -> int:
    """Attained age at start of policy year t (1-indexed)."""
    return issue_age + policy_year - 1


def rollup_mortality_by_age(policy_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exposure and deaths by attained age (start of segment)."""
    tmp = policy_df.copy()
    tmp["_attained_age"] = tmp["issue_age"].astype(int) + tmp["policy_year"].astype(int) - 1
    g = tmp.groupby("_attained_age", as_index=False).agg(
        exposure_years=("exposure_years", "sum"),
        observed_deaths=("died", "sum"),
    )
    g = g.rename(columns={"_attained_age": "attained_age"})
    return g.sort_values("attained_age").reset_index(drop=True)


def rollup_lapse_by_policy_year(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lapse exposure net of deaths in the same segment: exposure * (1 - died).
    Lapses counted on rows where lapsed=1.
    """
    el = policy_df["exposure_years"] * (1.0 - policy_df["died"].astype(float))
    tmp = policy_df.copy()
    tmp["_exposure_lapse"] = el
    g = tmp.groupby("policy_year", as_index=False).agg(
        exposure_at_lapse_risk=("_exposure_lapse", "sum"),
        observed_lapses=("lapsed", "sum"),
    )
    return g.sort_values("policy_year").reset_index(drop=True)


def expected_decrements_from_rollups(
    mort_by_age: pd.DataFrame,
    lapse_by_year: pd.DataFrame,
    mortality_table: MortalityTable,
    lapse_curve_base: LapseCurve,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 2: expected deaths = exposure * q_table(age); expected lapses = exposure_lapse * base lapse(t).

    Uses base table and base lapse curve only (no k_mort / k_lapse scalars).
    """
    m = mort_by_age.copy()
    m["q_table"] = m["attained_age"].map(lambda a: mortality_table.q(int(a)))
    m["expected_deaths"] = m["exposure_years"] * m["q_table"]

    lp = lapse_by_year.copy()
    lp["lapse_rate_base"] = lp["policy_year"].map(lambda t: lapse_curve_base.rate(int(t)))
    lp["expected_lapses"] = lp["exposure_at_lapse_risk"] * lp["lapse_rate_base"]
    return m, lp
