"""
Term-life projection sandbox: tune assumptions, upload CSV curves / experience, view charts.

Run: streamlit run streamlit_app.py
Requires: pip install -e ".[ui]"
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from term_life.csv_loaders import (
    lapse_curve_from_csv,
    mortality_table_from_csv,
    parse_lapse_experience,
    parse_mortality_experience,
)
from term_life.experience import lapse_experience_summary, mortality_ae_ratio, suggest_k_mort_from_ae
from term_life.projection import Assumptions, Portfolio, project_term_life
from term_life.tables import stylized_base_lapse_by_duration, stylized_base_mortality_table


def _fig_decrements(policy_years: list[int], proj: Any) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("In-force at start of year", "Deaths and lapses during year"),
    )
    fig.add_trace(
        go.Bar(x=policy_years, y=list(proj.active_start), name="Active (start)", marker_color="#2E86AB"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=policy_years, y=list(proj.deaths), name="Deaths", marker_color="#A23B72"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=policy_years, y=list(proj.lapses), name="Lapses", marker_color="#F18F01"),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Policy year", row=2, col=1)
    fig.update_layout(height=520, showlegend=True, barmode="group", margin=dict(t=40, b=40))
    return fig


def _fig_decrements_compare(
    policy_years: list[int],
    proj_a: Any,
    proj_b: Any,
    name_a: str = "A",
    name_b: str = "B",
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"In-force at start ({name_a} vs {name_b})", "Deaths and lapses"),
    )
    fig.add_trace(
        go.Scatter(
            x=policy_years,
            y=list(proj_a.active_start),
            mode="lines+markers",
            name=f"Active {name_a}",
            line=dict(color="#2E86AB"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=policy_years,
            y=list(proj_b.active_start),
            mode="lines+markers",
            name=f"Active {name_b}",
            line=dict(color="#2E86AB", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=policy_years, y=list(proj_a.deaths), mode="lines+markers", name=f"Deaths {name_a}", line=dict(color="#A23B72")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=policy_years,
            y=list(proj_b.deaths),
            mode="lines+markers",
            name=f"Deaths {name_b}",
            line=dict(color="#A23B72", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=policy_years, y=list(proj_a.lapses), mode="lines+markers", name=f"Lapses {name_a}", line=dict(color="#F18F01")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=policy_years,
            y=list(proj_b.lapses),
            mode="lines+markers",
            name=f"Lapses {name_b}",
            line=dict(color="#F18F01", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Policy year", row=2, col=1)
    fig.update_layout(height=560, showlegend=True, margin=dict(t=40, b=40))
    return fig


def _fig_cashflows(policy_years: list[int], proj: Any) -> go.Figure:
    prem = list(proj.premium_cf)
    death = [-x for x in proj.death_benefit_cf]
    exp = [-x for x in proj.expense_cf]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=policy_years, y=prem, mode="lines+markers", name="Premiums", line=dict(color="#2E86AB")))
    fig.add_trace(go.Scatter(x=policy_years, y=death, mode="lines+markers", name="Death benefits", line=dict(color="#A23B72")))
    fig.add_trace(go.Scatter(x=policy_years, y=exp, mode="lines+markers", name="Expenses", line=dict(color="#6A994E")))
    fig.update_layout(
        title="Undiscounted cash flows by policy year",
        xaxis_title="Policy year",
        yaxis_title="Amount",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fig_cashflows_compare(policy_years: list[int], proj_a: Any, proj_b: Any, name_a: str = "A", name_b: str = "B") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=policy_years,
            y=list(proj_a.premium_cf),
            mode="lines+markers",
            name=f"Premiums {name_a}",
            line=dict(color="#2E86AB"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=policy_years,
            y=list(proj_b.premium_cf),
            mode="lines+markers",
            name=f"Premiums {name_b}",
            line=dict(color="#2E86AB", dash="dash"),
        )
    )
    da = [-x for x in proj_a.death_benefit_cf]
    db = [-x for x in proj_b.death_benefit_cf]
    fig.add_trace(go.Scatter(x=policy_years, y=da, mode="lines+markers", name=f"Death ben. {name_a}", line=dict(color="#A23B72")))
    fig.add_trace(
        go.Scatter(x=policy_years, y=db, mode="lines+markers", name=f"Death ben. {name_b}", line=dict(color="#A23B72", dash="dash"))
    )
    fig.update_layout(
        title=f"Cash flows: {name_a} (solid) vs {name_b} (dashed)",
        xaxis_title="Policy year",
        yaxis_title="Amount",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fig_effective_rates(issue_age: int, term: int, assumptions: Assumptions) -> go.Figure:
    years = list(range(1, term + 1))
    q_eff = [assumptions.q_effective(issue_age + t - 1) for t in years]
    l_eff = [assumptions.lapse_effective(t) for t in years]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=q_eff, name="q effective (age path)", line=dict(color="#A23B72")))
    fig.add_trace(go.Scatter(x=years, y=l_eff, name="lapse effective", line=dict(color="#F18F01")))
    fig.update_layout(
        title="Effective annual rates by policy year",
        xaxis_title="Policy year",
        yaxis_title="Rate",
        height=380,
        yaxis_tickformat=".3%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fig_effective_rates_compare(
    issue_age_a: int,
    term_a: int,
    asmp_a: Assumptions,
    issue_age_b: int,
    term_b: int,
    asmp_b: Assumptions,
    name_a: str = "A",
    name_b: str = "B",
) -> go.Figure:
    years = list(range(1, max(term_a, term_b) + 1))
    q_a = [asmp_a.q_effective(issue_age_a + t - 1) if t <= term_a else float("nan") for t in years]
    q_b = [asmp_b.q_effective(issue_age_b + t - 1) if t <= term_b else float("nan") for t in years]
    l_a = [asmp_a.lapse_effective(t) if t <= term_a else float("nan") for t in years]
    l_b = [asmp_b.lapse_effective(t) if t <= term_b else float("nan") for t in years]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=q_a, name=f"q {name_a}", line=dict(color="#A23B72")))
    fig.add_trace(go.Scatter(x=years, y=q_b, name=f"q {name_b}", line=dict(color="#A23B72", dash="dash")))
    fig.add_trace(go.Scatter(x=years, y=l_a, name=f"lapse {name_a}", line=dict(color="#F18F01")))
    fig.add_trace(go.Scatter(x=years, y=l_b, name=f"lapse {name_b}", line=dict(color="#F18F01", dash="dash")))
    fig.update_layout(
        title=f"Effective rates: {name_a} (solid) vs {name_b} (dashed)",
        xaxis_title="Policy year",
        yaxis_title="Rate",
        height=400,
        yaxis_tickformat=".3%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fig_pv_delta_bar(labels: list[str], deltas: list[float]) -> go.Figure:
    colors = ["#6A994E" if d >= 0 else "#BC4749" for d in deltas]
    fig = go.Figure(go.Bar(x=labels, y=deltas, marker_color=colors, text=[f"{d:,.0f}" for d in deltas], textposition="outside"))
    fig.update_layout(
        title="Δ PV (B − A)",
        yaxis_title="Amount",
        height=360,
        showlegend=False,
    )
    return fig


def _projection_ae_by_year(
    portfolio: Portfolio,
    assumptions: Assumptions,
    proj: Any,
) -> pd.DataFrame:
    """
    Actual/expected vs the scenario's own effective rates (pre-rounding expected counts).
    Deaths: E[D_t] = A_t * q_eff; Lapses: E[L_t] = (A_t - D_t) * l_eff — uses rounded D_t for lapse exposure to match projection step order.
    """
    rows: list[dict[str, Any]] = []
    t_end = assumptions.term_years
    for t in range(t_end):
        py = t + 1
        age = portfolio.issue_age + t
        a_start = proj.active_start[t]
        d_act = proj.deaths[t]
        l_act = proj.lapses[t]
        q_eff = assumptions.q_effective(age)
        l_eff = assumptions.lapse_effective(py)
        exp_deaths = float(a_start) * q_eff
        surv_after_death = max(0.0, float(a_start) - float(d_act))
        exp_lapses = surv_after_death * l_eff
        ae_death = (d_act / exp_deaths) if exp_deaths > 0 else float("nan")
        ae_lapse = (l_act / exp_lapses) if exp_lapses > 0 else float("nan")
        rows.append(
            {
                "policy_year": py,
                "age": age,
                "active_start": a_start,
                "deaths_actual": d_act,
                "deaths_expected": exp_deaths,
                "death_A/E": ae_death,
                "lapses_actual": l_act,
                "lapses_expected": exp_lapses,
                "lapse_A/E": ae_lapse,
            }
        )
    return pd.DataFrame(rows)


def _fig_ae_timeseries(policy_years: list[float], death_ae: list[float], lapse_ae: list[float], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=policy_years, y=death_ae, mode="lines+markers", name="Death A/E", line=dict(color="#A23B72")))
    fig.add_trace(go.Scatter(x=policy_years, y=lapse_ae, mode="lines+markers", name="Lapse A/E", line=dict(color="#F18F01")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="1.0")
    fig.update_layout(
        title=title,
        xaxis_title="Policy year",
        yaxis_title="A/E (actual ÷ expected)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fig_ae_compare_timeseries(
    policy_years: list[int],
    death_ae_a: list[float],
    lapse_ae_a: list[float],
    death_ae_b: list[float],
    lapse_ae_b: list[float],
    name_a: str = "A",
    name_b: str = "B",
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"Death A/E ({name_a} solid, {name_b} dashed)", f"Lapse A/E ({name_a} solid, {name_b} dashed)"),
    )
    fig.add_trace(
        go.Scatter(x=policy_years, y=death_ae_a, mode="lines+markers", name=f"Death {name_a}", line=dict(color="#A23B72")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=policy_years,
            y=death_ae_b,
            mode="lines+markers",
            name=f"Death {name_b}",
            line=dict(color="#A23B72", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_trace(
        go.Scatter(x=policy_years, y=lapse_ae_a, mode="lines+markers", name=f"Lapse {name_a}", line=dict(color="#F18F01")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=policy_years,
            y=lapse_ae_b,
            mode="lines+markers",
            name=f"Lapse {name_b}",
            line=dict(color="#F18F01", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_xaxes(title_text="Policy year", row=2, col=1)
    fig.update_yaxes(title_text="A/E", row=1, col=1)
    fig.update_yaxes(title_text="A/E", row=2, col=1)
    fig.update_layout(height=520, showlegend=True, margin=dict(t=40, b=40))
    return fig


@dataclass
class ScenarioParams:
    n0: int
    issue_age: int
    term_years: int
    sum_assured: float
    annual_premium: float
    annual_expense: float
    discount_rate: float
    k_mort: float
    k_lapse: float
    mort_mode: str
    lapse_mode: str


def _scenario_sidebar(prefix: str) -> ScenarioParams:
    """Widgets for one scenario; `prefix` is '' or 'a_' or 'b_' for Streamlit keys."""
    pk = lambda k: f"{prefix}{k}" if prefix else k

    if not prefix:
        st.subheader("Portfolio")
    st.number_input("Initial active policies (N₀)", min_value=1, value=100_000, step=1_000, key=pk("n0"))
    st.number_input("Issue age", min_value=18, max_value=95, value=40, key=pk("issue_age"))
    st.number_input("Term (years)", min_value=1, max_value=40, value=20, key=pk("term_years"))
    st.number_input("Sum assured (per policy)", min_value=0.0, value=100_000.0, step=1_000.0, key=pk("sum_assured"))
    st.number_input("Annual premium (per policy)", min_value=0.0, value=500.0, step=10.0, key=pk("annual_premium"))
    st.number_input("Annual expense (per policy)", min_value=0.0, value=50.0, step=5.0, key=pk("annual_expense"))
    if not prefix:
        st.subheader("Assumptions")
    st.slider("Discount rate r", min_value=0.0, max_value=0.15, value=0.03, step=0.005, format="%.3f", key=pk("discount_rate"))
    st.slider("Mortality scalar k_mort", min_value=0.5, max_value=1.5, value=0.95, step=0.01, key=pk("k_mort"))
    st.slider("Lapse scalar k_lapse", min_value=0.5, max_value=1.5, value=1.0, step=0.01, key=pk("k_lapse"))

    st.subheader("Mortality table")
    st.radio("Source", ("Built-in stylized", "Upload CSV (merge into stylized)"), key=pk("mort_src"))
    st.file_uploader("Mortality CSV", type=["csv"], key=pk("mort_csv"))

    st.subheader("Lapse curve")
    st.radio("Source", ("Built-in stylized", "Upload CSV (merge into stylized)"), key=pk("lapse_src"))
    st.file_uploader("Lapse CSV", type=["csv"], key=pk("lapse_csv"))

    return ScenarioParams(
        n0=int(st.session_state[pk("n0")]),
        issue_age=int(st.session_state[pk("issue_age")]),
        term_years=int(st.session_state[pk("term_years")]),
        sum_assured=float(st.session_state[pk("sum_assured")]),
        annual_premium=float(st.session_state[pk("annual_premium")]),
        annual_expense=float(st.session_state[pk("annual_expense")]),
        discount_rate=float(st.session_state[pk("discount_rate")]),
        k_mort=float(st.session_state[pk("k_mort")]),
        k_lapse=float(st.session_state[pk("k_lapse")]),
        mort_mode=str(st.session_state[pk("mort_src")]),
        lapse_mode=str(st.session_state[pk("lapse_src")]),
    )


def _build_mortality_lapse(
    p: ScenarioParams,
    base_mort: Any,
    mort_upload: Any,
    lapse_upload: Any,
) -> tuple[Any, Any]:
    if p.mort_mode == "Upload CSV (merge into stylized)" and mort_upload is not None:
        mort_df = pd.read_csv(mort_upload)
        mortality_table = mortality_table_from_csv(mort_df, merge_with_stylized=True)
    else:
        mortality_table = base_mort

    if p.lapse_mode == "Upload CSV (merge into stylized)" and lapse_upload is not None:
        lapse_df = pd.read_csv(lapse_upload)
        lapse_curve = lapse_curve_from_csv(lapse_df, int(p.term_years), merge_with_stylized=True)
    else:
        lapse_curve = stylized_base_lapse_by_duration(int(p.term_years))
    return mortality_table, lapse_curve


def _run_scenario(
    p: ScenarioParams,
    base_mort: Any,
    mort_upload: Any,
    lapse_upload: Any,
) -> tuple[Portfolio, Assumptions, Any, pd.DataFrame]:
    mortality_table, lapse_curve = _build_mortality_lapse(p, base_mort, mort_upload, lapse_upload)
    max_age = p.issue_age + int(p.term_years) - 1
    missing_ages = [x for x in range(p.issue_age, max_age + 1) if x not in mortality_table.rates]
    if missing_ages:
        raise ValueError(
            f"Mortality table missing ages {missing_ages[:8]}{'…' if len(missing_ages) > 8 else ''}. "
            "Upload CSV covering issue age through issue age + term − 1."
        )

    portfolio = Portfolio(
        initial_active=int(p.n0),
        issue_age=int(p.issue_age),
        sum_assured=float(p.sum_assured),
        annual_premium=float(p.annual_premium),
        annual_expense_per_policy=float(p.annual_expense),
    )
    assumptions = Assumptions(
        term_years=int(p.term_years),
        mortality_table=mortality_table,
        lapse_curve=lapse_curve,
        k_mort=float(p.k_mort),
        k_lapse=float(p.k_lapse),
        discount_rate=float(p.discount_rate),
    )
    proj = project_term_life(portfolio, assumptions)
    ae_df = _projection_ae_by_year(portfolio, assumptions, proj)
    return portfolio, assumptions, proj, ae_df


def _mortality_experience_comparison_df(
    rows: list[dict[str, Any]],
    label_a: str,
    assumptions_a: Assumptions,
    label_b: str | None,
    assumptions_b: Assumptions | None,
) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for mort_row in rows:
        age = mort_row["age"]
        exp = mort_row["exposure"]
        d_obs = mort_row["observed_deaths"]
        ae_a = mortality_ae_ratio(d_obs, exp, age, assumptions_a.mortality_table)
        exp_a = exp * assumptions_a.mortality_table.q(age)
        row: dict[str, Any] = {
            "age": age,
            "exposure": exp,
            "observed_deaths": d_obs,
            f"expected_deaths_{label_a}_table": exp_a,
            f"A/E_vs_{label_a}_table": ae_a,
        }
        if assumptions_b is not None:
            ae_b = mortality_ae_ratio(d_obs, exp, age, assumptions_b.mortality_table)
            exp_b = exp * assumptions_b.mortality_table.q(age)
            row[f"expected_deaths_{label_b}_table"] = exp_b
            row[f"A/E_vs_{label_b}_table"] = ae_b
            if ae_a is not None and ae_b is not None and ae_b != 0:
                row["A/E_ratio (A÷B)"] = ae_a / ae_b
            else:
                row["A/E_ratio (A÷B)"] = float("nan")
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def main() -> None:
    st.set_page_config(page_title="Term-life projection", layout="wide")
    st.title("Term-life projection sandbox")
    st.caption(
        "Level-premium term: mortality table × k_mort, lapse curve × k_lapse, mid-year discounting."
    )

    base_mort = stylized_base_mortality_table()

    with st.sidebar:
        compare_mode = st.radio(
            "Dashboard mode",
            ("Single scenario", "Compare A vs B"),
            horizontal=True,
            help="Compare runs two full assumption sets and shows deltas (B − A).",
        )

        if compare_mode == "Single scenario":
            _scenario_sidebar("")
            p = ScenarioParams(
                n0=int(st.session_state["n0"]),
                issue_age=int(st.session_state["issue_age"]),
                term_years=int(st.session_state["term_years"]),
                sum_assured=float(st.session_state["sum_assured"]),
                annual_premium=float(st.session_state["annual_premium"]),
                annual_expense=float(st.session_state["annual_expense"]),
                discount_rate=float(st.session_state["discount_rate"]),
                k_mort=float(st.session_state["k_mort"]),
                k_lapse=float(st.session_state["k_lapse"]),
                mort_mode=str(st.session_state["mort_src"]),
                lapse_mode=str(st.session_state["lapse_src"]),
            )
            mort_upload = st.session_state.get("mort_csv")
            lapse_upload = st.session_state.get("lapse_csv")
        else:
            st.caption("Configure **A** and **B** in the tabs below.")
            tab_a, tab_b = st.tabs(["Scenario A", "Scenario B"])
            with tab_a:
                st.markdown("**Scenario A**")
                _scenario_sidebar("a_")
            with tab_b:
                st.markdown("**Scenario B**")
                _scenario_sidebar("b_")
            p_a = ScenarioParams(
                n0=int(st.session_state["a_n0"]),
                issue_age=int(st.session_state["a_issue_age"]),
                term_years=int(st.session_state["a_term_years"]),
                sum_assured=float(st.session_state["a_sum_assured"]),
                annual_premium=float(st.session_state["a_annual_premium"]),
                annual_expense=float(st.session_state["a_annual_expense"]),
                discount_rate=float(st.session_state["a_discount_rate"]),
                k_mort=float(st.session_state["a_k_mort"]),
                k_lapse=float(st.session_state["a_k_lapse"]),
                mort_mode=str(st.session_state["a_mort_src"]),
                lapse_mode=str(st.session_state["a_lapse_src"]),
            )
            p_b = ScenarioParams(
                n0=int(st.session_state["b_n0"]),
                issue_age=int(st.session_state["b_issue_age"]),
                term_years=int(st.session_state["b_term_years"]),
                sum_assured=float(st.session_state["b_sum_assured"]),
                annual_premium=float(st.session_state["b_annual_premium"]),
                annual_expense=float(st.session_state["b_annual_expense"]),
                discount_rate=float(st.session_state["b_discount_rate"]),
                k_mort=float(st.session_state["b_k_mort"]),
                k_lapse=float(st.session_state["b_k_lapse"]),
                mort_mode=str(st.session_state["b_mort_src"]),
                lapse_mode=str(st.session_state["b_lapse_src"]),
            )
            mort_upload_a = st.session_state.get("a_mort_csv")
            lapse_upload_a = st.session_state.get("a_lapse_csv")
            mort_upload_b = st.session_state.get("b_mort_csv")
            lapse_upload_b = st.session_state.get("b_lapse_csv")
            p = None

        st.divider()
        st.subheader("Experience uploads (optional)")
        st.file_uploader("Mortality experience CSV", type=["csv"], key="mort_exp")
        st.file_uploader("Lapse experience CSV", type=["csv"], key="lapse_exp")
        st.slider("Credibility w (for suggested k from aggregate A/E)", 0.0, 1.0, 1.0, 0.05, key="cred_w")

        with st.expander("CSV column help"):
            st.markdown(
                """
**Mortality:** `age` (or `x`) and `q` (or `q_x`, `rate`). Uploaded rows override the built-in table for those ages.

**Lapse:** `policy_year` (or `year`, `t`, `duration`) and `lapse` (or `rate`). Merged with built-in by policy year.

**Mortality experience (optional):** `age`, `exposure`, `observed_deaths` (or `deaths`).

**Lapse experience (optional):** `policy_year`, `exposure`, `lapses`.
                """
            )

        tmpl_mort = "age,q\n40,0.001\n50,0.002\n"
        tmpl_lapse = "policy_year,lapse\n1,0.08\n2,0.06\n3,0.04\n"
        tmpl_mort_exp = "age,exposure,observed_deaths\n45,50000,60\n46,48000,65\n"
        tmpl_lapse_exp = "policy_year,exposure,lapses\n1,100000,5500\n2,92000,2800\n"
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Template: mortality", tmpl_mort, "mortality_template.csv", "text/csv")
            st.download_button("Template: lapse", tmpl_lapse, "lapse_template.csv", "text/csv")
        with c2:
            st.download_button("Template: mort. exp.", tmpl_mort_exp, "mortality_experience_template.csv", "text/csv")
            st.download_button("Template: lapse exp.", tmpl_lapse_exp, "lapse_experience_template.csv", "text/csv")

    mort_exp_upload = st.session_state.get("mort_exp")
    lapse_exp_upload = st.session_state.get("lapse_exp")
    cred_w = float(st.session_state.get("cred_w", 1.0))

    try:
        if compare_mode == "Single scenario":
            assert p is not None
            portfolio, assumptions, proj, ae_proj_df = _run_scenario(p, base_mort, mort_upload, lapse_upload)
            proj_b = assumptions_b = portfolio_b = ae_proj_df_b = None
        else:
            portfolio, assumptions, proj, ae_proj_df = _run_scenario(p_a, base_mort, mort_upload_a, lapse_upload_a)
            portfolio_b, assumptions_b, proj_b, ae_proj_df_b = _run_scenario(p_b, base_mort, mort_upload_b, lapse_upload_b)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if compare_mode == "Single scenario":
        policy_years = list(range(1, proj.term_years + 1))

        st.subheader("A/E vs assumptions (projection)")
        st.caption(
            "Per policy year: actual decrements from the run ÷ expected counts using effective q and lapse "
            "(integer decrements vs continuous expectation; values near 1.0 indicate consistency with rates)."
        )
        c_ae1, c_ae2 = st.columns(2)
        with c_ae1:
            disp = ae_proj_df.copy()
            for c in ("deaths_expected", "lapses_expected", "death_A/E", "lapse_A/E"):
                if c in disp.columns:
                    disp[c] = disp[c].round(4) if "A/E" in c else disp[c].round(2)
            st.dataframe(disp, use_container_width=True, hide_index=True)
        with c_ae2:
            st.plotly_chart(
                _fig_ae_timeseries(
                    list(ae_proj_df["policy_year"]),
                    list(ae_proj_df["death_A/E"]),
                    list(ae_proj_df["lapse_A/E"]),
                    "Model consistency: death & lapse A/E by year",
                ),
                use_container_width=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PV premiums", f"{proj.pv_premiums:,.0f}")
        m2.metric("PV death benefits", f"{proj.pv_death_benefits:,.0f}")
        m3.metric("PV expenses", f"{proj.pv_expenses:,.0f}")
        m4.metric("PV profit (prem + ben + exp)", f"{proj.pv_profit:,.0f}")

        st.subheader("Charts")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_fig_decrements(policy_years, proj), use_container_width=True)
        with c2:
            st.plotly_chart(_fig_cashflows(policy_years, proj), use_container_width=True)
        st.plotly_chart(_fig_effective_rates(int(portfolio.issue_age), int(assumptions.term_years), assumptions), use_container_width=True)

        st.subheader("Projection table")
        tbl = pd.DataFrame(
            {
                "policy_year": policy_years,
                "active_start": list(proj.active_start),
                "deaths": list(proj.deaths),
                "lapses": list(proj.lapses),
                "premium_cf": list(proj.premium_cf),
                "death_benefit_cf": list(proj.death_benefit_cf),
                "expense_cf": list(proj.expense_cf),
            }
        )
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        csv_buf = io.StringIO()
        tbl.to_csv(csv_buf, index=False)
        st.download_button(
            "Download projection CSV",
            csv_buf.getvalue(),
            "term_life_projection.csv",
            "text/csv",
        )

    else:
        # Align to common policy years for comparison
        t_common = min(proj.term_years, proj_b.term_years)
        if proj.term_years != proj_b.term_years:
            st.warning(
                f"Scenarios use different terms ({proj.term_years} vs {proj_b.term_years} years). "
                f"Comparisons and deltas use the first **{t_common}** policy years."
            )
        policy_years = list(range(1, t_common + 1))

        st.subheader("Compare A vs B — present value summary")
        d_prem = proj_b.pv_premiums - proj.pv_premiums
        d_ben = proj_b.pv_death_benefits - proj.pv_death_benefits
        d_exp = proj_b.pv_expenses - proj.pv_expenses
        d_profit = proj_b.pv_profit - proj.pv_profit
        ca, cb, cd = st.columns(3)
        with ca:
            st.markdown("**Scenario A**")
            st.metric("PV premiums", f"{proj.pv_premiums:,.0f}")
            st.metric("PV death benefits", f"{proj.pv_death_benefits:,.0f}")
            st.metric("PV expenses", f"{proj.pv_expenses:,.0f}")
            st.metric("PV profit", f"{proj.pv_profit:,.0f}")
        with cb:
            st.markdown("**Scenario B**")
            st.metric("PV premiums", f"{proj_b.pv_premiums:,.0f}")
            st.metric("PV death benefits", f"{proj_b.pv_death_benefits:,.0f}")
            st.metric("PV expenses", f"{proj_b.pv_expenses:,.0f}")
            st.metric("PV profit", f"{proj_b.pv_profit:,.0f}")
        with cd:
            st.markdown("**Δ (B − A)**")
            st.metric("Δ PV premiums", f"{d_prem:+,.0f}")
            st.metric("Δ PV death benefits", f"{d_ben:+,.0f}")
            st.metric("Δ PV expenses", f"{d_exp:+,.0f}")
            st.metric("Δ PV profit", f"{d_profit:+,.0f}")

        st.plotly_chart(
            _fig_pv_delta_bar(
                ["Premiums", "Death ben.", "Expenses", "Profit"],
                [d_prem, d_ben, d_exp, d_profit],
            ),
            use_container_width=True,
        )

        st.subheader("A/E vs assumptions (both scenarios)")
        st.caption("Projection-path A/E for each scenario on the same policy years (truncated to common length).")
        st.plotly_chart(
            _fig_ae_compare_timeseries(
                policy_years,
                list(ae_proj_df["death_A/E"].iloc[:t_common]),
                list(ae_proj_df["lapse_A/E"].iloc[:t_common]),
                list(ae_proj_df_b["death_A/E"].iloc[:t_common]),
                list(ae_proj_df_b["lapse_A/E"].iloc[:t_common]),
                "A",
                "B",
            ),
            use_container_width=True,
        )

        diff_tbl = pd.DataFrame(
            {
                "policy_year": policy_years,
                "Δ active_start": [proj_b.active_start[t] - proj.active_start[t] for t in range(t_common)],
                "Δ deaths": [proj_b.deaths[t] - proj.deaths[t] for t in range(t_common)],
                "Δ lapses": [proj_b.lapses[t] - proj.lapses[t] for t in range(t_common)],
                "Δ premium_cf": [proj_b.premium_cf[t] - proj.premium_cf[t] for t in range(t_common)],
                "Δ death_benefit_cf": [proj_b.death_benefit_cf[t] - proj.death_benefit_cf[t] for t in range(t_common)],
                "Δ expense_cf": [proj_b.expense_cf[t] - proj.expense_cf[t] for t in range(t_common)],
            }
        )
        st.subheader("Year-by-year differences (B − A)")
        st.dataframe(diff_tbl, use_container_width=True, hide_index=True)

        st.subheader("Overlay charts (A solid, B dashed)")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(_fig_decrements_compare(policy_years, proj, proj_b), use_container_width=True)
        with cc2:
            st.plotly_chart(_fig_cashflows_compare(policy_years, proj, proj_b), use_container_width=True)
        st.plotly_chart(
            _fig_effective_rates_compare(
                portfolio.issue_age,
                assumptions.term_years,
                assumptions,
                portfolio_b.issue_age,
                assumptions_b.term_years,
                assumptions_b,
            ),
            use_container_width=True,
        )

        st.subheader("Projection side-by-side")
        tbl_a = pd.DataFrame(
            {
                "policy_year": policy_years,
                "A_active": list(proj.active_start[:t_common]),
                "A_deaths": list(proj.deaths[:t_common]),
                "A_lapses": list(proj.lapses[:t_common]),
            }
        )
        tbl_b = pd.DataFrame(
            {
                "policy_year": policy_years,
                "B_active": list(proj_b.active_start[:t_common]),
                "B_deaths": list(proj_b.deaths[:t_common]),
                "B_lapses": list(proj_b.lapses[:t_common]),
            }
        )
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**A**")
            st.dataframe(tbl_a, use_container_width=True, hide_index=True)
        with s2:
            st.markdown("**B**")
            st.dataframe(tbl_b, use_container_width=True, hide_index=True)

    # Experience panels (shared uploads); compare A vs B expected when in compare mode
    has_mort_exp = mort_exp_upload is not None
    has_lapse_exp = lapse_exp_upload is not None
    if has_mort_exp or has_lapse_exp:
        st.divider()
        st.subheader("Experience study — A/E vs assumptions")
        if compare_mode == "Compare A vs B":
            st.caption("Same uploaded experience evaluated against **A** and **B** base mortality tables (before k_mort).")
        ec1, ec2 = st.columns(2)

        if has_mort_exp:
            with ec1:
                try:
                    medf = pd.read_csv(mort_exp_upload)
                    rows = parse_mortality_experience(medf)
                    if compare_mode == "Compare A vs B":
                        cmp_df = _mortality_experience_comparison_df(rows, "A", assumptions, "B", assumptions_b)
                        total_d_obs = sum(float(r["observed_deaths"]) for r in rows)
                        total_exp_a = sum(float(r["exposure"]) * assumptions.mortality_table.q(int(r["age"])) for r in rows)
                        total_exp_b = sum(float(r["exposure"]) * assumptions_b.mortality_table.q(int(r["age"])) for r in rows)
                        agg_a = total_d_obs / total_exp_a if total_exp_a > 0 else None
                        agg_b = total_d_obs / total_exp_b if total_exp_b > 0 else None
                        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
                        if agg_a is not None and agg_b is not None:
                            cma, cmb = st.columns(2)
                            with cma:
                                st.metric("Aggregate A/E vs A table", f"{agg_a:.4f}")
                                st.caption(f"Suggested k_mort (A): {suggest_k_mort_from_ae(agg_a, cred_w):.4f}")
                            with cmb:
                                st.metric("Aggregate A/E vs B table", f"{agg_b:.4f}")
                                st.caption(f"Suggested k_mort (B): {suggest_k_mort_from_ae(agg_b, cred_w):.4f}")
                        fig_cmp = go.Figure()
                        if "A/E_vs_A_table" in cmp_df.columns:
                            fig_cmp.add_trace(
                                go.Bar(x=cmp_df["age"], y=cmp_df["A/E_vs_A_table"], name="A/E vs A table", marker_color="#2E86AB")
                            )
                        if "A/E_vs_B_table" in cmp_df.columns:
                            fig_cmp.add_trace(
                                go.Bar(x=cmp_df["age"], y=cmp_df["A/E_vs_B_table"], name="A/E vs B table", marker_color="#F4A261")
                            )
                        fig_cmp.add_hline(y=1.0, line_dash="dash", line_color="gray")
                        fig_cmp.update_layout(
                            title="Mortality experience A/E: A vs B base tables",
                            barmode="group",
                            height=400,
                            yaxis_title="A/E",
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)
                    else:
                        ae_rows = []
                        total_d_obs = 0.0
                        total_d_exp = 0.0
                        for mort_row in rows:
                            age = mort_row["age"]
                            exp = mort_row["exposure"]
                            d_obs = mort_row["observed_deaths"]
                            ae = mortality_ae_ratio(d_obs, exp, age, assumptions.mortality_table)
                            q_tbl = assumptions.mortality_table.q(age)
                            d_exp = exp * q_tbl
                            total_d_obs += d_obs
                            total_d_exp += d_exp
                            ae_rows.append(
                                {
                                    "age": age,
                                    "exposure": exp,
                                    "observed_deaths": d_obs,
                                    "expected_deaths_table": d_exp,
                                    "A/E": ae,
                                    "q_table": q_tbl,
                                }
                            )
                        ae_df = pd.DataFrame(ae_rows)
                        st.dataframe(ae_df, use_container_width=True, hide_index=True)
                        agg_ae = total_d_obs / total_d_exp if total_d_exp > 0 else None
                        if agg_ae is not None:
                            k_sugg = suggest_k_mort_from_ae(agg_ae, partial_credibility=cred_w)
                            st.metric("Aggregate A/E (vs base table)", f"{agg_ae:.4f}")
                            st.metric("Suggested k_mort (toy credibility)", f"{k_sugg:.4f}")
                        fig_ae = go.Figure()
                        plot_ae = ae_df.dropna(subset=["A/E"])
                        if not plot_ae.empty:
                            fig_ae.add_trace(
                                go.Bar(
                                    x=plot_ae["age"],
                                    y=plot_ae["A/E"],
                                    name="A/E by age",
                                    marker_color="#7209B7",
                                )
                            )
                        fig_ae.add_hline(y=1.0, line_dash="dash", line_color="gray")
                        fig_ae.update_layout(title="Mortality A/E by age", height=360, yaxis_title="A/E")
                        st.plotly_chart(fig_ae, use_container_width=True)
                except Exception as e:
                    st.warning(f"Mortality experience: {e}")

        if has_lapse_exp:
            with ec2:
                try:
                    ledf = pd.read_csv(lapse_exp_upload)
                    by_year = parse_lapse_experience(ledf)
                    summary = lapse_experience_summary(by_year)
                    summ_df = pd.DataFrame(
                        {
                            "policy_year": [r.policy_year for r in summary],
                            "exposure": [r.exposure for r in summary],
                            "lapses_observed": [r.lapses_observed for r in summary],
                            "crude_rate": [r.crude_rate for r in summary],
                        }
                    )
                    model_rates_a: list[float] = []
                    model_rates_b: list[float] = []
                    for lap_row in summary:
                        try:
                            base_a = assumptions.lapse_curve.rate(lap_row.policy_year)
                            model_rates_a.append(min(1.0, float(assumptions.k_lapse) * base_a))
                        except Exception:
                            model_rates_a.append(float("nan"))
                        if compare_mode == "Compare A vs B":
                            try:
                                base_b = assumptions_b.lapse_curve.rate(lap_row.policy_year)
                                model_rates_b.append(min(1.0, float(assumptions_b.k_lapse) * base_b))
                            except Exception:
                                model_rates_b.append(float("nan"))
                    summ_df["A_model_lapse"] = model_rates_a
                    summ_df["A/E_vs_A"] = summ_df.apply(
                        lambda r: (r["crude_rate"] / r["A_model_lapse"])
                        if pd.notna(r["crude_rate"]) and pd.notna(r["A_model_lapse"]) and r["A_model_lapse"] > 0
                        else float("nan"),
                        axis=1,
                    )
                    if compare_mode == "Compare A vs B":
                        summ_df["B_model_lapse"] = model_rates_b
                        summ_df["A/E_vs_B"] = summ_df.apply(
                            lambda r: (r["crude_rate"] / r["B_model_lapse"])
                            if pd.notna(r["crude_rate"]) and pd.notna(r["B_model_lapse"]) and r["B_model_lapse"] > 0
                            else float("nan"),
                            axis=1,
                        )
                    st.dataframe(summ_df, use_container_width=True, hide_index=True)
                    fig_l = go.Figure()
                    if summ_df["crude_rate"].notna().any():
                        fig_l.add_trace(
                            go.Scatter(
                                x=summ_df["policy_year"],
                                y=summ_df["crude_rate"],
                                mode="lines+markers",
                                name="Crude lapse",
                                line=dict(color="#333"),
                            )
                        )
                    fig_l.add_trace(
                        go.Scatter(
                            x=summ_df["policy_year"],
                            y=summ_df["A_model_lapse"],
                            mode="lines+markers",
                            name="Model A",
                            line=dict(color="#2E86AB"),
                        )
                    )
                    if compare_mode == "Compare A vs B":
                        fig_l.add_trace(
                            go.Scatter(
                                x=summ_df["policy_year"],
                                y=summ_df["B_model_lapse"],
                                mode="lines+markers",
                                name="Model B",
                                line=dict(color="#F4A261", dash="dash"),
                            )
                        )
                    fig_l.update_layout(
                        title="Lapse: crude vs model (A solid, B dashed)",
                        xaxis_title="Policy year",
                        yaxis_title="Rate",
                        yaxis_tickformat=".1%",
                        height=400,
                    )
                    st.plotly_chart(fig_l, use_container_width=True)

                    if summ_df["A/E_vs_A"].notna().any() or (
                        compare_mode == "Compare A vs B" and summ_df.get("A/E_vs_B") is not None and summ_df["A/E_vs_B"].notna().any()
                    ):
                        fig_lae = go.Figure()
                        fig_lae.add_trace(
                            go.Scatter(
                                x=summ_df["policy_year"],
                                y=summ_df["A/E_vs_A"],
                                mode="lines+markers",
                                name="A/E crude ÷ model A",
                                line=dict(color="#2E86AB"),
                            )
                        )
                        if compare_mode == "Compare A vs B" and "A/E_vs_B" in summ_df.columns:
                            fig_lae.add_trace(
                                go.Scatter(
                                    x=summ_df["policy_year"],
                                    y=summ_df["A/E_vs_B"],
                                    mode="lines+markers",
                                    name="A/E crude ÷ model B",
                                    line=dict(color="#F4A261", dash="dash"),
                                )
                            )
                        fig_lae.add_hline(y=1.0, line_dash="dash", line_color="gray")
                        fig_lae.update_layout(
                            title="Lapse experience A/E (crude ÷ model effective lapse)",
                            xaxis_title="Policy year",
                            yaxis_title="A/E",
                            height=360,
                        )
                        st.plotly_chart(fig_lae, use_container_width=True)
                except Exception as e:
                    st.warning(f"Lapse experience: {e}")


if __name__ == "__main__":
    main()
