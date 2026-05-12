"""
Term-life projection sandbox: tune assumptions, upload CSV curves / experience, view charts.

Run: streamlit run streamlit_app.py
Requires: pip install -e ".[ui]"
"""

from __future__ import annotations

import io
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


def _fig_cashflows(policy_years: list[int], proj: Any) -> go.Figure:
    prem = list(proj.premium_cf)
    death = [-x for x in proj.death_benefit_cf]  # positive outflow for display
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


def main() -> None:
    st.set_page_config(page_title="Term-life projection", layout="wide")
    st.title("Term-life projection sandbox")
    st.caption(
        "Level-premium term: mortality table × k_mort, lapse curve × k_lapse, mid-year discounting."
    )

    base_mort = stylized_base_mortality_table()
    default_term = 20

    with st.sidebar:
        st.header("Portfolio")
        n0 = st.number_input("Initial active policies (N₀)", min_value=1, value=100_000, step=1_000)
        issue_age = st.number_input("Issue age", min_value=18, max_value=95, value=40)
        term_years = st.number_input("Term (years)", min_value=1, max_value=40, value=default_term)
        sum_assured = st.number_input("Sum assured (per policy)", min_value=0.0, value=100_000.0, step=1_000.0)
        annual_premium = st.number_input("Annual premium (per policy)", min_value=0.0, value=500.0, step=10.0)
        annual_expense = st.number_input("Annual expense (per policy)", min_value=0.0, value=50.0, step=5.0)

        st.header("Assumptions")
        discount_rate = st.slider("Discount rate r", min_value=0.0, max_value=0.15, value=0.03, step=0.005, format="%.3f")
        k_mort = st.slider("Mortality scalar k_mort", min_value=0.5, max_value=1.5, value=0.95, step=0.01)
        k_lapse = st.slider("Lapse scalar k_lapse", min_value=0.5, max_value=1.5, value=1.0, step=0.01)

        st.subheader("Mortality table")
        mort_mode = st.radio("Source", ("Built-in stylized", "Upload CSV (merge into stylized)"), key="mort_src")
        mort_upload = st.file_uploader("Mortality CSV", type=["csv"], key="mort_csv")

        st.subheader("Lapse curve")
        lapse_mode = st.radio("Source", ("Built-in stylized", "Upload CSV (merge into stylized)"), key="lapse_src")
        lapse_upload = st.file_uploader("Lapse CSV", type=["csv"], key="lapse_csv")

        with st.expander("CSV column help"):
            st.markdown(
                """
**Mortality:** `age` (or `x`) and `q` (or `q_x`, `rate`). Uploaded rows override the built-in table for those ages.

**Lapse:** `policy_year` (or `year`, `t`, `duration`) and `lapse` (or `rate`). Merged with built-in by policy year.

**Mortality experience (optional):** `age`, `exposure`, `observed_deaths` (or `deaths`).

**Lapse experience (optional):** `policy_year`, `exposure`, `lapses`.
                """
            )

        st.subheader("Experience uploads (optional)")
        mort_exp_upload = st.file_uploader("Mortality experience CSV", type=["csv"], key="mort_exp")
        lapse_exp_upload = st.file_uploader("Lapse experience CSV", type=["csv"], key="lapse_exp")
        cred_w = st.slider("Credibility w (for suggested k from aggregate A/E)", 0.0, 1.0, 1.0, 0.05)

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

    # Build tables
    try:
        if mort_mode == "Upload CSV (merge into stylized)" and mort_upload is not None:
            mort_df = pd.read_csv(mort_upload)
            mortality_table = mortality_table_from_csv(mort_df, merge_with_stylized=True)
        else:
            mortality_table = base_mort

        if lapse_mode == "Upload CSV (merge into stylized)" and lapse_upload is not None:
            lapse_df = pd.read_csv(lapse_upload)
            lapse_curve = lapse_curve_from_csv(lapse_df, int(term_years), merge_with_stylized=True)
        else:
            lapse_curve = stylized_base_lapse_by_duration(int(term_years))
    except Exception as e:
        st.error(f"Could not load uploaded assumptions: {e}")
        st.stop()

    max_age = issue_age + int(term_years) - 1
    missing_ages = [x for x in range(issue_age, max_age + 1) if x not in mortality_table.rates]
    if missing_ages:
        st.error(
            f"Mortality table missing rates for ages needed by the projection: "
            f"{missing_ages[:10]}{'…' if len(missing_ages) > 10 else ''}. "
            "Upload a CSV that covers issue age through issue age + term − 1, or use built-in stylized."
        )
        st.stop()

    portfolio = Portfolio(
        initial_active=int(n0),
        issue_age=int(issue_age),
        sum_assured=float(sum_assured),
        annual_premium=float(annual_premium),
        annual_expense_per_policy=float(annual_expense),
    )
    assumptions = Assumptions(
        term_years=int(term_years),
        mortality_table=mortality_table,
        lapse_curve=lapse_curve,
        k_mort=float(k_mort),
        k_lapse=float(k_lapse),
        discount_rate=float(discount_rate),
    )

    proj = project_term_life(portfolio, assumptions)
    policy_years = list(range(1, proj.term_years + 1))

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
    st.plotly_chart(_fig_effective_rates(int(issue_age), int(term_years), assumptions), use_container_width=True)

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

    # Experience panels
    has_mort_exp = mort_exp_upload is not None
    has_lapse_exp = lapse_exp_upload is not None
    if has_mort_exp or has_lapse_exp:
        st.divider()
        st.subheader("Experience study (uploaded)")
        ec1, ec2 = st.columns(2)

        if has_mort_exp:
            with ec1:
                try:
                    medf = pd.read_csv(mort_exp_upload)
                    rows = parse_mortality_experience(medf)
                    ae_rows = []
                    total_d_obs = 0.0
                    total_d_exp = 0.0
                    for mort_row in rows:
                        age = mort_row["age"]
                        exp = mort_row["exposure"]
                        d_obs = mort_row["observed_deaths"]
                        ae = mortality_ae_ratio(d_obs, exp, age, mortality_table)
                        q_tbl = mortality_table.q(age)
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
                        st.metric("Aggregate A/E (vs uploaded/base table)", f"{agg_ae:.4f}")
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
                    st.dataframe(summ_df, use_container_width=True, hide_index=True)
                    # Compare crude to model base × k_lapse for same years
                    model_rates: list[float] = []
                    for lap_row in summary:
                        try:
                            base = lapse_curve.rate(lap_row.policy_year)
                            model_rates.append(min(1.0, float(k_lapse) * base))
                        except Exception:
                            model_rates.append(float("nan"))
                    summ_df["model_lapse_effective"] = model_rates
                    fig_l = go.Figure()
                    if summ_df["crude_rate"].notna().any():
                        fig_l.add_trace(
                            go.Scatter(
                                x=summ_df["policy_year"],
                                y=summ_df["crude_rate"],
                                mode="lines+markers",
                                name="Crude lapse rate",
                            )
                        )
                    fig_l.add_trace(
                        go.Scatter(
                            x=summ_df["policy_year"],
                            y=summ_df["model_lapse_effective"],
                            mode="lines+markers",
                            name="Model lapse (current UI)",
                        )
                    )
                    fig_l.update_layout(
                        title="Lapse: crude experience vs model",
                        xaxis_title="Policy year",
                        yaxis_title="Rate",
                        yaxis_tickformat=".1%",
                        height=360,
                    )
                    st.plotly_chart(fig_l, use_container_width=True)
                except Exception as e:
                    st.warning(f"Lapse experience: {e}")


if __name__ == "__main__":
    main()
