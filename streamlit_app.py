"""
Term-life experience workflow (Steps 1–6) + projection sandbox.

Run: streamlit run streamlit_app.py
Requires: pip install -e ".[ui]"
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from term_life.credibility import credibility_weighted_ae, limited_fluctuation_z
from term_life.csv_loaders import lapse_curve_from_csv, mortality_table_from_csv
from term_life.experience import lapse_experience_summary, mortality_ae_ratio, suggest_k_mort_from_ae
from term_life.policy_experience import (
    expected_decrements_from_rollups,
    parse_policy_level_experience,
    rollup_lapse_by_policy_year,
    rollup_mortality_by_age,
)
from term_life.projection import Assumptions, Portfolio, project_term_life
from term_life.simulated_experience import demo_policy_csv_text, generate_policy_level_term_experience
from term_life.tables import stylized_base_lapse_by_duration, stylized_base_mortality_table
from term_life.version_store import (
    FORMAT_TAG,
    MAX_SHARE_URL_CHARS,
    VersionGraph,
    apply_payload_to_session,
    bundle_for_share,
    collect_payload_from_session,
    decode_bundle,
    encode_bundle,
    graph_from_bundle,
    share_token_fits_url,
)

BUNDLED_POLICY_CSV = Path(__file__).resolve().parent / "data/simulated_experience/policy_level_term_cohort.csv"

STEP_GUIDE = {
    1: (
        "**Step 1 — Ingest & prepare** · Pull policy attributes, exposure (policy-years, including partial), "
        "and decrement flags (death, lapse, active). Use the bundled sample or upload your own CSV."
    ),
    2: (
        "**Step 2 — Expected decrements** · Using the **base** mortality table and **base** lapse curve "
        "(no k scalars), compute expected deaths = exposure × q_table and expected lapses = "
        "lapse-risk exposure × ℓ_t^base."
    ),
    3: (
        "**Step 3 — A/E ratios** · Compare actual decrements from the experience file to Step 2 expectations "
        "by attained age and policy year."
    ),
    4: (
        "**Step 4 — Credibility** · Limited-fluctuation Z from claim counts (defaults: full credibility at "
        "1,082 deaths / 1,082 lapses), then blend observed A/E toward 1.0."
    ),
    5: (
        "**Step 5 — Update parameters** · Map blended mortality A/E to a new k_mort on the table; "
        "scale k_lapse by blended lapse A/E vs the base curve. Apply these into the projection sliders."
    ),
    6: (
        "**Step 6 — Impact analysis** · Run the term-life projection with **current sliders** vs "
        "**experience-informed** parameters and compare PV cash flows and profit."
    ),
}


def _init_workflow_session() -> None:
    defaults = {
        "policy_exp_df": None,
        "exp_k_mort": None,
        "exp_k_lapse": None,
        "n_full_deaths": 1082.0,
        "n_full_lapses": 1082.0,
        "wf_step": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Legacy key from older builds conflicts with Streamlit's widget binding rules.
    st.session_state.pop("workflow_step", None)


def _init_version_graph() -> None:
    if "vc_graph" not in st.session_state or not isinstance(st.session_state.vc_graph, VersionGraph):
        st.session_state.vc_graph = VersionGraph()
    if "vc_head_id" not in st.session_state:
        st.session_state.vc_head_id = None


def _maybe_load_vc_from_url() -> None:
    """Import graph + apply focused snapshot from ?vc=… then strip param and rerun."""
    _init_version_graph()
    if "vc" not in st.query_params:
        return
    token = st.query_params["vc"]
    if isinstance(token, list):
        token = token[0] if token else ""
    token = str(token).strip()
    if not token:
        return
    try:
        bundle = decode_bundle(token)
        graph, focus = graph_from_bundle(bundle)
        st.session_state.vc_graph = graph
        st.session_state.vc_head_id = focus
        apply_payload_to_session(st.session_state, graph.nodes[focus].payload)
    except Exception as e:
        st.session_state["_vc_url_error"] = str(e)
    try:
        del st.query_params["vc"]
    except Exception:
        try:
            st.query_params.clear()
        except Exception:
            pass
    st.rerun()


def _augment_payload_with_uploads(
    payload: dict[str, Any],
    mort_mode: str,
    mort_upload: Any,
    lapse_mode: str,
    lapse_upload: Any,
) -> None:
    if mort_mode == "Upload CSV (merge)" and mort_upload is not None:
        payload["mort_csv_text"] = mort_upload.getvalue().decode("utf-8", errors="replace")
    if lapse_mode == "Upload CSV (merge)" and lapse_upload is not None:
        payload["lapse_csv_text"] = lapse_upload.getvalue().decode("utf-8", errors="replace")


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


def _render_version_control(
    *,
    mort_mode: str,
    mort_upload: Any,
    lapse_mode: str,
    lapse_upload: Any,
) -> None:
    _init_version_graph()
    graph: VersionGraph = st.session_state.vc_graph

    err_show = st.session_state.pop("_vc_url_error", None)
    if err_show:
        st.error(f"Could not load version link: {err_show}")

    with st.expander("Version control & sharing (this browser session only)", expanded=False):
        st.markdown(
            "Save **named snapshots** with notes, build a **DAG** of revisions, and **share** via URL or JSON. "
            "Another tab can open the link, edit assumptions, and save a **new child** snapshot."
        )

        if graph.nodes:
            sorted_ids = sorted(graph.nodes.keys(), key=lambda sid: graph.nodes[sid].created_at)
            try:
                st.graphviz_chart(graph.dot_graph(highlight_id=st.session_state.get("vc_head_id")))
            except Exception:
                st.caption("(Graph preview unavailable in this environment.)")
            rows = []
            for sid in sorted_ids:
                sn = graph.nodes[sid]
                rows.append(
                    {
                        "id": sid,
                        "parent": sn.parent_id or "—",
                        "name": sn.name,
                        "notes": sn.notes[:60] + ("…" if len(sn.notes) > 60 else ""),
                        "created": sn.created_at,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Save new snapshot")
        col_a, col_b = st.columns(2)
        with col_a:
            snap_name = st.text_input("Snapshot name", key="vc_snap_name", placeholder="e.g. Q4 base case")
        with col_b:
            snap_notes = st.text_input("Version notes", key="vc_snap_notes", placeholder="Like a commit message")

        parent_options = [("(root — no parent)", None)]
        for sid, sn in sorted(graph.nodes.items(), key=lambda kv: kv[1].created_at):
            parent_options.append((f"{sn.name} ({sid})", sid))
        default_ix = 0
        hid = st.session_state.get("vc_head_id")
        for i, (_, pid) in enumerate(parent_options):
            if pid == hid:
                default_ix = i
                break
        parent_label = st.selectbox(
            "Parent snapshot (new node branches from here)",
            options=[x[0] for x in parent_options],
            index=default_ix,
            key="vc_parent_pick",
        )
        parent_id = parent_options[[x[0] for x in parent_options].index(parent_label)][1]

        if st.button("Save snapshot from current UI", type="primary", key="vc_save_snap"):
            if not (snap_name or "").strip():
                st.warning("Enter a snapshot name.")
            else:
                payload = collect_payload_from_session(st.session_state)
                _augment_payload_with_uploads(payload, mort_mode, mort_upload, lapse_mode, lapse_upload)
                graph.add_snapshot(
                    parent_id=parent_id,
                    name=(snap_name or "").strip(),
                    notes=(snap_notes or "").strip(),
                    payload=payload,
                )
                st.session_state.vc_head_id = graph.head_id
                st.success(f"Saved snapshot **{graph.head_id}** — {snap_name.strip()}.")
                st.rerun()

        st.subheader("Load snapshot into UI")
        if not graph.nodes:
            st.caption("No snapshots yet.")
        else:
            sorted_ids = sorted(graph.nodes.keys(), key=lambda sid: graph.nodes[sid].created_at)
            load_labels = [f"{graph.nodes[sid].name} ({sid}) — {graph.nodes[sid].created_at}" for sid in sorted_ids]
            pick = st.selectbox(
                "Choose snapshot", options=range(len(sorted_ids)), format_func=lambda i: load_labels[i], key="vc_load_pick"
            )
            if st.button("Load selected into UI", key="vc_load_btn"):
                sid = sorted_ids[pick]
                apply_payload_to_session(st.session_state, graph.nodes[sid].payload)
                st.session_state.vc_head_id = sid
                st.success(f"Loaded **{sid}** into the form.")
                st.rerun()

        st.subheader("Share via link (localhost)")
        if graph.nodes:
            sorted_ids = sorted(graph.nodes.keys(), key=lambda sid: graph.nodes[sid].created_at)
            share_labels = [f"{graph.nodes[sid].name} ({sid})" for sid in sorted_ids]
            si = st.selectbox(
                "Snapshot to encode in link",
                options=range(len(sorted_ids)),
                format_func=lambda i: share_labels[i],
                key="vc_share_pick",
            )
            sid = sorted_ids[si]
            bundle = bundle_for_share(graph, sid)
            token = encode_bundle(bundle)
            if share_token_fits_url(token):
                rel = f"?vc={token}"
                st.code(rel, language=None)
                st.caption("Append the line above to your app URL (e.g. `http://127.0.0.1:8501` + line).")
            else:
                st.warning(
                    f"Encoded link is **{len(token)}** chars (limit {MAX_SHARE_URL_CHARS}). Use **JSON export** below and send the file, "
                    "or reduce policy CSV size."
                )
                st.text_area("Encoded token (for short payloads / manual copy)", token, height=120, key="vc_tok_big")

        st.subheader("Share via JSON file")
        if graph.nodes:
            sorted_ids = sorted(graph.nodes.keys(), key=lambda sid: graph.nodes[sid].created_at)
            json_labels = [f"{graph.nodes[sid].name} ({sid})" for sid in sorted_ids]
            si2 = st.selectbox(
                "Snapshot for JSON bundle",
                options=range(len(sorted_ids)),
                format_func=lambda i: json_labels[i],
                key="vc_json_pick",
            )
            share_id = sorted_ids[si2]
            jbundle = bundle_for_share(graph, share_id)
            jst = json.dumps(jbundle, indent=2)
            st.download_button(
                "Download JSON bundle",
                jst,
                file_name=f"snapshot-{share_id}.json",
                mime="application/json",
                key="vc_dl_json",
            )

        st.subheader("Import JSON bundle (replace graph & load focus)")
        pasted = st.text_area("Paste JSON bundle here", height=160, key="vc_paste_json")
        if st.button("Apply pasted JSON", key="vc_apply_json"):
            try:
                data = json.loads(pasted.strip() or "{}")
                if data.get("format") != FORMAT_TAG:
                    raise ValueError(f"Expected format {FORMAT_TAG!r}.")
                g2, focus = graph_from_bundle(data)
                st.session_state.vc_graph = g2
                st.session_state.vc_head_id = focus
                apply_payload_to_session(st.session_state, g2.nodes[focus].payload)
                st.success("Imported graph and loaded focus snapshot.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        up_json = st.file_uploader("Or upload JSON bundle", type=["json"], key="vc_up_json")
        if up_json is not None and st.button("Apply uploaded JSON", key="vc_apply_up_json"):
            try:
                data = json.loads(up_json.getvalue().decode("utf-8"))
                if data.get("format") != FORMAT_TAG:
                    raise ValueError(f"Expected format {FORMAT_TAG!r}.")
                g2, focus = graph_from_bundle(data)
                st.session_state.vc_graph = g2
                st.session_state.vc_head_id = focus
                apply_payload_to_session(st.session_state, g2.nodes[focus].payload)
                st.success("Imported graph and loaded focus snapshot.")
                st.rerun()
            except Exception as e:
                st.error(str(e))


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


def _step_labels() -> dict[int, str]:
    return {
        1: "1 · Prepare data",
        2: "2 · Expected decrements",
        3: "3 · A/E ratios",
        4: "4 · Credibility",
        5: "5 · Update parameters",
        6: "6 · Impact analysis",
    }


def _wf_go_next() -> None:
    s = min(6, int(st.session_state.get("wf_step", 1)) + 1)
    st.session_state.wf_step = s
    st.session_state.wf_step_radio = s


def _wf_go_prev() -> None:
    s = max(1, int(st.session_state.get("wf_step", 1)) - 1)
    st.session_state.wf_step = s
    st.session_state.wf_step_radio = s


def _step_radio() -> int:
    """Workflow step selector. Uses wf_step + wf_step_radio; do not assign to workflow_step (widget-owned)."""
    labels = _step_labels()
    options = list(range(1, 7))
    ws = max(1, min(6, int(st.session_state.get("wf_step", 1))))
    step = st.radio(
        "Workflow step",
        options=options,
        format_func=lambda i: labels[i],
        horizontal=True,
        index=ws - 1,
        key="wf_step_radio",
        label_visibility="collapsed",
    )
    st.session_state.wf_step = int(step)
    return int(step)


def main() -> None:
    st.set_page_config(page_title="Term-life experience workflow", layout="wide")
    _init_workflow_session()
    _init_version_graph()
    _maybe_load_vc_from_url()
    st.title("Term-life model · experience workflow")
    st.caption(
        "Guided Steps 1–6: from policy-level experience to credibility-weighted parameters and impact on projections."
    )

    base_mort = stylized_base_mortality_table()
    default_term = 20

    with st.sidebar:
        st.header("Projection model")
        n0 = st.number_input("Initial active policies (N₀)", min_value=1, value=100_000, step=1_000, key="n0")
        issue_age = st.number_input("Issue age", min_value=18, max_value=95, value=40, key="issue_age")
        term_years = st.number_input("Term (years)", min_value=1, max_value=40, value=default_term, key="term_years")
        sum_assured = st.number_input("Sum assured (per policy)", min_value=0.0, value=100_000.0, step=1_000.0, key="sa")
        annual_premium = st.number_input("Annual premium (per policy)", min_value=0.0, value=500.0, step=10.0, key="prem")
        annual_expense = st.number_input("Annual expense (per policy)", min_value=0.0, value=50.0, step=5.0, key="exp")
        discount_rate = st.slider("Discount rate r", 0.0, 0.15, 0.03, 0.005, format="%.3f", key="disc")
        k_mort = st.slider("k_mort (mortality on table)", 0.5, 1.5, 0.95, 0.01, key="k_mort")
        k_lapse = st.slider("k_lapse (on base lapse curve)", 0.5, 1.5, 1.0, 0.01, key="k_lapse")

        st.subheader("Base assumptions (CSV optional)")
        mort_mode = st.radio("Mortality table", ("Built-in stylized", "Upload CSV (merge)"), key="mort_mode")
        mort_upload = st.file_uploader("Mortality CSV", type=["csv"], key="mort_csv")
        lapse_mode = st.radio("Lapse curve", ("Built-in stylized", "Upload CSV (merge)"), key="lapse_mode")
        lapse_upload = st.file_uploader("Lapse CSV", type=["csv"], key="lapse_csv")

        with st.expander("CSV column help"):
            st.markdown(
                """
**Policy-level (Step 1):** `policy_id`, `issue_age`, `gender`, `smoker`, `sum_assured`, `policy_year`, `exposure_years`, `died`, `lapsed` (0/1; at most one event per row).

**Mortality table:** `age`, `q` · **Lapse curve:** `policy_year`, `lapse`

**Legacy aggregate experience (optional):** age / policy_year exposure files — use Step 1 policy file for the full workflow.
                """
            )

        st.subheader("Optional: aggregate experience only")
        mort_exp_upload = st.file_uploader("Mortality experience (aggregate)", type=["csv"], key="mort_exp")
        lapse_exp_upload = st.file_uploader("Lapse experience (aggregate)", type=["csv"], key="lapse_exp")
        cred_w = st.slider("Legacy: credibility w for suggest_k_mort", 0.0, 1.0, 1.0, 0.05, key="cred_w")

    try:
        if mort_mode == "Built-in stylized":
            st.session_state.pop("vc_mort_csv_text", None)
        if lapse_mode == "Built-in stylized":
            st.session_state.pop("vc_lapse_csv_text", None)

        if mort_mode == "Upload CSV (merge)" and mort_upload is not None:
            mortality_table = mortality_table_from_csv(pd.read_csv(mort_upload), merge_with_stylized=True)
            st.session_state.pop("vc_mort_csv_text", None)
        elif st.session_state.get("vc_mort_csv_text"):
            mortality_table = mortality_table_from_csv(
                pd.read_csv(io.StringIO(st.session_state.vc_mort_csv_text)), merge_with_stylized=True
            )
        else:
            mortality_table = base_mort

        if lapse_mode == "Upload CSV (merge)" and lapse_upload is not None:
            lapse_curve = lapse_curve_from_csv(pd.read_csv(lapse_upload), int(term_years), merge_with_stylized=True)
            st.session_state.pop("vc_lapse_csv_text", None)
        elif st.session_state.get("vc_lapse_csv_text"):
            lapse_curve = lapse_curve_from_csv(
                pd.read_csv(io.StringIO(st.session_state.vc_lapse_csv_text)), int(term_years), merge_with_stylized=True
            )
        else:
            lapse_curve = stylized_base_lapse_by_duration(int(term_years))
    except Exception as e:
        st.error(f"Could not load base assumptions: {e}")
        st.stop()

    _render_version_control(
        mort_mode=mort_mode,
        mort_upload=mort_upload,
        lapse_mode=lapse_mode,
        lapse_upload=lapse_upload,
    )

    max_age = int(issue_age) + int(term_years) - 1
    missing_ages = [x for x in range(int(issue_age), max_age + 1) if x not in mortality_table.rates]
    if missing_ages:
        st.error(
            f"Mortality table missing ages for projection: {missing_ages[:8]}{'…' if len(missing_ages) > 8 else ''}. "
            "Use built-in stylized or upload rates for the full issue-age window."
        )
        st.stop()

    lapse_curve_base = lapse_curve  # expected decrements vs this base (no k_lapse)

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

    st.subheader("Quick projection (current sliders)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PV premiums", f"{proj.pv_premiums:,.0f}")
    m2.metric("PV death benefits", f"{proj.pv_death_benefits:,.0f}")
    m3.metric("PV expenses", f"{proj.pv_expenses:,.0f}")
    m4.metric("PV profit", f"{proj.pv_profit:,.0f}")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_fig_decrements(policy_years, proj), use_container_width=True, key="quick_decrements")
    with c2:
        st.plotly_chart(_fig_cashflows(policy_years, proj), use_container_width=True, key="quick_cashflows")

    st.divider()
    st.header("Actuarial workflow · Steps 1–6")
    st.caption("Use the step control below; each screen matches one step in the experience-to-assumption process.")
    step = _step_radio()
    st.progress(step / 6.0)
    st.caption(f"Step {step} of 6 — {_step_labels()[step]}")

    policy_df = st.session_state["policy_exp_df"]

    # --- Step 1 ---
    if step == 1:
        st.markdown(STEP_GUIDE[1])
        b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
        with b1:
            if st.button("Load bundled sample", help=str(BUNDLED_POLICY_CSV), key="btn_bundle"):
                if BUNDLED_POLICY_CSV.is_file():
                    st.session_state["policy_exp_df"] = pd.read_csv(BUNDLED_POLICY_CSV)
                    st.rerun()
                else:
                    st.error("Bundled CSV not found on disk.")
        with b2:
            if st.button("Generate fresh demo", key="btn_gen"):
                st.session_state["policy_exp_df"] = generate_policy_level_term_experience(
                    n_policies=3000,
                    issue_age=40,
                    term_years=20,
                    study_policy_years=5,
                    true_k_mort=1.07,
                    true_k_lapse=0.93,
                    mortality_table=stylized_base_mortality_table(),
                    lapse_curve=stylized_base_lapse_by_duration(20),
                    seed=42,
                )
                st.rerun()
        with b3:
            if st.button("Clear experience", key="btn_clear_exp"):
                st.session_state["policy_exp_df"] = None
                st.session_state["exp_k_mort"] = None
                st.session_state["exp_k_lapse"] = None
                st.rerun()
        with b4:
            st.download_button(
                "Download policy CSV template (small demo)",
                demo_policy_csv_text(),
                "policy_level_demo.csv",
                "text/csv",
                key="dl_demo_pol",
            )

        up = st.file_uploader("Upload policy-level experience CSV", type=["csv"], key="pol_up")
        if up is not None:
            try:
                st.session_state["policy_exp_df"] = parse_policy_level_experience(pd.read_csv(up))
                st.success("Loaded and validated policy-level file.")
            except Exception as e:
                st.error(str(e))

        if policy_df is None:
            st.info("Load the bundled sample, generate a demo, or upload a CSV to enable Steps 2–6.")
        else:
            st.success(
                f"{len(policy_df):,} rows · {policy_df['policy_id'].nunique():,} policies · "
                f"sum(deaths)={int(policy_df['died'].sum())} · sum(lapses)={int(policy_df['lapsed'].sum())}"
            )
            st.dataframe(policy_df.head(50), use_container_width=True, hide_index=True)

    # Rollups for steps 2–5 (need policy data)
    mort_roll: pd.DataFrame | None = None
    lapse_roll: pd.DataFrame | None = None
    mort_exp_tbl: pd.DataFrame | None = None
    lapse_exp_tbl: pd.DataFrame | None = None
    if policy_df is not None:
        try:
            mort_roll = rollup_mortality_by_age(policy_df)
            lapse_roll = rollup_lapse_by_policy_year(policy_df)
            mort_exp_tbl, lapse_exp_tbl = expected_decrements_from_rollups(
                mort_roll, lapse_roll, mortality_table, lapse_curve_base
            )
        except Exception as e:
            st.warning(f"Rollup / expected decrement issue: {e}")

    # --- Step 2 ---
    if step == 2:
        st.markdown(STEP_GUIDE[2])
        if mort_exp_tbl is None or lapse_exp_tbl is None:
            st.warning("Complete Step 1 first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Mortality — by attained age (start of policy year)")
                st.dataframe(mort_exp_tbl, use_container_width=True, hide_index=True)
            with c2:
                st.caption("Lapse — by policy year (lapse exposure = exposure × (1 − died))")
                st.dataframe(lapse_exp_tbl, use_container_width=True, hide_index=True)
            tde = float(mort_exp_tbl["expected_deaths"].sum())
            tle = float(lapse_exp_tbl["expected_lapses"].sum())
            st.metric("Total expected deaths (base table)", f"{tde:,.2f}")
            st.metric("Total expected lapses (base curve)", f"{tle:,.2f}")

    # --- Step 3 ---
    ae_mort_df: pd.DataFrame | None = None
    ae_lapse_df: pd.DataFrame | None = None
    agg_ae_mort: float | None = None
    agg_ae_lapse: float | None = None
    if mort_exp_tbl is not None and lapse_exp_tbl is not None:
        ae_mort_df = mort_exp_tbl.copy()
        ae_mort_df["A/E"] = ae_mort_df.apply(
            lambda r: mortality_ae_ratio(float(r["observed_deaths"]), float(r["exposure_years"]), int(r["attained_age"]), mortality_table),
            axis=1,
        )
        ae_lapse_df = lapse_exp_tbl.copy()
        ae_lapse_df["A/E"] = ae_lapse_df.apply(
            lambda r: (float(r["observed_lapses"]) / float(r["expected_lapses"])) if float(r["expected_lapses"]) > 0 else None,
            axis=1,
        )
        td_obs = float(ae_mort_df["observed_deaths"].sum())
        tde = float(ae_mort_df["expected_deaths"].sum())
        agg_ae_mort = td_obs / tde if tde > 0 else None
        tl_obs = float(ae_lapse_df["observed_lapses"].sum())
        tle = float(ae_lapse_df["expected_lapses"].sum())
        agg_ae_lapse = tl_obs / tle if tle > 0 else None

    if step == 3:
        st.markdown(STEP_GUIDE[3])
        if ae_mort_df is None or ae_lapse_df is None:
            st.warning("Complete Steps 1–2 first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(ae_mort_df, use_container_width=True, hide_index=True)
                fig = go.Figure()
                plot_m = ae_mort_df.dropna(subset=["A/E"])
                if not plot_m.empty:
                    fig.add_trace(go.Bar(x=plot_m["attained_age"], y=plot_m["A/E"], name="Mortality A/E"))
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig.update_layout(title="Mortality A/E by attained age", height=360, yaxis_title="A/E")
                st.plotly_chart(fig, use_container_width=True, key="step3_mort_ae_chart")
            with c2:
                st.dataframe(ae_lapse_df, use_container_width=True, hide_index=True)
                fig2 = go.Figure()
                plot_l = ae_lapse_df.dropna(subset=["A/E"])
                if not plot_l.empty:
                    fig2.add_trace(go.Scatter(x=plot_l["policy_year"], y=plot_l["A/E"], mode="lines+markers", name="Lapse A/E"))
                fig2.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig2.update_layout(title="Lapse A/E by policy year", height=360, yaxis_title="A/E")
                st.plotly_chart(fig2, use_container_width=True, key="step3_lapse_ae_chart")
            c3, c4 = st.columns(2)
            if agg_ae_mort is not None:
                c3.metric("Aggregate mortality A/E", f"{agg_ae_mort:.4f}")
            if agg_ae_lapse is not None:
                c4.metric("Aggregate lapse A/E", f"{agg_ae_lapse:.4f}")

    # --- Step 4 ---
    if step == 4:
        st.markdown(STEP_GUIDE[4])
        if agg_ae_mort is None or agg_ae_lapse is None or mort_roll is None or lapse_roll is None:
            st.warning("Complete Steps 1–3 first.")
        else:
            n_full_d = st.number_input(
                "Full credibility deaths threshold (n)",
                min_value=1.0,
                value=float(st.session_state["n_full_deaths"]),
                step=1.0,
                key="nfd",
            )
            n_full_l = st.number_input(
                "Full credibility lapses threshold (n)",
                min_value=1.0,
                value=float(st.session_state["n_full_lapses"]),
                step=1.0,
                key="nfl",
            )
            st.session_state["n_full_deaths"] = float(n_full_d)
            st.session_state["n_full_lapses"] = float(n_full_l)

            total_deaths = float(mort_roll["observed_deaths"].sum())
            total_lapses = float(lapse_roll["observed_lapses"].sum())
            nfd = float(st.session_state["n_full_deaths"])
            nfl = float(st.session_state["n_full_lapses"])
            z_d_disp = limited_fluctuation_z(total_deaths, nfd)
            z_l_disp = limited_fluctuation_z(total_lapses, nfl)
            blend_m_disp = credibility_weighted_ae(float(agg_ae_mort), z_d_disp, 1.0)
            blend_l_disp = credibility_weighted_ae(float(agg_ae_lapse), z_l_disp, 1.0)

            st.metric("Total observed deaths", f"{int(total_deaths):,}")
            st.metric("Mortality credibility Z", f"{z_d_disp:.4f}")
            st.metric("Credibility-weighted mortality A/E", f"{blend_m_disp:.4f}")

            st.metric("Total observed lapses", f"{int(total_lapses):,}")
            st.metric("Lapse credibility Z", f"{z_l_disp:.4f}")
            st.metric("Credibility-weighted lapse A/E", f"{blend_l_disp:.4f}")

    z_d = z_l = blend_m = blend_l = None
    if mort_roll is not None and lapse_roll is not None and agg_ae_mort is not None and agg_ae_lapse is not None:
        total_deaths = float(mort_roll["observed_deaths"].sum())
        total_lapses = float(lapse_roll["observed_lapses"].sum())
        nfd = float(st.session_state.get("n_full_deaths", 1082.0))
        nfl = float(st.session_state.get("n_full_lapses", 1082.0))
        z_d = limited_fluctuation_z(total_deaths, nfd)
        z_l = limited_fluctuation_z(total_lapses, nfl)
        blend_m = credibility_weighted_ae(float(agg_ae_mort), z_d, 1.0)
        blend_l = credibility_weighted_ae(float(agg_ae_lapse), z_l, 1.0)
        st.session_state["_blend_mort"] = blend_m
        st.session_state["_blend_lapse"] = blend_l

    # --- Step 5 ---
    if step == 5:
        st.markdown(STEP_GUIDE[5])
        if blend_m is None or blend_l is None:
            st.warning("Load policy-level experience in Step 1 and complete Steps 2–3 first.")
        else:
            prop_k_mort = float(blend_m)
            prop_k_lapse = float(k_lapse) * float(blend_l)
            st.metric("Proposed k_mort (replace slider)", f"{prop_k_mort:.4f}")
            st.metric("Proposed k_lapse (multiply current k by blended lapse A/E)", f"{prop_k_lapse:.4f}")
            st.caption(
                "Mortality: blended A/E is applied as the new table multiplier k_mort. "
                "Lapse: current k_lapse is scaled by the blended lapse A/E vs the **base** curve."
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Apply proposed parameters", type="primary", key="apply_prop"):
                    st.session_state["exp_k_mort"] = prop_k_mort
                    st.session_state["exp_k_lapse"] = prop_k_lapse
                    st.success("Stored for Step 6 and sidebar comparison. Copy to sliders if you want the main projection above to match.")
            with b2:
                if st.button("Reset stored experience parameters", key="reset_prop"):
                    st.session_state["exp_k_mort"] = None
                    st.session_state["exp_k_lapse"] = None
                    st.rerun()

            if st.session_state.get("exp_k_mort") is not None:
                st.info(
                    f"Stored experience-informed: k_mort={st.session_state['exp_k_mort']:.4f}, "
                    f"k_lapse={st.session_state['exp_k_lapse']:.4f}"
                )

    # --- Step 6 ---
    if step == 6:
        st.markdown(STEP_GUIDE[6])
        km_exp = st.session_state.get("exp_k_mort")
        kl_exp = st.session_state.get("exp_k_lapse")
        assump_curr = assumptions
        if km_exp is None or kl_exp is None:
            st.info(
                "Go to **Step 5** and click **Apply proposed parameters** to store an experience-informed "
                "k_mort and k_lapse, then return here to compare impact on PV profit and cash flows."
            )
            st.metric("PV profit (current sliders)", f"{proj.pv_profit:,.0f}")
            st.plotly_chart(_fig_cashflows(policy_years, proj), use_container_width=True, key="step6_placeholder_cf")
        else:
            assump_exp = Assumptions(
                term_years=int(term_years),
                mortality_table=mortality_table,
                lapse_curve=lapse_curve,
                k_mort=float(km_exp),
                k_lapse=float(kl_exp),
                discount_rate=float(discount_rate),
            )
            proj_curr = proj
            proj_exp = project_term_life(portfolio, assump_exp)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### Current sliders")
                st.metric("k_mort", f"{assump_curr.k_mort:.4f}")
                st.metric("k_lapse", f"{assump_curr.k_lapse:.4f}")
                st.metric("PV profit", f"{proj_curr.pv_profit:,.0f}")
                st.plotly_chart(
                    _fig_cashflows(policy_years, proj_curr), use_container_width=True, key="step6_current_cf"
                )
            with col_b:
                st.markdown("##### After experience update")
                st.metric("k_mort", f"{assump_exp.k_mort:.4f}")
                st.metric("k_lapse", f"{assump_exp.k_lapse:.4f}")
                st.metric(
                    "PV profit",
                    f"{proj_exp.pv_profit:,.0f}",
                    delta=f"{proj_exp.pv_profit - proj_curr.pv_profit:,.0f}",
                )
                st.plotly_chart(
                    _fig_cashflows(policy_years, proj_exp), use_container_width=True, key="step6_experience_cf"
                )

            st.plotly_chart(
                _fig_effective_rates(int(issue_age), int(term_years), assump_exp),
                use_container_width=True,
                key="step6_effective_rates",
            )

    st.divider()
    nprev, nnext, _ = st.columns([1, 1, 6])
    with nprev:
        st.button("← Previous step", key="nav_prev", on_click=_wf_go_prev, disabled=step <= 1)
    with nnext:
        st.button("Next step →", key="nav_next", on_click=_wf_go_next, disabled=step >= 6)

    # --- Legacy aggregate uploads (sidebar) ---
    has_mort_exp = mort_exp_upload is not None
    has_lapse_exp = lapse_exp_upload is not None
    if has_mort_exp or has_lapse_exp:
        st.divider()
        st.subheader("Legacy: aggregate experience uploads")
        ec1, ec2 = st.columns(2)
        if has_mort_exp:
            with ec1:
                try:
                    from term_life.csv_loaders import parse_mortality_experience

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
                        st.metric("Aggregate A/E", f"{agg_ae:.4f}")
                        st.metric("Suggested k_mort (legacy toy)", f"{k_sugg:.4f}")
                except Exception as e:
                    st.warning(f"Mortality experience: {e}")
        if has_lapse_exp:
            with ec2:
                try:
                    from term_life.csv_loaders import parse_lapse_experience

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
                    model_rates: list[float] = []
                    for lap_row in summary:
                        try:
                            base = lapse_curve.rate(lap_row.policy_year)
                            model_rates.append(min(1.0, float(k_lapse) * base))
                        except Exception:
                            model_rates.append(float("nan"))
                    summ_df["model_lapse_effective"] = model_rates
                    st.line_chart(summ_df.set_index("policy_year")[["crude_rate", "model_lapse_effective"]])
                except Exception as e:
                    st.warning(f"Lapse experience: {e}")


if __name__ == "__main__":
    main()
