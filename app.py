"""EffortAI - Software Effort Prediction & Delivery Planning System.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Deployment safety: always resolve imports and files relative to this file,
# no matter which directory the host launches Streamlit from.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st

import effort_analytics as an
from effort_model import ModelService, engineer
from effort_reporting import build_html_report
from effort_schema import BASE_FEATURES, BASE_KEYS, FEATURE_BY_KEY, PRESETS, clamp, validate
from effort_theme import PALETTES, get_css

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="EffortAI | Software Effort Intelligence",
                   page_icon="◆", layout="wide", initial_sidebar_state="expanded")

# ==========================================================================
# STATE
# ==========================================================================
st.session_state.setdefault("theme", "light")
st.session_state.setdefault("history", [])
for f in BASE_FEATURES:
    st.session_state.setdefault(f.key, f.default)

st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)
P = PALETTES[st.session_state.theme]


@st.cache_resource(show_spinner="Loading model...")
def get_service(path: str) -> ModelService:
    p = Path(path)
    if not p.is_absolute():
        p = APP_DIR / p
    if not p.exists():
        raise FileNotFoundError(
            f"'{p.name}' was not found in {p.parent}. Place your trained model file "
            "next to app.py (or in the models/ folder) and enter its name in the sidebar."
        )
    return ModelService.from_path(str(p))


# ==========================================================================
# UI HELPERS
# ==========================================================================
def kpi(label, value, sub="", delay=0.0, sheen=False):
    cls = "kpi-value sheen" if sheen else "kpi-value"
    st.markdown(
        f"<div class='glass' style='animation-delay:{delay}s'>"
        f"<div class='kpi-label'>{label}</div><div class='{cls}'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def styled(fig, height=380, title=None):
    fig.update_layout(
        height=height, title=title, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["plot"],
        font=dict(family="Inter", color=P["ink"], size=13),
        transition=dict(duration=650, easing="cubic-in-out"),
        legend=dict(orientation="h", y=-0.18),
        xaxis=dict(gridcolor=P["g200"]), yaxis=dict(gridcolor=P["g200"]),
    )
    return fig


def current_inputs() -> dict:
    return {k: int(st.session_state[k]) for k in BASE_KEYS}


# ==========================================================================
# SIDEBAR
# ==========================================================================
with st.sidebar:
    st.markdown("## ◆ EffortAI")
    st.caption("Estimation, risk and delivery planning")

    icon = "Dark mode" if st.session_state.theme == "light" else "Light mode"
    if st.button(f"Switch to {icon}", width="stretch"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

    st.divider()
    model_path = st.text_input("Model file", "model.pkl",
                               help="Path relative to app.py, e.g. model.pkl or models/model.pkl")
    st.markdown("### Presets")
    for name, vals in PRESETS.items():
        if st.button(name, width="stretch", key=f"p_{name}"):
            for k, v in vals.items():
                st.session_state[k] = v
            st.toast(f"Loaded: {name}")
            st.rerun()

    st.divider()
    st.markdown("### Planning assumptions")
    rate = st.number_input("Blended hourly rate (USD)", 1, 1000, 25)
    team_size = st.slider("Team size", 1, 40, 5)
    hours_month = st.slider("Hours / person / month", 100, 200, 160)
    confidence = st.select_slider("Interval confidence", [0.80, 0.90], value=0.80,
                                  format_func=lambda v: f"{v:.0%}")
    comm_penalty = st.slider("Communication overhead per pair", 0.0, 0.06, 0.02, 0.005,
                             help="Brooks' law coefficient: how much each pair of people costs in coordination.")
    budget = st.number_input("Budget ceiling (hours, 0 = none)", 0, 200000, 0, 500)

    st.divider()
    project_name = st.text_input("Project name", "Untitled project")
    author = st.text_input("Prepared by", "Estimation team")
    st.caption(f"{len(st.session_state.history)} prediction(s) this session")
    if st.session_state.history and st.button("Clear history", width="stretch"):
        st.session_state.history = []
        st.rerun()

try:
    svc = get_service(model_path)
except Exception as exc:
    st.error(f"Could not load `{model_path}`: {exc}")
    st.info("Common causes: the file is missing from the deployed repo, or scikit-learn in "
            "requirements.txt is a different version from the one used to train the model.")
    st.stop()

# ==========================================================================
# HERO
# ==========================================================================
st.markdown(
    """<div class="hero">
    <span class="badge">Uncertainty quantification</span>
    <span class="badge">Shapley attribution</span>
    <span class="badge">Monte Carlo risk</span>
    <span class="badge">Delivery planning</span>
    <h1>Software Effort Intelligence</h1>
    <p>More than a number. EffortAI turns a machine-learning effort prediction into a
    defensible estimate: a confidence interval, an additive explanation of what drives it,
    a simulated risk distribution, a staffed delivery plan and a client-ready report.</p>
    </div>""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Estimate", "Risk & Simulation", "Explainability", "Delivery Plan",
                "Portfolio", "Model Card", "History"])

# ==========================================================================
# 1. ESTIMATE
# ==========================================================================
with tabs[0]:
    st.markdown("### Project inputs")
    cols = st.columns(3)
    for i, f in enumerate(BASE_FEATURES):
        with cols[i % 3]:
            st.session_state[f.key] = clamp(f.key, st.session_state[f.key])
            if f.key == "Language":
                st.selectbox(f.label, [1, 2, 3], key=f.key, help=f.help)
            else:
                st.number_input(f.label, min_value=f.lo, max_value=f.hi, step=1,
                                key=f.key, help=f.help)

    values = current_inputs()
    for severity, msg in validate(values):
        getattr(st, severity)(msg)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        run = st.button("Run estimate", width="stretch")

    if run:
        with st.spinner("Predicting, quantifying uncertainty and simulating risk..."):
            pred = svc.predict_with_interval(values, confidence)
            samples = an.monte_carlo(svc, values)
            st.session_state.result = dict(pred=pred, values=values, samples=samples)
        st.session_state.history.append(
            dict(Time=datetime.now().strftime("%H:%M:%S"), Project=project_name,
                 Effort=round(pred.point), Low=round(pred.low), High=round(pred.high), **values)
        )
        st.balloons()

    res = st.session_state.get("result")
    if res:
        pred, values, samples = res["pred"], res["values"], res["samples"]
        sched = an.schedule(pred.point, team_size, hours_month, comm_penalty)
        pct = an.percentiles(samples)
        size_name, size_color = an.classify_size(pred.point)

        st.divider()
        st.markdown(
            f"### Estimate &nbsp;<span class='tag' style='background:{size_color}'>{size_name}</span>",
            unsafe_allow_html=True,
        )
        c = st.columns(4)
        with c[0]:
            kpi("Expected effort", f"{pred.point:,.0f} h", "point estimate", 0.0, sheen=True)
        with c[1]:
            kpi(f"{confidence:.0%} interval", f"{pred.low:,.0f} - {pred.high:,.0f}",
                pred.method, 0.08)
        with c[2]:
            kpi("Commit figure (P80)", f"{pct['P80']:,.0f} h",
                "80% of simulations finish under this", 0.16)
        with c[3]:
            kpi("Budget at P80", f"${pct['P80'] * rate:,.0f}", f"at ${rate}/hour", 0.24)

        st.write("")
        c = st.columns(4)
        with c[0]:
            kpi("Person-months", f"{sched['person_months']:,.1f}", f"{hours_month} h/month", 0.0)
        with c[1]:
            kpi("Duration", f"{sched['duration_months']:,.1f} mo",
                f"team of {team_size}, {sched['efficiency']:.0%} efficient", 0.08)
        with c[2]:
            kpi("Delivery per point", f"{pred.point / max(values['PointsAjust'], 1):,.1f} h/FP",
                "hours per adjusted function point", 0.16)
        with c[3]:
            if budget:
                prob = an.probability_within(samples, budget)
                kpi("Chance of meeting budget", f"{prob:.0%}", f"budget {budget:,} h", 0.24)
            else:
                kpi("Estimate spread", f"{pred.relative_spread:.0%}",
                    "interval width vs point estimate", 0.24)

        if not sched["feasible"]:
            st.warning(
                f"This plan compresses the schedule to {sched['compression']:.0%} of the nominal "
                f"{sched['nominal_months']:.1f}-month duration. Schedules below about 75% of nominal "
                "usually cost more effort than they save."
            )

        if HAS_PLOTLY:
            left, right = st.columns([1.15, 1])
            with left:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[pred.point], y=["Effort"], orientation="h", width=[.45],
                    marker_color=P["g500"], name="Point estimate",
                    error_x=dict(type="data", symmetric=False,
                                 array=[pred.high - pred.point], arrayminus=[pred.point - pred.low],
                                 color=P["g700"], thickness=3, width=12),
                ))
                fig.add_vline(x=pct["P80"], line_dash="dot", line_color=P["g700"],
                              annotation_text="P80 commit")
                fig.update_layout(xaxis_title="Person-hours", showlegend=False)
                st.plotly_chart(styled(fig, 240, "Estimate with confidence interval"),
                                width="stretch")
            with right:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=pred.point,
                    delta=dict(reference=pct["P80"], increasing=dict(color=P["g600"])),
                    number=dict(suffix=" h", font=dict(size=34, color=P["g700"])),
                    gauge=dict(
                        axis=dict(range=[0, max(pct["P95"] * 1.25, 1000)]),
                        bar=dict(color=P["g500"], thickness=.3),
                        threshold=dict(line=dict(color=P["g700"], width=4), value=pct["P80"]),
                        steps=[dict(range=[0, pct["P50"]], color=P["g100"]),
                               dict(range=[pct["P50"], pct["P80"]], color=P["g200"]),
                               dict(range=[pct["P80"], max(pct["P95"] * 1.25, 1001)], color=P["g300"])],
                    )))
                st.plotly_chart(styled(fig, 240, "Point estimate vs P80"), width="stretch")

        st.markdown("### Engineered feature vector sent to the model")
        st.dataframe(engineer(pd.DataFrame([values])), width="stretch")

# ==========================================================================
# 2. RISK & SIMULATION
# ==========================================================================
with tabs[1]:
    res = st.session_state.get("result")
    if not res:
        st.info("Run an estimate first - the simulation is built from those inputs.")
    else:
        pred, values, samples = res["pred"], res["values"], res["samples"]
        sched = an.schedule(pred.point, team_size, hours_month, comm_penalty)
        pct = an.percentiles(samples)

        st.markdown("### Monte Carlo outcome distribution")
        st.caption(
            "1,500 simulated projects. Scope inputs are perturbed because estimates of size are "
            "themselves estimates, then a lognormal model-error term is applied - which is why the "
            "tail leans right, exactly like real overruns."
        )
        c = st.columns(5)
        for i, (k, v) in enumerate(pct.items()):
            with c[i]:
                kpi(k, f"{v:,.0f} h", f"${v * rate:,.0f}", i * 0.06)

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=samples, nbinsx=55, marker_color=P["g400"],
                                       opacity=.85, name="Simulations"))
            for label, key, dash in (("P50", "P50", "dot"), ("P80", "P80", "dash"), ("P95", "P95", "longdash")):
                fig.add_vline(x=pct[key], line_dash=dash, line_color=P["g700"],
                              annotation_text=label)
            if budget:
                fig.add_vline(x=budget, line_color="#d1495b", annotation_text="budget")
            fig.update_layout(xaxis_title="Effort (person-hours)", yaxis_title="Simulations",
                              showlegend=False)
            st.plotly_chart(styled(fig, 400, "Where the project is likely to land"),
                            width="stretch")

            srt = np.sort(samples)
            cdf = np.arange(1, len(srt) + 1) / len(srt)
            fig = go.Figure(go.Scatter(x=srt, y=cdf * 100, mode="lines",
                                       line=dict(color=P["g600"], width=3),
                                       fill="tozeroy", fillcolor=f"rgba(95,201,138,.22)"))
            fig.update_layout(xaxis_title="Effort (hours)", yaxis_title="Probability of finishing under (%)")
            st.plotly_chart(styled(fig, 340, "Confidence curve - pick your commitment level"),
                            width="stretch")

        if budget:
            prob = an.probability_within(samples, budget)
            st.metric("Probability of delivering within the budget ceiling", f"{prob:.1%}")

        st.markdown("### Risk register")
        risks = an.risk_profile(values, pred, sched)
        score, level, color = an.overall_risk(risks)
        st.markdown(
            f"Overall risk: <span class='tag' style='background:{color}'>{level} - {score:.0f}/100</span>",
            unsafe_allow_html=True,
        )
        if HAS_PLOTLY:
            fig = go.Figure(go.Scatterpolar(
                r=list(risks["Score"]) + [risks["Score"].iloc[0]],
                theta=list(risks["Factor"]) + [risks["Factor"].iloc[0]],
                fill="toself", line=dict(color=P["g600"], width=3),
                fillcolor="rgba(95,201,138,.32)"))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor=P["g200"]),
                                         bgcolor=P["plot"]))
            st.plotly_chart(styled(fig, 420, "Risk radar"), width="stretch")
        st.dataframe(risks.round(1), width="stretch")

# ==========================================================================
# 3. EXPLAINABILITY
# ==========================================================================
with tabs[2]:
    res = st.session_state.get("result")
    if not res:
        st.info("Run an estimate first.")
    else:
        values = res["values"]
        pred = res["pred"]
        st.markdown("### Why the model produced this number")
        st.caption(
            "Monte-Carlo Shapley attribution. Every feature is switched on in random order, "
            "starting from a neutral mid-range project, and its contribution is the average jump "
            "it causes. The bars therefore add up exactly to the gap between the neutral baseline "
            "and this project's estimate."
        )
        n_perm = st.slider("Permutations (higher = more stable, slower)", 20, 200, 80, 20)
        with st.spinner("Attributing contributions..."):
            shap = svc.shapley(values, n_permutations=n_perm)
            base_pred = svc.baseline_prediction()

        c = st.columns(3)
        with c[0]:
            kpi("Neutral baseline", f"{base_pred:,.0f} h", "mid-range project", 0.0)
        with c[1]:
            kpi("Sum of contributions", f"{shap['Contribution'].sum():+,.0f} h",
                "adds up to the estimate", 0.08)
        with c[2]:
            kpi("This project", f"{pred.point:,.0f} h", "point estimate", 0.16)

        if HAS_PLOTLY:
            d = shap.sort_values("Contribution")
            fig = go.Figure(go.Bar(
                x=d["Contribution"], y=d["Feature"], orientation="h",
                marker_color=[P["g600"] if v >= 0 else P["g300"] for v in d["Contribution"]],
                text=[f"{v:+,.0f} h" for v in d["Contribution"]], textposition="outside"))
            fig.update_layout(xaxis_title="Contribution to effort (hours)")
            st.plotly_chart(styled(fig, 430, "Feature contributions"), width="stretch")

            wf = shap.sort_values("Contribution", key=np.abs, ascending=False)
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute"] + ["relative"] * len(wf) + ["total"],
                x=["Baseline"] + list(wf["Feature"]) + ["Estimate"],
                y=[base_pred] + list(wf["Contribution"]) + [0],
                increasing=dict(marker=dict(color=P["g600"])),
                decreasing=dict(marker=dict(color=P["g300"])),
                totals=dict(marker=dict(color=P["g500"]))))
            fig.update_layout(yaxis_title="Effort (hours)", xaxis_tickangle=-35)
            st.plotly_chart(styled(fig, 460, "From a neutral project to yours"),
                            width="stretch")

        st.dataframe(shap.round(1), width="stretch")

        st.divider()
        st.markdown("### Sensitivity explorer")
        c1, c2 = st.columns(2)
        with c1:
            sweep_key = st.selectbox("Sweep this feature", BASE_KEYS,
                                     format_func=lambda k: FEATURE_BY_KEY[k].label)
            sw = svc.sweep(values, sweep_key)
            if HAS_PLOTLY:
                fig = go.Figure(go.Scatter(
                    x=sw[sweep_key], y=sw["Effort"], mode="lines",
                    line=dict(color=P["g600"], width=4, shape="spline"),
                    fill="tozeroy", fillcolor="rgba(95,201,138,.22)"))
                fig.add_vline(x=values[sweep_key], line_dash="dot",
                              line_color=P["g700"], annotation_text="current")
                fig.update_layout(xaxis_title=FEATURE_BY_KEY[sweep_key].label,
                                  yaxis_title="Effort (hours)")
                st.plotly_chart(styled(fig, 360, "One-way sensitivity"), width="stretch")
        with c2:
            kx = st.selectbox("Interaction X", BASE_KEYS, index=BASE_KEYS.index("PointsAjust"),
                              format_func=lambda k: FEATURE_BY_KEY[k].label)
            ky = st.selectbox("Interaction Y", BASE_KEYS, index=BASE_KEYS.index("TeamExp"),
                              format_func=lambda k: FEATURE_BY_KEY[k].label)
            if HAS_PLOTLY and kx != ky:
                xs, ys, z = svc.interaction_grid(values, kx, ky)
                fig = go.Figure(go.Heatmap(x=xs, y=ys, z=z, colorscale="Greens",
                                           colorbar=dict(title="Hours")))
                fig.update_layout(xaxis_title=FEATURE_BY_KEY[kx].label,
                                  yaxis_title=FEATURE_BY_KEY[ky].label)
                st.plotly_chart(styled(fig, 360, "Two-way interaction surface"),
                                width="stretch")
            elif kx == ky:
                st.info("Pick two different features to see their interaction.")

# ==========================================================================
# 4. DELIVERY PLAN
# ==========================================================================
with tabs[3]:
    res = st.session_state.get("result")
    if not res:
        st.info("Run an estimate first.")
    else:
        pred, values, samples = res["pred"], res["values"], res["samples"]
        pct = an.percentiles(samples)
        plan_basis = st.radio("Plan against", ["Point estimate", "P80 commit figure"],
                              horizontal=True, index=1)
        effort = pred.point if plan_basis == "Point estimate" else pct["P80"]
        sched = an.schedule(effort, team_size, hours_month, comm_penalty)

        c = st.columns(4)
        with c[0]: kpi("Planned effort", f"{effort:,.0f} h", plan_basis.lower(), 0.0)
        with c[1]: kpi("Duration", f"{sched['duration_months']:,.1f} mo", f"team of {team_size}", .08)
        with c[2]: kpi("Effective capacity", f"{sched['effective_team']:.1f} FTE",
                       f"{sched['efficiency']:.0%} of head-count", .16)
        with c[3]: kpi("Total cost", f"${effort * rate:,.0f}", f"at ${rate}/hour", .24)

        st.markdown("### Phase breakdown")
        phases = an.phase_plan(effort, sched["duration_months"], rate)
        if HAS_PLOTLY:
            fig = go.Figure()
            shades = [P["g200"], P["g300"], P["g400"], P["g500"], P["g600"]]
            for i, row in phases.iterrows():
                fig.add_trace(go.Bar(
                    y=["Timeline"], x=[row["End"] - row["Start"]], base=[row["Start"]],
                    orientation="h", name=row["Phase"], marker_color=shades[i % len(shades)],
                    hovertemplate=f"{row['Phase']}: {row['Hours']:,} h<extra></extra>"))
            fig.update_layout(barmode="stack", xaxis_title="Months from kickoff")
            st.plotly_chart(styled(fig, 260, "Gantt-style phase timeline"), width="stretch")
        st.dataframe(phases, width="stretch")

        st.markdown("### Staffing profile")
        st.caption("Rayleigh (Putnam/Norden) curve: teams ramp up, peak before the end, then taper "
                   "into stabilisation - flat staffing is a planning fiction.")
        curve = an.staffing_curve(effort, sched["duration_months"])
        if HAS_PLOTLY and len(curve):
            fig = go.Figure(go.Scatter(x=curve["Month"], y=curve["Staff"], mode="lines",
                                       line=dict(color=P["g600"], width=4, shape="spline"),
                                       fill="tozeroy", fillcolor="rgba(95,201,138,.25)"))
            fig.add_hline(y=team_size, line_dash="dot", line_color=P["g700"],
                          annotation_text="planned head-count")
            fig.update_layout(xaxis_title="Month", yaxis_title="People")
            st.plotly_chart(styled(fig, 340), width="stretch")

        st.markdown("### Team-size trade-off")
        sizes = list(range(1, 26))
        rows = [an.schedule(effort, n, hours_month, comm_penalty) for n in sizes]
        trade = pd.DataFrame(dict(Team=sizes, Months=[r["duration_months"] for r in rows],
                                  Efficiency=[r["efficiency"] for r in rows]))
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trade["Team"], y=trade["Months"], mode="lines+markers",
                                     name="Duration (months)", line=dict(color=P["g600"], width=3)))
            fig.add_trace(go.Scatter(x=trade["Team"], y=trade["Efficiency"] * 100, mode="lines",
                                     name="Efficiency (%)", yaxis="y2",
                                     line=dict(color=P["g300"], width=3, dash="dot")))
            fig.update_layout(xaxis_title="Team size",
                              yaxis=dict(title="Months"),
                              yaxis2=dict(title="Efficiency (%)", overlaying="y", side="right"))
            st.plotly_chart(styled(fig, 360, "Adding people stops helping"), width="stretch")
        best = trade.loc[trade["Months"].idxmin()]
        st.info(f"Under these assumptions the schedule stops improving beyond about "
                f"{int(best['Team'])} people ({best['Months']:.1f} months); past that, coordination "
                "overhead outweighs the extra capacity.")

        st.markdown("### Export")
        risks = an.risk_profile(values, pred, sched)
        score, level, _ = an.overall_risk(risks)
        drivers = svc.shapley(values, n_permutations=60)[["Feature", "Value", "Contribution"]].round(1)
        html = build_html_report(
            project=project_name, author=author, prediction=pred, sched=sched, pct=pct,
            rate=rate, inputs=engineer(pd.DataFrame([values])), phases=phases,
            drivers=drivers, risks=risks.round(1), risk_score=score, risk_level=level)
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("Download HTML report", html.encode(),
                               f"{project_name.replace(' ', '_')}_estimate.html",
                               "text/html", width="stretch")
        with e2:
            st.download_button("Download phase plan (CSV)", phases.to_csv(index=False).encode(),
                               "phase_plan.csv", "text/csv", width="stretch")

# ==========================================================================
# 5. PORTFOLIO (batch)
# ==========================================================================
with tabs[4]:
    st.markdown("### Portfolio estimation")
    st.caption("Upload a CSV of projects to estimate them all at once, then see the capacity "
               "the portfolio needs.")
    template = pd.DataFrame([{f.key: f.default for f in BASE_FEATURES}])
    st.download_button("Download CSV template", template.to_csv(index=False).encode(),
                       "portfolio_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(up)
            missing = [k for k in BASE_KEYS if k not in data.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                with st.spinner(f"Estimating {len(data)} projects..."):
                    data["Effort_Hours"] = svc.predict(data[BASE_KEYS]).round()
                    data["Cost_USD"] = (data["Effort_Hours"] * rate).round()
                    data["Duration_Months"] = [
                        round(an.schedule(h, team_size, hours_month, comm_penalty)["duration_months"], 1)
                        for h in data["Effort_Hours"]]
                    data["Size"] = [an.classify_size(h)[0] for h in data["Effort_Hours"]]
                st.success(f"Estimated {len(data)} projects.")
                c = st.columns(4)
                c[0].metric("Total effort", f"{data['Effort_Hours'].sum():,.0f} h")
                c[1].metric("Total cost", f"${data['Cost_USD'].sum():,.0f}")
                c[2].metric("Median project", f"{data['Effort_Hours'].median():,.0f} h")
                c[3].metric("Peak duration", f"{data['Duration_Months'].max():,.1f} mo")

                if HAS_PLOTLY:
                    g1, g2 = st.columns(2)
                    with g1:
                        fig = go.Figure(go.Histogram(x=data["Effort_Hours"], nbinsx=30,
                                                     marker_color=P["g400"]))
                        fig.update_layout(xaxis_title="Hours", yaxis_title="Projects")
                        st.plotly_chart(styled(fig, 330, "Effort distribution"),
                                        width="stretch")
                    with g2:
                        mix = data["Size"].value_counts()
                        fig = go.Figure(go.Pie(labels=mix.index, values=mix.values, hole=.55,
                                               marker=dict(colors=[P["g200"], P["g400"],
                                                                   P["g500"], P["g700"]])))
                        st.plotly_chart(styled(fig, 330, "Portfolio mix"), width="stretch")

                    fig = go.Figure(go.Scatter(
                        x=data["PointsAjust"], y=data["Effort_Hours"], mode="markers",
                        marker=dict(size=10, color=data["Effort_Hours"], colorscale="Greens",
                                    showscale=True, line=dict(width=1, color=P["g700"]))))
                    fig.update_layout(xaxis_title="Adjusted function points",
                                      yaxis_title="Predicted effort (hours)")
                    st.plotly_chart(styled(fig, 380, "Scope vs effort across the portfolio"),
                                    width="stretch")

                st.dataframe(data, width="stretch")
                st.download_button("Download portfolio results",
                                   data.to_csv(index=False).encode(),
                                   "portfolio_estimates.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process that file: {exc}")

# ==========================================================================
# 6. MODEL CARD
# ==========================================================================
with tabs[5]:
    st.markdown("### Model card")
    info = svc.info()
    c = st.columns(3)
    for i, (k, v) in enumerate(info.items()):
        with c[i % 3]:
            kpi(k, str(v), "", i * 0.05)

    st.write("")
    imp = svc.global_importance()
    if imp is not None and HAS_PLOTLY:
        st.markdown("### Global feature importance")
        d = imp.sort_values("Importance")
        fig = go.Figure(go.Bar(x=d["Importance"], y=d["Feature"], orientation="h",
                               marker_color=P["g500"]))
        st.plotly_chart(styled(fig, 420, "What the model relies on overall"),
                        width="stretch")
    elif imp is None:
        st.caption("This model does not expose feature importances; use the Explainability tab, "
                   "which works with any regressor.")

    st.markdown(
        """
#### Intended use
Early-stage effort estimation for function-point-sized software projects, to support
budgeting and staffing conversations. It is a decision aid, not a contract.

#### Engineered features
- `AdjustmentImpact = PointsNonAdjust - PointsAjust`
- `ExperienceScore = TeamExp + ManagerExp`

#### Known limitations
- The training data comes from an era of `YearEnd` values 82-88, so the effort levels reflect
  the practices of that period rather than modern tooling.
- Predictions outside the input ranges shown are extrapolation and are not validated.
- The interval is a spread measure, not a calibrated frequentist interval, unless the
  underlying model is a bagging ensemble.
- The schedule, staffing and risk models are transparent published heuristics
  (Brooks' law, Putnam/Norden Rayleigh curve, phase splits), not learned from your data.

#### Responsible use
Commit to the P80 figure rather than the point estimate, state the assumptions alongside
the number, and re-estimate as scope becomes clearer.
        """
    )

# ==========================================================================
# 7. HISTORY
# ==========================================================================
with tabs[6]:
    st.markdown("### Session history")
    if not st.session_state.history:
        st.info("No estimates yet.")
    else:
        hist = pd.DataFrame(st.session_state.history)
        hist.insert(0, "Run", range(1, len(hist) + 1))
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["Run"], y=hist["High"], mode="lines",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=hist["Run"], y=hist["Low"], mode="lines",
                                     line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(95,201,138,.25)", name="Interval"))
            fig.add_trace(go.Scatter(x=hist["Run"], y=hist["Effort"], mode="lines+markers",
                                     line=dict(color=P["g600"], width=3), name="Estimate"))
            fig.update_layout(xaxis_title="Run", yaxis_title="Effort (hours)")
            st.plotly_chart(styled(fig, 360, "How your estimates evolved"),
                            width="stretch")
        st.dataframe(hist, width="stretch")
        st.download_button("Export history", hist.to_csv(index=False).encode(),
                           "estimate_history.csv", "text/csv")

st.markdown("<div class='footer'>EffortAI &middot; Software Effort Intelligence &middot; "
            "machine learning, uncertainty quantification and delivery planning</div>",
            unsafe_allow_html=True)
