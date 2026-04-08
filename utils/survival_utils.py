import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import plotly.graph_objects as go


def fit_km(durations, events, label="All"):
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label=label)
    return kmf


def km_plot_stratified(df: pd.DataFrame, group_col: str, time_col="os_time",
                       event_col="os_event", title="Kaplan-Meier Survival") -> tuple[go.Figure, dict]:
    """Plot KM curves stratified by group_col. Returns (figure, stats_dict)."""
    colors = ["#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#f9e2af"]
    fig = go.Figure()
    groups = sorted(df[group_col].dropna().unique())
    stats = {}

    for i, group in enumerate(groups):
        mask = df[group_col] == group
        sub = df.loc[mask].dropna(subset=[time_col, event_col])
        if len(sub) < 2:
            continue
        kmf = fit_km(sub[time_col], sub[event_col], label=str(group))
        timeline = kmf.survival_function_
        fig.add_trace(go.Scatter(
            x=timeline.index, y=timeline.iloc[:, 0],
            mode="lines", name=f"{group} (n={len(sub)})",
            line=dict(color=colors[i % len(colors)], width=2)
        ))
        stats[group] = {"n": len(sub), "median_survival": kmf.median_survival_time_}

    # Log-rank test if 2 groups
    if len(groups) == 2:
        g1 = df[df[group_col] == groups[0]].dropna(subset=[time_col, event_col])
        g2 = df[df[group_col] == groups[1]].dropna(subset=[time_col, event_col])
        if len(g1) >= 2 and len(g2) >= 2:
            lr = logrank_test(g1[time_col], g2[time_col], g1[event_col], g2[event_col])
            stats["logrank_p"] = lr.p_value
            title += f" (log-rank p={lr.p_value:.4f})"

    fig.update_layout(
        title=title, xaxis_title="Time (days)", yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.95)", font=dict(color="#cdd6f4"),
        legend=dict(x=0.7, y=0.95)
    )
    return fig, stats


def fit_cox(df: pd.DataFrame, covariates: list[str], time_col="os_time",
            event_col="os_event", penalizer=0.1) -> CoxPHFitter:
    cols = covariates + [time_col, event_col]
    sub = df[cols].dropna()
    # standardize continuous covariates
    for c in covariates:
        if sub[c].nunique() > 5:
            sub[c] = (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-8)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(sub, duration_col=time_col, event_col=event_col)
    return cph


def cox_forest_plot(cph: CoxPHFitter, title="Cox PH Hazard Ratios") -> go.Figure:
    summary = cph.summary
    names = summary.index.tolist()
    hrs = summary["exp(coef)"].values
    ci_lower = summary["exp(coef) lower 95%"].values
    ci_upper = summary["exp(coef) upper 95%"].values
    pvals = summary["p"].values

    colors = ["#f38ba8" if p < 0.05 else "#6c7086" for p in pvals]

    fig = go.Figure()
    fig.add_vline(x=1, line_dash="dash", line_color="#585b70", opacity=0.7)
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=[ci_lower[i], ci_upper[i]], y=[name, name],
            mode="lines", line=dict(color=colors[i], width=2), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[hrs[i]], y=[name], mode="markers",
            marker=dict(color=colors[i], size=10), showlegend=False,
            text=f"HR={hrs[i]:.2f}, p={pvals[i]:.3f}", hoverinfo="text"
        ))

    fig.update_layout(
        title=title, xaxis_title="Hazard Ratio (95% CI)",
        xaxis_type="log", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,17,27,0.95)",
        font=dict(color="#cdd6f4"), height=max(300, len(names) * 35)
    )
    return fig


def compute_risk_scores(cph: CoxPHFitter, df: pd.DataFrame, covariates: list[str],
                        time_col="os_time", event_col="os_event") -> pd.DataFrame:
    cols = covariates + [time_col, event_col]
    sub = df[cols].dropna().copy()
    for c in covariates:
        if sub[c].nunique() > 5:
            sub[c] = (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-8)
    sub["risk_score"] = cph.predict_partial_hazard(sub)
    tertiles = pd.qcut(sub["risk_score"], q=3, labels=["Low", "Medium", "High"])
    sub["risk_group"] = tertiles
    return sub
