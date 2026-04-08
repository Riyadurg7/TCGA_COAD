import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from utils.plotting import apply_theme, COLORS, THEME


# Variables exposed for cross-tabulation
CATEGORICAL_VARS = {
    "Gender": "gender",
    "Stage": "stage_coarse",
    "Vital Status": "vital_status",
    "Colon Side": "colon_side",
    "Metastatic": "is_metastatic",
    "Node Positive": "node_positive",
    "Age Group": "age_group",
    "Sample Type": "sample_type",
}

NUMERIC_VARS = {
    "Age": "age",
    "Stage (numeric)": "stage_numeric",
}


def render(clinical: pd.DataFrame, survival: pd.DataFrame):
    st.title("Clinical Explorer")

    tab_cross, tab_side, tab_treat = st.tabs(["Cross-tabulation", "Right vs Left Colon", "Treatment Overview"])

    # ── Tab 1: Cross-tabulation ───────────────────────────
    with tab_cross:
        st.subheader("Interactive Variable Explorer")
        st.caption("Pick any two clinical variables to see their relationship.")

        all_vars = {**CATEGORICAL_VARS, **NUMERIC_VARS}
        available_vars = {k: v for k, v in all_vars.items() if v in clinical.columns}

        col1, col2 = st.columns(2)
        var_a_name = col1.selectbox("Variable A", list(available_vars.keys()), index=0, key="var_a")
        var_b_name = col2.selectbox("Variable B", list(available_vars.keys()), index=1, key="var_b")

        var_a = available_vars[var_a_name]
        var_b = available_vars[var_b_name]

        is_cat_a = var_a_name in CATEGORICAL_VARS
        is_cat_b = var_b_name in CATEGORICAL_VARS

        sub = clinical[[var_a, var_b]].dropna()

        if len(sub) < 5:
            st.warning("Not enough data for this combination.")
        elif is_cat_a and is_cat_b:
            _plot_cat_cat(sub, var_a, var_b, var_a_name, var_b_name)
        elif is_cat_a and not is_cat_b:
            _plot_cat_num(sub, var_a, var_b, var_a_name, var_b_name)
        elif not is_cat_a and is_cat_b:
            _plot_cat_num(sub, var_b, var_a, var_b_name, var_a_name)
        else:
            _plot_num_num(sub, var_a, var_b, var_a_name, var_b_name)

    # ── Tab 2: Right vs Left Colon ────────────────────────
    with tab_side:
        st.subheader("Right vs Left Colon Comparison")
        st.caption("Right-sided (cecum → transverse) vs left-sided (splenic → sigmoid) colon cancers.")

        if "colon_side" not in clinical.columns:
            st.warning("Colon side data not available.")
            return

        sides = clinical[clinical["colon_side"].isin(["Right colon", "Left colon"])].copy()

        if len(sides) < 10:
            st.warning("Insufficient data for comparison.")
            return

        # Merge survival for this analysis
        sides_merged = sides.merge(survival, left_on="sample_id", right_on="sample", how="inner")

        c1, c2 = st.columns(2)

        with c1:
            # Stage distribution by side
            ct = pd.crosstab(sides["stage_coarse"], sides["colon_side"], normalize="columns") * 100
            fig = px.bar(ct, barmode="group", title="Stage Distribution by Colon Side (%)",
                         color_discrete_sequence=[COLORS["primary"], COLORS["secondary"]])
            apply_theme(fig)
            fig.update_layout(height=350, xaxis_title="Stage", yaxis_title="%")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Age distribution by side
            fig = px.box(sides, x="colon_side", y="age", color="colon_side",
                         color_discrete_map={"Right colon": COLORS["primary"],
                                             "Left colon": COLORS["secondary"]},
                         title="Age by Colon Side")
            apply_theme(fig)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            # Gender by side
            ct = pd.crosstab(sides["gender"], sides["colon_side"])
            fig = px.bar(ct, barmode="group", title="Gender by Colon Side",
                         color_discrete_sequence=[COLORS["primary"], COLORS["secondary"]])
            apply_theme(fig)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            # Survival comparison
            if "os_time" in sides_merged.columns:
                fig = px.box(sides_merged, x="colon_side", y="os_time", color="colon_side",
                             color_discrete_map={"Right colon": COLORS["primary"],
                                                 "Left colon": COLORS["secondary"]},
                             title="Overall Survival Time by Colon Side")
                apply_theme(fig)
                fig.update_layout(height=350, showlegend=False, yaxis_title="OS Time (days)")
                st.plotly_chart(fig, use_container_width=True)

        # Statistical summary
        st.markdown("**Statistical Tests**")
        summaries = []

        # Chi-square: stage vs side
        ct_raw = pd.crosstab(sides["stage_coarse"], sides["colon_side"])
        if ct_raw.shape[0] > 1 and ct_raw.shape[1] > 1:
            chi2, p, _, _ = stats.chi2_contingency(ct_raw)
            summaries.append({"Test": "Stage vs Side", "Method": "Chi-square", "p-value": f"{p:.4f}"})

        # Mann-Whitney: age vs side
        right_age = sides[sides["colon_side"] == "Right colon"]["age"].dropna()
        left_age = sides[sides["colon_side"] == "Left colon"]["age"].dropna()
        if len(right_age) > 5 and len(left_age) > 5:
            _, p = stats.mannwhitneyu(right_age, left_age, alternative="two-sided")
            summaries.append({"Test": "Age vs Side", "Method": "Mann-Whitney U", "p-value": f"{p:.4f}"})

        if summaries:
            st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)

    # ── Tab 3: Treatment Overview ─────────────────────────
    with tab_treat:
        st.subheader("Treatment Overview")

        if "received_chemo" not in clinical.columns:
            st.warning("Treatment data not available.")
            return

        tumor_only = clinical[clinical["sample_type"] == "Primary Tumor"].copy()

        c1, c2 = st.columns(2)

        with c1:
            # Treatment modality counts
            chemo = tumor_only["received_chemo"].sum()
            radiation = tumor_only.get("received_radiation", pd.Series([0])).sum()
            neither = len(tumor_only) - max(chemo, radiation)

            labels = ["Chemotherapy", "Radiation", "Neither/Unknown"]
            values = [int(chemo), int(radiation), int(neither)]
            fig = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["muted"]]
            ))
            fig.update_layout(title="Treatment Modalities", yaxis_title="Count")
            apply_theme(fig)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Treatment by stage
            if "stage_coarse" in tumor_only.columns:
                stage_treat = tumor_only.groupby("stage_coarse")["received_chemo"].mean() * 100
                stage_treat = stage_treat.sort_index()
                fig = go.Figure(go.Bar(
                    x=stage_treat.index, y=stage_treat.values,
                    marker_color=COLORS["accent"]
                ))
                fig.update_layout(title="% Receiving Chemotherapy by Stage",
                                  yaxis_title="% Patients", xaxis_title="Stage")
                apply_theme(fig)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)


def _plot_cat_cat(df, var_a, var_b, name_a, name_b):
    ct = pd.crosstab(df[var_a], df[var_b])
    chi2, p, dof, _ = stats.chi2_contingency(ct)

    fig = px.bar(ct, barmode="group", title=f"{name_a} vs {name_b}",
                 color_discrete_sequence=[COLORS["primary"], COLORS["secondary"],
                                          COLORS["success"], COLORS["warning"]])
    apply_theme(fig)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Chi-square test: χ²={chi2:.2f}, p={p:.4f}, df={dof}")


def _plot_cat_num(df, cat_col, num_col, cat_name, num_name):
    groups = df[cat_col].unique()
    group_data = [df[df[cat_col] == g][num_col].dropna().values for g in groups]
    group_data = [g for g in group_data if len(g) > 0]

    fig = px.box(df, x=cat_col, y=num_col, color=cat_col, title=f"{num_name} by {cat_name}",
                 color_discrete_sequence=[COLORS["primary"], COLORS["secondary"],
                                          COLORS["success"], COLORS["warning"]])
    apply_theme(fig)
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if len(group_data) >= 2:
        stat, p = stats.kruskal(*group_data)
        st.caption(f"Kruskal-Wallis test: H={stat:.2f}, p={p:.4f}")


def _plot_num_num(df, var_a, var_b, name_a, name_b):
    r, p = stats.pearsonr(df[var_a].astype(float), df[var_b].astype(float))

    fig = px.scatter(df, x=var_a, y=var_b, trendline="ols",
                     title=f"{name_a} vs {name_b}",
                     color_discrete_sequence=[COLORS["accent"]])
    apply_theme(fig)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Pearson correlation: r={r:.3f}, p={p:.4f}")
