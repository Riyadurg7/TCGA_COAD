import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.plotting import make_donut, apply_theme, THEME, COLORS


def render(expr: pd.DataFrame, clinical: pd.DataFrame, survival: pd.DataFrame):
    st.title("Data Overview")
    st.caption("TCGA Colon Adenocarcinoma — Dataset Summary")

    # ── KPI Metrics ───────────────────────────────────────
    tumor_cols = [c for c in expr.columns if c.endswith("01A") or c.endswith("01B")]
    normal_cols = [c for c in expr.columns if c.endswith("11A") or c.endswith("11B")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(expr.columns))
    c2.metric("Tumor Samples", len(tumor_cols))
    c3.metric("Normal Samples", len(normal_cols))
    c4.metric("Genes Measured", f"{len(expr):,}")

    st.markdown("---")

    # ── Sample Composition & Stage Distribution ───────────
    col1, col2 = st.columns(2)

    with col1:
        if "sample_type" in clinical.columns:
            type_counts = clinical["sample_type"].value_counts()
            fig = make_donut(type_counts.index.tolist(), type_counts.values.tolist(),
                             "Sample Type Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "stage_coarse" in clinical.columns:
            stage_counts = clinical["stage_coarse"].dropna().value_counts().sort_index()
            fig = make_donut(stage_counts.index.tolist(), stage_counts.values.tolist(),
                             "AJCC Stage Distribution",
                             colors=[COLORS["primary"], COLORS["success"],
                                     COLORS["warning"], COLORS["secondary"]])
            st.plotly_chart(fig, use_container_width=True)

    # ── Demographics ──────────────────────────────────────
    st.subheader("Demographics")
    col1, col2 = st.columns(2)

    with col1:
        if "gender" in clinical.columns:
            gender_counts = clinical["gender"].dropna().value_counts()
            fig = px.bar(x=gender_counts.index, y=gender_counts.values,
                         color=gender_counts.index,
                         color_discrete_sequence=[COLORS["primary"], COLORS["secondary"]],
                         labels={"x": "Gender", "y": "Count"}, title="Gender Distribution")
            apply_theme(fig)
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "age" in clinical.columns:
            ages = clinical["age"].dropna()
            fig = px.histogram(ages, nbins=25, title="Age Distribution",
                               color_discrete_sequence=[COLORS["accent"]])
            fig.update_layout(xaxis_title="Age at Diagnosis", yaxis_title="Count")
            apply_theme(fig)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # ── Data Quality Heatmap ──────────────────────────────
    st.subheader("Data Quality — Non-missing Rate")
    quality_cols = ["gender", "age", "race", "vital_status", "stage", "stage_coarse",
                    "tumor_t", "tumor_n", "tumor_m", "anatomical_site", "diagnosis",
                    "sample_type", "tumor_grade", "treatment_types", "morphology"]
    available = [c for c in quality_cols if c in clinical.columns]
    if available:
        pct_present = clinical[available].notna().mean().sort_values(ascending=True)
        # Exclude values that are "not reported" or "Not Reported"
        for col in available:
            mask = clinical[col].isin(["not reported", "Not Reported", ""])
            pct_present[col] = (~mask & clinical[col].notna()).mean()

        fig = px.bar(x=pct_present.values * 100, y=pct_present.index, orientation="h",
                     color=pct_present.values, color_continuous_scale="RdYlGn",
                     labels={"x": "% Available", "y": ""},
                     title="Clinical Variable Completeness")
        apply_theme(fig)
        fig.update_layout(height=max(350, len(available) * 30), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Anatomical Site ───────────────────────────────────
    st.subheader("Tumor Anatomical Location")
    if "anatomical_site" in clinical.columns and "colon_side" in clinical.columns:
        site_df = clinical.dropna(subset=["anatomical_site"])
        site_counts = site_df.groupby(["anatomical_site", "colon_side"]).size().reset_index(name="count")
        site_counts = site_counts.sort_values("count", ascending=True)

        fig = px.bar(site_counts, x="count", y="anatomical_site", color="colon_side",
                     orientation="h",
                     color_discrete_map={"Right colon": COLORS["primary"],
                                         "Left colon": COLORS["secondary"],
                                         "Other/NOS": COLORS["muted"]},
                     labels={"count": "Count", "anatomical_site": "", "colon_side": "Colon Side"},
                     title="Tumor Sites (Right vs Left Colon)")
        apply_theme(fig)
        fig.update_layout(height=400, legend=dict(x=0.65, y=0.1))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Clinical note:** Right-sided colon cancers (cecum → transverse colon) and "
            "left-sided (splenic flexure → sigmoid) have different molecular profiles, "
            "prognosis, and treatment responses. This is a key distinction in COAD research."
        )

    # ── Dataset Previews ──────────────────────────────────
    st.subheader("Raw Data Preview")
    with st.expander("Expression Matrix (first 10 genes x 5 samples)"):
        st.dataframe(expr.iloc[:10, :5])

    with st.expander("Clinical Data (first 10 rows)"):
        display_cols = [c for c in ["sample_id", "sample_type", "gender", "age",
                                     "stage_coarse", "anatomical_site", "vital_status"]
                        if c in clinical.columns]
        st.dataframe(clinical[display_cols].head(10))

    with st.expander("Survival Data (first 10 rows)"):
        st.dataframe(survival.head(10))
