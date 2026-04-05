import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
from utils.plotting import apply_theme, COLORS, THEME


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def render(expr: pd.DataFrame, clinical: pd.DataFrame, survival: pd.DataFrame):
    st.title("Patient Risk Calculator")
    st.caption("Enter patient clinical features and gene expression values to predict risk group and survival outlook.")

    # Check for pre-trained models
    risk_path = os.path.join(MODELS_DIR, "risk_model.joblib")
    if not os.path.exists(risk_path):
        st.error(
            "Pre-trained risk model not found. Run the training script first:\n\n"
            "```\npython scripts/train_all_models.py\n```"
        )
        return

    risk_data = joblib.load(risk_path)
    cph = risk_data["model"]
    gene_features = risk_data["gene_features"]
    clinical_features = risk_data["clinical_features"]
    test_df = risk_data["test_df"]
    feature_stats = risk_data.get("feature_stats", {})  # {feat: (mean, std)}

    st.markdown("---")

    # ── Input Form ────────────────────────────────────────
    st.subheader("Patient Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Clinical Features**")

        age = st.slider("Age at diagnosis", 20, 95, 65, key="rc_age")
        gender = st.radio("Gender", ["Male", "Female"], key="rc_gender", horizontal=True)
        stage = st.selectbox("AJCC Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"], index=1, key="rc_stage")
        colon_side = st.selectbox("Colon Side", ["Right colon", "Left colon"], key="rc_side")
        node_status = st.radio("Lymph Node Involvement", ["N0 (negative)", "N+ (positive)"],
                               key="rc_node", horizontal=True)
        metastatic = st.radio("Distant Metastasis", ["M0 (none)", "M1 (present)"],
                              key="rc_meta", horizontal=True)

    with col2:
        st.markdown("**Gene Expression Values** (optional)")
        st.caption("Enter log2-normalized expression values for top prognostic genes. "
                   "Leave at defaults to use population median.")

        # Get median values from expression data as defaults
        gene_values = {}
        display_genes = gene_features[:10]  # Show top 10 prognostic genes

        for gene in display_genes:
            if gene in expr.index:
                default_val = float(expr.loc[gene].median())
                gene_values[gene] = st.number_input(
                    f"{gene}", value=round(default_val, 2),
                    min_value=0.0, max_value=20.0, step=0.1,
                    key=f"rc_gene_{gene}"
                )
            else:
                gene_values[gene] = st.number_input(
                    f"{gene}", value=5.0, min_value=0.0, max_value=20.0, step=0.1,
                    key=f"rc_gene_{gene}"
                )

    st.markdown("---")

    # ── Predict ───────────────────────────────────────────
    if st.button("Calculate Risk", type="primary", key="rc_predict"):
        with st.spinner("Computing risk prediction..."):
            # Build feature vector
            stage_map = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
            input_data = {}

            # Clinical features
            if "age" in clinical_features:
                input_data["age"] = age
            if "stage_numeric" in clinical_features:
                input_data["stage_numeric"] = stage_map[stage]
            if "node_positive" in clinical_features:
                input_data["node_positive"] = 0 if "N0" in node_status else 1
            if "is_metastatic" in clinical_features:
                input_data["is_metastatic"] = 0 if "M0" in metastatic else 1

            # Gene features — use input values for displayed genes, median for rest
            for gene in gene_features:
                if gene in gene_values:
                    input_data[gene] = gene_values[gene]
                elif gene in expr.index:
                    input_data[gene] = float(expr.loc[gene].median())
                else:
                    input_data[gene] = 5.0

            # Create DataFrame matching model's expected features
            all_features = [f for f in gene_features + clinical_features
                            if f in cph.summary.index]
            input_df = pd.DataFrame([input_data])

            # Standardize using saved training statistics
            for feat in all_features:
                if feat in input_df.columns and feat in feature_stats:
                    feat_mean, feat_std = feature_stats[feat]
                    input_df[feat] = (input_df[feat] - feat_mean) / feat_std

            # Ensure all required columns exist
            for feat in all_features:
                if feat not in input_df.columns:
                    input_df[feat] = 0.0

            try:
                risk_score = float(cph.predict_partial_hazard(input_df[all_features]).values[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            # Determine risk group based on test set distribution
            test_scores = test_df["risk_score"].values
            q33 = np.percentile(test_scores, 33)
            q66 = np.percentile(test_scores, 66)

            if risk_score <= q33:
                risk_group = "Low"
                risk_color = COLORS["success"]
                risk_desc = "Lower than average risk. Favorable prognosis indicators."
            elif risk_score <= q66:
                risk_group = "Medium"
                risk_color = COLORS["warning"]
                risk_desc = "Moderate risk. Standard monitoring and treatment protocols recommended."
            else:
                risk_group = "High"
                risk_color = COLORS["secondary"]
                risk_desc = "Elevated risk. Aggressive treatment and close follow-up may be warranted."

            # ── Display Results ───────────────────────────
            st.markdown("---")
            st.subheader("Prediction Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score", f"{risk_score:.3f}")
            c2.metric("Risk Group", risk_group)
            c3.metric("Model C-index", f"{risk_data['c_index_train']:.3f}")

            st.markdown(f"<div style='background-color:{risk_color}22; border-left:4px solid {risk_color}; "
                        f"padding:15px; border-radius:5px; margin:10px 0;'>"
                        f"<strong style='color:{risk_color}'>{risk_group} Risk</strong><br>"
                        f"{risk_desc}</div>", unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": "Patient Risk Score", "font": {"color": COLORS["text"]}},
                gauge={
                    "axis": {"range": [float(test_scores.min()), float(test_scores.max())],
                             "tickcolor": COLORS["text"]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [float(test_scores.min()), q33], "color": "rgba(166,227,161,0.2)"},
                        {"range": [q33, q66], "color": "rgba(250,179,135,0.2)"},
                        {"range": [q66, float(test_scores.max())], "color": "rgba(243,139,168,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["text"], "width": 3},
                        "thickness": 0.8, "value": risk_score
                    }
                },
                number={"font": {"color": risk_color}},
            ))
            fig.update_layout(**THEME, height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Contributing factors
            st.subheader("Contributing Factors")
            summary = cph.summary
            contributions = []
            for feat in all_features:
                if feat in summary.index and feat in input_df.columns:
                    coef = summary.loc[feat, "coef"]
                    val = float(input_df[feat].values[0])
                    contribution = coef * val
                    hr = np.exp(coef)
                    contributions.append({
                        "Feature": feat,
                        "Your Value": f"{input_data.get(feat, 'N/A')}",
                        "Hazard Ratio": f"{hr:.2f}",
                        "Contribution": contribution,
                        "Effect": "Increases risk" if contribution > 0 else "Decreases risk"
                    })

            contrib_df = pd.DataFrame(contributions)
            contrib_df = contrib_df.sort_values("Contribution", key=abs, ascending=False)

            # Bar chart of contributions
            top_contrib = contrib_df.head(15)
            colors = [COLORS["secondary"] if c > 0 else COLORS["success"]
                      for c in top_contrib["Contribution"]]
            fig = go.Figure(go.Bar(
                x=top_contrib["Contribution"].values[::-1],
                y=top_contrib["Feature"].values[::-1],
                orientation="h", marker_color=colors[::-1]
            ))
            fig.update_layout(
                title="Top Contributing Factors to Risk Score",
                xaxis_title="Contribution (+ = increases risk)", **THEME,
                height=max(300, len(top_contrib) * 30), margin=dict(l=150)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            with st.expander("Full Factor Breakdown"):
                st.dataframe(contrib_df[["Feature", "Your Value", "Hazard Ratio", "Effect"]],
                             use_container_width=True, hide_index=True)

            st.caption(
                "**Disclaimer:** This is a research/educational tool based on TCGA-COAD data. "
                "Results should not be used for clinical decision-making."
            )
