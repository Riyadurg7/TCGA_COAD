import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing import merge_all, filter_by_variance, get_tumor_normal_split
from utils.survival_utils import km_plot_stratified, fit_cox, cox_forest_plot, compute_risk_scores
from utils.plotting import COLORS, apply_theme


def render(expr: pd.DataFrame, clinical: pd.DataFrame, survival: pd.DataFrame):
    st.title("Survival Analysis")

    # Merge clinical + survival for this page
    merged = clinical.merge(survival, left_on="sample_id", right_on="sample", how="inner")
    # Keep only tumor samples for survival analysis
    tumor_mask = merged["sample_type"].str.contains("Tumor", case=False, na=False)
    tumor_surv = merged[tumor_mask].copy()

    if len(tumor_surv) < 10:
        st.error("Insufficient tumor samples with survival data.")
        return

    tab_km, tab_cox, tab_risk = st.tabs(["Kaplan-Meier", "Cox Regression", "Risk Stratification"])

    # ── Tab 1: Kaplan-Meier ───────────────────────────────
    with tab_km:
        st.subheader("Kaplan-Meier Survival Curves")

        stratify_options = {}
        if "stage_coarse" in tumor_surv.columns and tumor_surv["stage_coarse"].notna().sum() > 20:
            stratify_options["AJCC Stage"] = "stage_coarse"
        if "colon_side" in tumor_surv.columns and tumor_surv["colon_side"].notna().sum() > 20:
            stratify_options["Colon Side (Right vs Left)"] = "colon_side"
        if "age_group" in tumor_surv.columns and tumor_surv["age_group"].notna().sum() > 20:
            stratify_options["Age Group"] = "age_group"
        if "gender" in tumor_surv.columns and tumor_surv["gender"].notna().sum() > 20:
            stratify_options["Gender"] = "gender"
        if "is_metastatic" in tumor_surv.columns:
            tumor_surv["metastatic_status"] = tumor_surv["is_metastatic"].map({0: "M0", 1: "M1"})
            if tumor_surv["metastatic_status"].notna().sum() > 20:
                stratify_options["Metastatic Status"] = "metastatic_status"
        if "node_positive" in tumor_surv.columns:
            tumor_surv["node_status"] = tumor_surv["node_positive"].map({0: "N0 (negative)", 1: "N+ (positive)"})
            if tumor_surv["node_status"].notna().sum() > 20:
                stratify_options["Node Status"] = "node_status"

        # Gene expression stratification
        stratify_options["Gene Expression (median split)"] = "_gene_expr_"

        selected_strat = st.selectbox("Stratify by", list(stratify_options.keys()), key="km_strat")
        strat_col = stratify_options[selected_strat]

        if strat_col == "_gene_expr_":
            # Gene selection for expression-based stratification
            de_df = st.session_state.get("de_results")
            if de_df is not None:
                gene_options = de_df.head(100)["gene"].tolist()
            else:
                expr_filt = filter_by_variance(expr, top_n=1000)
                gene_options = expr_filt.index[:100].tolist()

            selected_gene = st.selectbox("Select gene", gene_options, key="km_gene")

            if selected_gene in expr.index:
                gene_expr = expr.loc[selected_gene]
                tumor_surv = tumor_surv.copy()
                tumor_surv["gene_expr"] = tumor_surv["sample_id"].map(gene_expr)
                median_val = tumor_surv["gene_expr"].median()
                tumor_surv["expression_group"] = tumor_surv["gene_expr"].apply(
                    lambda x: f"High {selected_gene}" if x >= median_val else f"Low {selected_gene}"
                )
                strat_col = "expression_group"
            else:
                st.warning(f"Gene {selected_gene} not found in expression data.")
                return

        fig, km_stats = km_plot_stratified(
            tumor_surv, strat_col, time_col="os_time", event_col="os_event",
            title=f"Survival by {selected_strat}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show stats
        with st.expander("Survival Statistics"):
            stats_rows = []
            for group, info in km_stats.items():
                if group == "logrank_p":
                    continue
                stats_rows.append({
                    "Group": group, "N": info["n"],
                    "Median Survival (days)": f"{info['median_survival']:.0f}" if np.isfinite(info["median_survival"]) else "Not reached"
                })
            if stats_rows:
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
            if "logrank_p" in km_stats:
                st.metric("Log-rank p-value", f"{km_stats['logrank_p']:.4f}")

    # ── Tab 2: Cox Proportional Hazards ───────────────────
    with tab_cox:
        st.subheader("Multivariate Cox Proportional Hazards")
        st.caption("Select covariates to include in the Cox PH model.")

        possible_covariates = {}
        if "age" in tumor_surv.columns:
            possible_covariates["Age"] = "age"
        if "stage_numeric" in tumor_surv.columns:
            possible_covariates["Stage (numeric)"] = "stage_numeric"
        if "gender" in tumor_surv.columns:
            tumor_surv["gender_binary"] = tumor_surv["gender"].map({"male": 0, "female": 1})
            possible_covariates["Gender"] = "gender_binary"
        if "node_positive" in tumor_surv.columns:
            possible_covariates["Node Positive"] = "node_positive"
        if "is_metastatic" in tumor_surv.columns:
            possible_covariates["Metastatic"] = "is_metastatic"

        selected_names = st.multiselect(
            "Covariates", list(possible_covariates.keys()),
            default=list(possible_covariates.keys())[:4],
            key="cox_covariates"
        )

        if len(selected_names) < 1:
            st.info("Select at least 1 covariate.")
            return

        selected_covs = [possible_covariates[n] for n in selected_names]

        if st.button("Fit Cox Model", key="fit_cox"):
            with st.spinner("Fitting Cox PH model..."):
                try:
                    cph = fit_cox(tumor_surv, selected_covs, time_col="os_time",
                                  event_col="os_event", penalizer=0.1)

                    c1, c2 = st.columns([2, 1])
                    with c1:
                        fig = cox_forest_plot(cph, title="Hazard Ratios (95% CI)")
                        st.plotly_chart(fig, use_container_width=True)

                    with c2:
                        st.metric("Concordance Index", f"{cph.concordance_index_:.3f}")
                        st.caption("C-index: 0.5 = random, 1.0 = perfect discrimination")

                        st.markdown("**Model Summary**")
                        summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
                        summary.columns = ["HR", "CI Lower", "CI Upper", "p-value"]
                        st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)

                    st.session_state["cox_model"] = cph
                    st.session_state["cox_covariates"] = selected_covs
                except Exception as e:
                    st.error(f"Cox model failed: {e}")

    # ── Tab 3: Risk Stratification ────────────────────────
    with tab_risk:
        st.subheader("Risk Group Stratification")

        cph = st.session_state.get("cox_model")
        covs = st.session_state.get("cox_covariates")

        if cph is None:
            st.info("Fit a Cox model in the **Cox Regression** tab first, then return here.")
            return

        with st.spinner("Computing risk scores..."):
            risk_df = compute_risk_scores(cph, tumor_surv, covs,
                                          time_col="os_time", event_col="os_event")

            if risk_df is None or len(risk_df) < 10:
                st.warning("Insufficient data for risk stratification.")
                return

            # KM by risk group
            fig, risk_stats = km_plot_stratified(
                risk_df, "risk_group", time_col="os_time", event_col="os_event",
                title="Survival by Predicted Risk Group"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk score distribution
            import plotly.express as px
            fig2 = px.histogram(risk_df, x="risk_score", color="risk_group",
                                color_discrete_map={"Low": COLORS["success"],
                                                    "Medium": COLORS["warning"],
                                                    "High": COLORS["secondary"]},
                                title="Risk Score Distribution", nbins=30)
            apply_theme(fig2)
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

            # Summary table
            summary_rows = []
            for group in ["Low", "Medium", "High"]:
                g = risk_df[risk_df["risk_group"] == group]
                if len(g) == 0:
                    continue
                summary_rows.append({
                    "Risk Group": group,
                    "N": len(g),
                    "Events": int(g["os_event"].sum()),
                    "Mean OS (days)": f"{g['os_time'].mean():.0f}",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
