import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from utils.survival_utils import km_plot_stratified, cox_forest_plot
from utils.plotting import (make_roc_curve, make_confusion_matrix, make_feature_importance,
                             COLORS, apply_theme)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _load_pretrained(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def render(expr: pd.DataFrame, clinical: pd.DataFrame, survival: pd.DataFrame):
    st.title("ML Prediction Lab")
    st.caption("Pre-trained machine learning models for cancer classification, staging, and risk prediction.")

    tab_tn, tab_stage, tab_risk = st.tabs([
        "Tumor vs Normal", "Stage Prediction", "Risk Stratification"
    ])

    # ── Tab A: Tumor vs Normal ────────────────────────────
    with tab_tn:
        results = _load_pretrained("tumor_normal.joblib")
        if results is None:
            st.error("Pre-trained model not found. Run `python scripts/train_all_models.py` first.")
            return

        meta = results["_meta"]
        # Pick the best model
        model_names = [k for k in results if k != "_meta"]
        best_name = max(model_names, key=lambda k: results[k]["auc"])
        best = results[best_name]

        st.subheader("Tumor vs Normal Classifier")

        # Model info card
        st.markdown(
            f"**Model:** {best_name}  \n"
            f"**Task:** Binary classification — Primary Tumor vs Solid Tissue Normal  \n"
            f"**Features:** Top 200 genes selected by ANOVA F-test (`SelectKBest`) from "
            f"5,000 most variable genes  \n"
            f"**Split:** Patient-aware 80/20 (`GroupShuffleSplit`) — same patient's tumor "
            f"and normal samples never split across train/test  \n"
            f"**Preprocessing:** `StandardScaler` → `SelectKBest(f_classif, k=200)`"
        )

        st.markdown("---")

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{best['auc']:.3f}")
        c2.metric("Tumor Samples", meta["n_tumor"])
        c3.metric("Normal Samples", meta["n_normal"])
        c4.metric("Test Samples", len(meta["y_test"]))

        # ROC
        fig_roc = make_roc_curve(
            {best_name: best["fpr"]}, {best_name: best["tpr"]},
            {best_name: best["auc"]}, title="ROC Curve — Tumor vs Normal"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        if best["auc"] > 0.95:
            st.info(
                f"**Why is AUC so high ({best['auc']:.3f})?** Tumor and normal tissue have "
                "fundamentally different transcriptomic profiles — thousands of genes are "
                "dysregulated in cancer. A patient-aware split prevents data leakage. "
                "The high AUC reflects genuine biological signal, not an artifact."
            )

        # Confusion matrix + classification report
        col1, col2 = st.columns([1, 1])
        with col1:
            cm = confusion_matrix(meta["y_test"], best["y_pred"])
            fig_cm = make_confusion_matrix(cm, meta["labels"], title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            report = classification_report(meta["y_test"], best["y_pred"],
                                           target_names=meta["labels"], output_dict=True)
            report_df = pd.DataFrame(report).T.iloc[:2]
            st.markdown("**Classification Report**")
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

        # Feature importance
        st.subheader("Top Biomarker Genes")
        st.caption("Genes most predictive of tumor vs normal — these drive the classifier's decisions.")
        fig_imp = make_feature_importance(
            best["feature_names"], best["importances"],
            title="Top 20 Predictive Genes", top_n=20
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Tab B: Stage Prediction ───────────────────────────
    with tab_stage:
        results = _load_pretrained("stage_predictor.joblib")
        if results is None:
            st.error("Pre-trained model not found. Run `python scripts/train_all_models.py` first.")
            return

        meta = results["_meta"]
        model_names = [k for k in results if k != "_meta"]
        best_name = max(model_names, key=lambda k: results[k]["mean_auc"])
        best = results[best_name]

        st.subheader("Cancer Stage Prediction")

        # Model info card
        st.markdown(
            f"**Model:** {best_name}  \n"
            f"**Task:** Binary classification — Early (Stage I/II) vs Late (Stage III/IV)  \n"
            f"**Feature selection:** Genes ranked by Mann-Whitney U test (stage-discriminative), "
            f"not by variance — ensures selected genes actually differ between stages  \n"
            f"**Evaluation:** Stratified 5-fold cross-validation  \n"
            f"**Preprocessing:** `StandardScaler` → per-fold `SelectKBest` (prevents feature selection leakage)"
        )

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean AUC (5-fold CV)", f"{best['mean_auc']:.3f}")
        c2.metric("Std AUC", f"\u00b1{best['std_auc']:.3f}")
        c3.metric("Early (I/II)", meta["n_early"])
        c4.metric("Late (III/IV)", meta["n_late"])

        # ROC
        fig_roc = make_roc_curve(
            {best_name: best["fpr"]}, {best_name: best["tpr"]},
            {best_name: best["mean_auc"]},
            title="ROC Curve — Stage Prediction"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Feature importance
        st.subheader("Top Predictive Features")
        st.caption("Features driving the early vs late stage distinction.")
        fig_imp = make_feature_importance(
            best["feature_names"], best["importances"],
            title="Top 25 Features for Stage Prediction", top_n=25
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.info(
            "**Interpretation:** Stage prediction from gene expression is inherently challenging. "
            f"An AUC of {best['mean_auc']:.3f} indicates the model captures meaningful signal "
            "beyond random chance (0.5). Clinical features like tumor extent and lymph node "
            "involvement remain the gold standard for staging, but gene expression adds "
            "complementary molecular information."
        )

    # ── Tab C: Risk Stratification ────────────────────────
    with tab_risk:
        result = _load_pretrained("risk_model.joblib")
        if result is None:
            st.error("Pre-trained model not found. Run `python scripts/train_all_models.py` first.")
            return

        st.subheader("Patient Risk Stratification")

        # Model info card
        st.markdown(
            f"**Model:** Cox Proportional Hazards (`lifelines.CoxPHFitter`)  \n"
            f"**Task:** Predict patient survival risk from gene expression + clinical features  \n"
            f"**Gene selection:** Top {len(result['gene_features'])} genes by univariate Cox p-value  \n"
            f"**Clinical features:** {', '.join(result['clinical_features']) if result['clinical_features'] else 'None'}  \n"
            f"**Split:** 70/30 train/test  \n"
            f"**Risk groups:** Patients split into Low/Medium/High by predicted hazard tertiles"
        )

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("C-index (train)", f"{result['c_index_train']:.3f}")
        c2.metric("Gene Features", len(result["gene_features"]))
        c3.metric("Clinical Features", len(result["clinical_features"]))

        st.caption("C-index: 0.5 = random, 1.0 = perfect. Values > 0.7 indicate good discrimination.")

        # KM by risk group on test set
        test_df = result["test_df"]
        fig, risk_stats = km_plot_stratified(
            test_df, "risk_group", time_col="os_time", event_col="os_event",
            title="Test Set — Survival by Predicted Risk Group"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forest plot
        cph = result["model"]
        fig_forest = cox_forest_plot(cph, title="Feature Hazard Ratios (95% CI)")
        st.plotly_chart(fig_forest, use_container_width=True)

        st.caption("HR > 1 = increases risk of death, HR < 1 = protective. Red = statistically significant (p < 0.05).")

        # Risk score distribution
        import plotly.express as px
        fig_hist = px.histogram(test_df, x="risk_score", color="risk_group",
                                color_discrete_map={"Low": COLORS["success"],
                                                    "Medium": COLORS["warning"],
                                                    "High": COLORS["secondary"]},
                                title="Risk Score Distribution (Test Set)", nbins=25)
        apply_theme(fig_hist)
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Summary table
        summary_rows = []
        for group in ["Low", "Medium", "High"]:
            g = test_df[test_df["risk_group"] == group]
            if len(g) == 0:
                continue
            summary_rows.append({
                "Risk Group": group, "N": len(g),
                "Events (deaths)": int(g["os_event"].sum()),
                "Mean OS (days)": f"{g['os_time'].mean():.0f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Top prognostic genes
        with st.expander("Top Prognostic Genes (by univariate Cox p-value)"):
            gene_pvals = result["gene_pvals"]
            top_genes_df = pd.DataFrame([
                {"Gene": g, "p-value": f"{p:.2e}"}
                for g, p in sorted(gene_pvals.items(), key=lambda x: x[1])[:30]
            ])
            st.dataframe(top_genes_df, use_container_width=True, hide_index=True)
