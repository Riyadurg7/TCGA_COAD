"""
Offline training script — run once before deployment.
Trains all ML models and saves artifacts to models/ directory.

Usage: python scripts/train_all_models.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib

# Patch streamlit cache decorators for offline use
import streamlit as st
st.cache_data = lambda **kwargs: lambda f: f

from utils.data_loader import load_expression, load_clinical, load_survival
from utils.preprocessing import (map_genes_to_symbols, filter_by_variance,
                                  clean_clinical, get_tumor_normal_split)
from utils.de_analysis import run_de
from utils.ml_models import train_tumor_normal, train_stage_predictor, train_risk_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("TCGA-COAD Insight Engine — Offline Model Training")
    print("=" * 60)

    # ── 1. Load and preprocess data ───────────────────────
    print("\n[1/6] Loading datasets...")
    expr_raw = load_expression()
    clinical = load_clinical()
    survival = load_survival()
    print(f"  Expression: {expr_raw.shape}, Clinical: {clinical.shape}, Survival: {survival.shape}")

    print("[2/6] Mapping genes and filtering...")
    expr = map_genes_to_symbols(expr_raw)
    clinical = clean_clinical(clinical)
    expr_filt = filter_by_variance(expr, top_n=5000)
    print(f"  Filtered expression: {expr_filt.shape}")

    # ── 2. Differential Expression ────────────────────────
    print("[3/6] Running differential expression...")
    tumor_df, normal_df = get_tumor_normal_split(expr_filt)
    de_results = run_de(tumor_df, normal_df)
    n_sig = de_results["significant"].sum()
    print(f"  {n_sig} significant DEGs found")
    de_results.to_csv(os.path.join(MODELS_DIR, "de_results.csv"), index=False)
    print("  Saved: models/de_results.csv")

    # ── 3. Train Tumor vs Normal ──────────────────────────
    print("[4/6] Training Tumor vs Normal classifier...")
    tn_results = train_tumor_normal(expr_filt, top_k_features=200)
    # Save only serializable parts
    tn_save = {}
    for name, data in tn_results.items():
        if name == "_meta":
            tn_save[name] = data
        else:
            tn_save[name] = {
                "model": data["model"],
                "fpr": data["fpr"],
                "tpr": data["tpr"],
                "auc": data["auc"],
                "y_pred": data["y_pred"],
                "y_proba": data["y_proba"],
                "importances": data["importances"],
                "feature_names": data["feature_names"],
            }
    joblib.dump(tn_save, os.path.join(MODELS_DIR, "tumor_normal.joblib"))
    for name in [k for k in tn_results if k != "_meta"]:
        print(f"  {name}: AUC = {tn_results[name]['auc']:.3f}")
    print("  Saved: models/tumor_normal.joblib")

    # ── 4. Build merged dataset ───────────────────────────
    print("[5/6] Building merged dataset and training stage predictor...")
    expr_t = expr_filt.T
    expr_t.index.name = "sample_id"
    expr_t = expr_t.reset_index()

    merged = clinical.merge(survival, left_on="sample_id", right_on="sample", how="inner")
    tumor_mask = merged["sample_type"].str.contains("Tumor", case=False, na=False)
    merged = merged[tumor_mask].copy()
    merged = merged.merge(expr_t, on="sample_id", how="inner")
    print(f"  Merged dataset: {merged.shape}")

    gene_cols = [c for c in merged.columns if c not in clinical.columns
                 and c not in ["sample_id", "sample", "os_time", "os_event", "patient_id"]]

    # Stage predictor
    stage_results = train_stage_predictor(merged, gene_cols, top_k_genes=200)
    stage_save = {}
    for name, data in stage_results.items():
        if name == "_meta":
            stage_save[name] = data
        else:
            stage_save[name] = {
                "model": data["model"],
                "mean_auc": data["mean_auc"],
                "std_auc": data["std_auc"],
                "fpr": data["fpr"],
                "tpr": data["tpr"],
                "importances": data["importances"],
                "feature_names": data["feature_names"],
            }
    joblib.dump(stage_save, os.path.join(MODELS_DIR, "stage_predictor.joblib"))
    best = max([k for k in stage_results if k != "_meta"], key=lambda k: stage_results[k]["mean_auc"])
    print(f"  Best model: {best}, Mean AUC = {stage_results[best]['mean_auc']:.3f}")
    print("  Saved: models/stage_predictor.joblib")

    # ── 5. Risk stratification ────────────────────────────
    print("[6/6] Training risk stratification model...")
    risk_result = train_risk_model(merged, gene_cols, top_k_genes=50)
    if risk_result is not None:
        risk_save = {
            "model": risk_result["model"],
            "c_index_train": risk_result["c_index_train"],
            "test_df": risk_result["test_df"],
            "feature_names": risk_result["feature_names"],
            "feature_stats": risk_result["feature_stats"],
            "gene_features": risk_result["gene_features"],
            "clinical_features": risk_result["clinical_features"],
            "gene_pvals": risk_result["gene_pvals"],
        }
        joblib.dump(risk_save, os.path.join(MODELS_DIR, "risk_model.joblib"))
        print(f"  C-index (train): {risk_result['c_index_train']:.3f}")
        print("  Saved: models/risk_model.joblib")
    else:
        print("  WARNING: Risk model training failed")

    # ── Save metadata ─────────────────────────────────────
    meta = {
        "n_genes_total": expr.shape[0],
        "n_genes_filtered": expr_filt.shape[0],
        "n_tumor": tumor_df.shape[1],
        "n_normal": normal_df.shape[1],
        "n_de_significant": int(n_sig),
        "gene_cols": gene_cols[:200],  # top gene columns used in ML
    }
    joblib.dump(meta, os.path.join(MODELS_DIR, "meta.joblib"))

    print("\n" + "=" * 60)
    print("All models trained and saved to models/ directory!")
    print("=" * 60)
    print("\nFiles created:")
    for f in os.listdir(MODELS_DIR):
        size = os.path.getsize(os.path.join(MODELS_DIR, f))
        print(f"  {f:30s} {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
