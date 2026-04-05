import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter
import warnings

warnings.filterwarnings("ignore")


# ---------- Task A: Tumor vs Normal Classifier ----------

def train_tumor_normal(expr: pd.DataFrame, top_k_features: int = 200):
    """
    expr: genes x samples DataFrame (already symbol-mapped and variance-filtered).
    Columns ending in 01A/01B = tumor, 11A/11B = normal.
    Returns dict with models, metrics, feature names.
    """
    tumor_cols = [c for c in expr.columns if c.endswith("01A") or c.endswith("01B")]
    normal_cols = [c for c in expr.columns if c.endswith("11A") or c.endswith("11B")]

    X_tumor = expr[tumor_cols].T.values
    X_normal = expr[normal_cols].T.values
    X = np.vstack([X_tumor, X_normal])
    y = np.array([1] * len(tumor_cols) + [0] * len(normal_cols))
    all_cols = tumor_cols + normal_cols
    feature_names = expr.index.tolist()

    # Patient-aware split: ensure same patient's tumor & normal
    # samples don't appear in both train and test sets
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    patient_ids = [c.rsplit("-", 2)[0] for c in all_cols]  # TCGA-XX-YYYY
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Feature selection + scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=min(top_k_features, X.shape[1]))
    X_train_sel = selector.fit_transform(X_train_sc, y_train)
    X_test_sel = selector.transform(X_test_sc)

    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]

    results = {}

    # Logistic Regression (L1)
    lr = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000,
                            class_weight="balanced", random_state=42)
    lr.fit(X_train_sel, y_train)
    lr_proba = lr.predict_proba(X_test_sel)[:, 1]
    lr_pred = lr.predict(X_test_sel)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    auc_lr = auc(fpr_lr, tpr_lr)
    results["Logistic Regression"] = {
        "model": lr, "fpr": fpr_lr, "tpr": tpr_lr, "auc": auc_lr,
        "y_pred": lr_pred, "y_proba": lr_proba,
        "importances": lr.coef_[0], "feature_names": selected_features,
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_sel, y_train)
    rf_proba = rf.predict_proba(X_test_sel)[:, 1]
    rf_pred = rf.predict(X_test_sel)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    auc_rf = auc(fpr_rf, tpr_rf)
    results["Random Forest"] = {
        "model": rf, "fpr": fpr_rf, "tpr": tpr_rf, "auc": auc_rf,
        "y_pred": rf_pred, "y_proba": rf_proba,
        "importances": rf.feature_importances_, "feature_names": selected_features,
    }

    results["_meta"] = {
        "y_test": y_test, "labels": ["Normal", "Tumor"],
        "n_tumor": len(tumor_cols), "n_normal": len(normal_cols),
    }
    return results


# ---------- Task B: Stage Predictor ----------

def _select_stage_genes(df, gene_cols, y, max_genes=300):
    """Select genes most associated with stage using Mann-Whitney U test,
    not just highest variance. This is the key improvement."""
    from scipy.stats import mannwhitneyu

    early_mask = y == 0
    late_mask = y == 1
    scored = []
    for g in gene_cols:
        if g not in df.columns:
            continue
        vals = df[g].values
        early_vals = vals[early_mask]
        late_vals = vals[late_mask]
        early_clean = early_vals[~np.isnan(early_vals)]
        late_clean = late_vals[~np.isnan(late_vals)]
        if len(early_clean) < 10 or len(late_clean) < 10:
            continue
        if np.std(early_clean) < 1e-6 and np.std(late_clean) < 1e-6:
            continue
        try:
            _, p = mannwhitneyu(early_clean, late_clean, alternative="two-sided")
            scored.append((g, p))
        except Exception:
            continue

    scored.sort(key=lambda x: x[1])
    return [g for g, _ in scored[:max_genes]]


def _build_engineered_features(df):
    """Engineer additional features beyond raw expression + basic clinical."""
    feats = {}

    # Clinical features
    if "age" in df.columns:
        feats["age"] = df["age"].values
    if "gender" in df.columns:
        feats["gender_encoded"] = df["gender"].map({"male": 0, "female": 1}).fillna(0).values
    if "colon_side" in df.columns:
        feats["side_right"] = (df["colon_side"] == "Right colon").astype(int).values
        feats["side_left"] = (df["colon_side"] == "Left colon").astype(int).values
    if "is_metastatic" in df.columns:
        feats["is_metastatic"] = df["is_metastatic"].fillna(0).values
    if "received_chemo" in df.columns:
        feats["received_chemo"] = df["received_chemo"].fillna(0).values

    return feats


def train_stage_predictor(merged_df: pd.DataFrame, gene_cols: list[str], top_k_genes: int = 200):
    """
    Predict Early (Stage I/II) vs Late (Stage III/IV) from gene expression + clinical.
    Uses stage-discriminative gene selection (Mann-Whitney) instead of variance-based.
    """
    df = merged_df.dropna(subset=["stage_numeric"]).copy()
    df["stage_binary"] = (df["stage_numeric"] >= 3).astype(int)  # 0=early, 1=late
    y = df["stage_binary"].values

    # KEY IMPROVEMENT: select genes by stage-association, not variance
    stage_genes = _select_stage_genes(df, gene_cols, y, max_genes=top_k_genes)

    # Engineered clinical features
    eng_feats = _build_engineered_features(df)
    clinical_feat_names = list(eng_feats.keys())
    for fname, fvals in eng_feats.items():
        df[fname] = fvals

    feature_sets = {
        "Expression only": stage_genes,
        "Expression + Clinical": stage_genes + clinical_feat_names,
    }

    results = {}

    # Expanded model grid with hyperparameter variations
    models = {
        "Logistic Regression (C=0.01)": LogisticRegression(
            penalty="l2", C=0.01, max_iter=5000, class_weight="balanced", random_state=42),
        "Logistic Regression (C=0.1)": LogisticRegression(
            penalty="l2", C=0.1, max_iter=5000, class_weight="balanced", random_state=42),
        "Logistic Regression (C=1.0)": LogisticRegression(
            penalty="l2", C=1.0, max_iter=5000, class_weight="balanced", random_state=42),
        "Random Forest (d=3)": RandomForestClassifier(
            n_estimators=300, max_depth=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "Random Forest (d=5)": RandomForestClassifier(
            n_estimators=300, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "Random Forest (d=8)": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced_subsample",
            min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting (lr=0.05)": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42),
        "Gradient Boosting (lr=0.1)": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42),
        "SVM (RBF)": None,  # handled separately
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fs_name, features in feature_sets.items():
        X = df[features].fillna(0).values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        # Also try SelectKBest within each fold for tighter feature selection
        for sel_k in [50, 100, None]:  # None = use all stage_genes
            if sel_k is not None and sel_k >= len(features):
                continue
            sel_label = f", top-{sel_k}" if sel_k else ""

            for model_name, model in models.items():
                key = f"{model_name} ({fs_name}{sel_label})"

                fold_aucs = []
                for train_idx, test_idx in cv.split(X_sc, y):
                    X_tr, X_te = X_sc[train_idx], X_sc[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]

                    # Per-fold feature selection (prevents leakage)
                    if sel_k is not None:
                        selector = SelectKBest(f_classif, k=sel_k)
                        X_tr = selector.fit_transform(X_tr, y_tr)
                        X_te = selector.transform(X_te)

                    if model_name == "SVM (RBF)":
                        from sklearn.svm import SVC
                        clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                                  class_weight="balanced", probability=True, random_state=42)
                    else:
                        clf = type(model)(**model.get_params())

                    clf.fit(X_tr, y_tr)
                    proba = clf.predict_proba(X_te)[:, 1]
                    fpr, tpr, _ = roc_curve(y_te, proba)
                    fold_aucs.append(auc(fpr, tpr))

                mean_auc = np.mean(fold_aucs)

                # Final model on all data
                X_final = X_sc
                if sel_k is not None:
                    final_selector = SelectKBest(f_classif, k=sel_k)
                    X_final = final_selector.fit_transform(X_sc, y)
                    selected_mask = final_selector.get_support()
                    final_features = [features[i] for i in range(len(features)) if selected_mask[i]]
                else:
                    final_features = features

                if model_name == "SVM (RBF)":
                    from sklearn.svm import SVC
                    final_model = SVC(kernel="rbf", C=1.0, gamma="scale",
                                      class_weight="balanced", probability=True, random_state=42)
                else:
                    final_model = type(model)(**model.get_params())

                final_model.fit(X_final, y)
                proba_all = final_model.predict_proba(X_final)[:, 1]
                fpr_all, tpr_all, _ = roc_curve(y, proba_all)

                if hasattr(final_model, "coef_"):
                    importances = final_model.coef_[0]
                elif hasattr(final_model, "feature_importances_"):
                    importances = final_model.feature_importances_
                else:
                    importances = np.zeros(len(final_features))

                results[key] = {
                    "mean_auc": mean_auc,
                    "std_auc": np.std(fold_aucs),
                    "fpr": fpr_all, "tpr": tpr_all,
                    "importances": importances,
                    "feature_names": final_features,
                    "model": final_model,
                }

    results["_meta"] = {
        "n_early": int((y == 0).sum()),
        "n_late": int((y == 1).sum()),
        "labels": ["Early (I/II)", "Late (III/IV)"],
    }
    return results


# ---------- Task C: Risk Stratification ----------

def train_risk_model(merged_df: pd.DataFrame, gene_cols: list[str],
                     top_k_genes: int = 50, penalizer: float = 0.5):
    """
    Cox PH-based risk stratification using expression + clinical features.
    Returns fitted model, risk scores, and evaluation metrics.
    """
    df = merged_df.dropna(subset=["os_time", "os_event"]).copy()
    df = df[df["os_time"] > 0]

    # Select top genes by univariate Cox p-value
    gene_pvals = {}
    for gene in gene_cols:
        if gene not in df.columns:
            continue
        sub = df[["os_time", "os_event", gene]].dropna()
        if len(sub) < 20 or sub[gene].std() < 1e-6:
            continue
        try:
            cph = CoxPHFitter(penalizer=1.0)
            cph.fit(sub, duration_col="os_time", event_col="os_event")
            gene_pvals[gene] = cph.summary["p"].values[0]
        except Exception:
            continue

    if len(gene_pvals) == 0:
        return None

    top_genes = sorted(gene_pvals, key=gene_pvals.get)[:top_k_genes]

    # Clinical features
    clinical_feats = []
    if "age" in df.columns and df["age"].notna().sum() > 20:
        clinical_feats.append("age")
    if "stage_numeric" in df.columns and df["stage_numeric"].notna().sum() > 20:
        clinical_feats.append("stage_numeric")
    if "node_positive" in df.columns and df["node_positive"].notna().sum() > 20:
        clinical_feats.append("node_positive")
    if "is_metastatic" in df.columns and df["is_metastatic"].notna().sum() > 20:
        clinical_feats.append("is_metastatic")

    all_features = top_genes + clinical_feats
    cols_needed = all_features + ["os_time", "os_event", "sample_id"]
    cols_available = [c for c in cols_needed if c in df.columns]
    sub = df[cols_available].dropna()

    if len(sub) < 30:
        return None

    # Standardize continuous features — save stats for inference
    feature_stats = {}  # {feat: (mean, std)}
    for feat in all_features:
        if feat in sub.columns and sub[feat].nunique() > 5:
            feat_mean = sub[feat].mean()
            feat_std = sub[feat].std() + 1e-8
            feature_stats[feat] = (float(feat_mean), float(feat_std))
            sub[feat] = (sub[feat] - feat_mean) / feat_std

    # Train/test split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(sub, test_size=0.3, random_state=42)

    fit_cols = [f for f in all_features if f in sub.columns] + ["os_time", "os_event"]

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(train_df[fit_cols], duration_col="os_time", event_col="os_event")
    except Exception:
        # Increase penalizer on convergence failure
        cph = CoxPHFitter(penalizer=penalizer * 5)
        cph.fit(train_df[fit_cols], duration_col="os_time", event_col="os_event")

    # Evaluate on test set
    c_index_train = cph.concordance_index_
    feature_cols = [f for f in all_features if f in sub.columns]
    test_risk = cph.predict_partial_hazard(test_df[fit_cols])
    test_df = test_df.copy()
    test_df["risk_score"] = test_risk.values

    # Risk tertiles
    try:
        test_df["risk_group"] = pd.qcut(test_df["risk_score"], q=3, labels=["Low", "Medium", "High"])
    except ValueError:
        test_df["risk_group"] = pd.cut(test_df["risk_score"], bins=3, labels=["Low", "Medium", "High"])

    return {
        "model": cph,
        "c_index_train": c_index_train,
        "test_df": test_df,
        "train_df": train_df,
        "feature_names": feature_cols,
        "feature_stats": feature_stats,  # {feat: (mean, std)} for standardizing new inputs
        "gene_features": top_genes,
        "clinical_features": clinical_feats,
        "gene_pvals": gene_pvals,
    }
