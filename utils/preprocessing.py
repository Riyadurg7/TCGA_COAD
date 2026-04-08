import pandas as pd
import numpy as np
import ast
from utils.gene_mapping import map_ensembl_to_symbol, strip_version


def map_genes_to_symbols(expr: pd.DataFrame) -> pd.DataFrame:
    mapping = map_ensembl_to_symbol(expr.index.tolist())
    expr = expr.copy()
    expr.index = [mapping.get(g, strip_version(g)) for g in expr.index]
    # drop duplicates by keeping the one with highest variance
    expr["_var"] = expr.var(axis=1)
    expr = expr.sort_values("_var", ascending=False)
    expr = expr[~expr.index.duplicated(keep="first")]
    expr = expr.drop(columns=["_var"])
    return expr


def filter_by_variance(expr: pd.DataFrame, top_n: int = 5000) -> pd.DataFrame:
    median = expr.median(axis=1)
    mad = (expr.sub(median, axis=0)).abs().median(axis=1)
    top_genes = mad.nlargest(top_n).index
    return expr.loc[top_genes]


def clean_clinical(clinical: pd.DataFrame) -> pd.DataFrame:
    df = clinical.copy()

    # Coarsen AJCC stage
    def coarsen_stage(s):
        if pd.isna(s) or s in ("not reported", "Not Reported", ""):
            return np.nan
        s = str(s).strip()
        for stage, label in [("IV", "Stage IV"), ("III", "Stage III"), ("II", "Stage II"), ("I", "Stage I")]:
            if stage in s:
                return label
        return np.nan

    if "stage" in df.columns:
        df["stage_coarse"] = df["stage"].apply(coarsen_stage)
        stage_map = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
        df["stage_numeric"] = df["stage_coarse"].map(stage_map)

    # Binary metastasis
    if "tumor_m" in df.columns:
        df["is_metastatic"] = df["tumor_m"].apply(
            lambda x: 1 if isinstance(x, str) and x.startswith("M1") else 0
        )

    # Node positive
    if "tumor_n" in df.columns:
        df["node_positive"] = df["tumor_n"].apply(
            lambda x: 0 if isinstance(x, str) and x == "N0" else (1 if isinstance(x, str) and x.startswith("N") else np.nan)
        )

    # Age numeric
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Age group
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 50, 65, 80, 120],
            labels=["<50", "50-65", "65-80", ">80"]
        )

    # Gender cleanup
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.lower().str.strip()
        df["gender"] = df["gender"].replace({"not reported": np.nan})

    # Anatomical region (right vs left colon)
    right_sites = ["Cecum", "Ascending colon", "Hepatic flexure of colon", "Transverse colon"]
    left_sites = ["Splenic flexure of colon", "Descending colon", "Sigmoid colon", "Rectosigmoid junction"]
    if "anatomical_site" in df.columns:
        def classify_side(s):
            if pd.isna(s):
                return np.nan
            s = str(s).strip()
            if any(r.lower() in s.lower() for r in right_sites):
                return "Right colon"
            if any(l.lower() in s.lower() for l in left_sites):
                return "Left colon"
            return "Other/NOS"
        df["colon_side"] = df["anatomical_site"].apply(classify_side)

    # Parse treatment types
    if "treatment_types" in df.columns:
        def parse_list_col(val):
            if pd.isna(val) or val == "":
                return []
            try:
                return ast.literal_eval(str(val))
            except (ValueError, SyntaxError):
                return [str(val)]

        df["treatment_list"] = df["treatment_types"].apply(parse_list_col)
        df["received_chemo"] = df["treatment_list"].apply(
            lambda x: 1 if any("Pharmaceutical" in t for t in x) else 0
        )
        df["received_radiation"] = df["treatment_list"].apply(
            lambda x: 1 if any("Radiation" in t for t in x) else 0
        )

    return df


def merge_all(expr: pd.DataFrame, clinical: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    # Transpose expression: samples as rows
    expr_t = expr.T
    expr_t.index.name = "sample_id"
    expr_t = expr_t.reset_index()

    # Merge clinical
    merged = clinical.merge(survival, left_on="sample_id", right_on="sample", how="inner")
    merged = merged.merge(expr_t, on="sample_id", how="inner")

    return merged


def get_tumor_normal_split(expr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split expression columns into tumor and normal based on TCGA barcode suffix."""
    tumor_cols = [c for c in expr.columns if c.endswith("01A") or c.endswith("01B")]
    normal_cols = [c for c in expr.columns if c.endswith("11A") or c.endswith("11B")]
    return expr[tumor_cols], expr[normal_cols]
