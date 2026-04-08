import pandas as pd
import numpy as np
from scipy import stats


def run_de(tumor_df: pd.DataFrame, normal_df: pd.DataFrame) -> pd.DataFrame:
    """Run differential expression analysis: tumor vs normal.

    Both inputs should have genes as rows, samples as columns.
    Returns DataFrame with gene, logFC, pvalue, padj, significant columns.
    """
    genes = tumor_df.index.intersection(normal_df.index)
    tumor = tumor_df.loc[genes]
    normal = normal_df.loc[genes]

    tumor_mean = tumor.mean(axis=1)
    normal_mean = normal.mean(axis=1)
    logfc = tumor_mean - normal_mean  # already log-scale, so difference = logFC

    pvalues = []
    for gene in genes:
        t_vals = tumor.loc[gene].dropna().values
        n_vals = normal.loc[gene].dropna().values
        if len(t_vals) < 2 or len(n_vals) < 2:
            pvalues.append(1.0)
            continue
        _, p = stats.ttest_ind(t_vals, n_vals, equal_var=False)
        pvalues.append(p if not np.isnan(p) else 1.0)

    pvalues = np.array(pvalues)

    # Benjamini-Hochberg FDR correction
    padj = _bh_correction(pvalues)

    result = pd.DataFrame({
        "gene": genes,
        "logFC": logfc.values,
        "avg_expr": ((tumor_mean + normal_mean) / 2).values,
        "pvalue": pvalues,
        "padj": padj,
        "neg_log10_p": -np.log10(np.clip(pvalues, 1e-300, 1)),
    })
    result["significant"] = (result["padj"] < 0.05) & (result["logFC"].abs() > 1)
    result = result.sort_values("padj").reset_index(drop=True)
    return result


def _bh_correction(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    ranked = np.argsort(pvalues)
    padj = np.ones(n)
    for i, rank_idx in enumerate(ranked):
        padj[rank_idx] = pvalues[rank_idx] * n / (i + 1)
    # ensure monotonicity
    padj_sorted_idx = np.argsort(pvalues)
    padj_sorted = padj[padj_sorted_idx]
    for i in range(len(padj_sorted) - 2, -1, -1):
        padj_sorted[i] = min(padj_sorted[i], padj_sorted[i + 1])
    padj[padj_sorted_idx] = padj_sorted
    padj = np.clip(padj, 0, 1)
    return padj
