import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from utils.preprocessing import filter_by_variance, get_tumor_normal_split
from utils.de_analysis import run_de
from utils.plotting import make_ma_plot, make_pca_scatter, make_scree_plot, make_heatmap, COLORS, apply_theme

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def render(expr: pd.DataFrame, clinical: pd.DataFrame, survival: pd.DataFrame):
    st.title("Gene Expression Analysis")

    # Sidebar controls for this page
    n_genes = st.sidebar.slider("Top variable genes", 1000, 10000, 5000, step=1000,
                                 key="expr_n_genes")

    # Filter by variance
    expr_filt = filter_by_variance(expr, top_n=n_genes)

    tab_de, tab_pca, tab_corr = st.tabs(["Differential Expression", "PCA", "Correlation Heatmap"])

    # ── Tab 1: Differential Expression ────────────────────
    with tab_de:
        st.subheader("Tumor vs Normal — Differential Expression")

        if "de_results" not in st.session_state:
            # Try loading pre-computed DE results
            de_path = os.path.join(MODELS_DIR, "de_results.csv")
            if os.path.exists(de_path):
                de_df = pd.read_csv(de_path)
                st.session_state["de_results"] = de_df
            else:
                with st.spinner("Running differential expression analysis..."):
                    tumor_df, normal_df = get_tumor_normal_split(expr_filt)
                    if normal_df.shape[1] < 2:
                        st.warning("Insufficient normal samples for DE analysis.")
                        return
                    de_df = run_de(tumor_df, normal_df)
                    st.session_state["de_results"] = de_df
        else:
            de_df = st.session_state["de_results"]

        n_sig = de_df["significant"].sum()
        n_up = ((de_df["significant"]) & (de_df["logFC"] > 0)).sum()
        n_down = ((de_df["significant"]) & (de_df["logFC"] < 0)).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Significant DEGs", n_sig)
        c2.metric("Up-regulated", n_up)
        c3.metric("Down-regulated", n_down)

        fig = make_ma_plot(de_df)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Top DE Genes Table"):
            display_df = de_df.head(100)[["gene", "logFC", "avg_expr", "pvalue", "padj", "significant"]]
            st.dataframe(display_df, use_container_width=True)

            csv = display_df.to_csv(index=False)
            st.download_button("Download DE Results", csv, "de_results.csv", "text/csv")

    # ── Tab 2: PCA ────────────────────────────────────────
    with tab_pca:
        st.subheader("Principal Component Analysis")

        color_by = st.selectbox("Color by", ["Sample type", "Stage", "Gender", "Colon side"],
                                key="pca_color")

        with st.spinner("Computing PCA..."):
            # Transpose and standardize
            X = expr_filt.T.values  # samples x genes
            X = np.nan_to_num(X, nan=0.0)

            pca = PCA(n_components=min(20, X.shape[0], X.shape[1]))
            pcs = pca.fit_transform(X)

            pca_df = pd.DataFrame({
                "PC1": pcs[:, 0], "PC2": pcs[:, 1],
                "sample_id": expr_filt.columns.tolist(),
            })

            # Map sample IDs to clinical data
            col_map = {
                "Sample type": "sample_type",
                "Stage": "stage_coarse",
                "Gender": "gender",
                "Colon side": "colon_side",
            }
            clin_col = col_map[color_by]

            # Match by sample_id
            clin_lookup = clinical.set_index("sample_id")[clin_col] if clin_col in clinical.columns else pd.Series(dtype=str)
            pca_df[color_by] = pca_df["sample_id"].map(clin_lookup).fillna("Unknown")

            # Handle sample_type from barcode if clinical match fails
            if color_by == "Sample type" and (pca_df[color_by] == "Unknown").sum() > len(pca_df) * 0.5:
                pca_df[color_by] = pca_df["sample_id"].apply(
                    lambda s: "Tumor" if s.endswith("01A") or s.endswith("01B")
                    else ("Normal" if s.endswith("11A") or s.endswith("11B") else "Other")
                )

            var_explained = pca.explained_variance_ratio_

            fig = make_pca_scatter(pca_df, color_by, var_explained=(var_explained[0]*100, var_explained[1]*100))
            st.plotly_chart(fig, use_container_width=True)

            fig_scree = make_scree_plot(pca.explained_variance_ratio_)
            st.plotly_chart(fig_scree, use_container_width=True)

    # ── Tab 3: Correlation Heatmap ────────────────────────
    with tab_corr:
        st.subheader("Gene Correlation Heatmap")

        de_df = st.session_state.get("de_results")
        if de_df is not None:
            top_genes = de_df.head(50)["gene"].tolist()
            available = [g for g in top_genes if g in expr_filt.index]
        else:
            available = expr_filt.index[:50].tolist()

        selected_genes = st.multiselect(
            "Select genes (10-30 recommended)",
            available, default=available[:15], key="corr_genes"
        )

        if len(selected_genes) >= 2:
            # Compute correlation on tumor samples only
            tumor_cols = [c for c in expr_filt.columns if c.endswith("01A") or c.endswith("01B")]
            sub = expr_filt.loc[expr_filt.index.isin(selected_genes), tumor_cols]
            corr = sub.T.corr()

            fig = make_heatmap(corr, title=f"Pairwise Correlation ({len(selected_genes)} genes, tumor samples)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 genes to view the correlation heatmap.")
