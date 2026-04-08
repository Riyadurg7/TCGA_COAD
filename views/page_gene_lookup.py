import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.preprocessing import get_tumor_normal_split
from utils.plotting import apply_theme, COLORS, THEME


def render(expr: pd.DataFrame, clinical: pd.DataFrame, survival: pd.DataFrame):
    st.title("Gene Lookup Tool")
    st.caption("Search for any gene — get its expression profile, differential expression stats, and survival impact.")

    # ── Gene Search ───────────────────────────────────────
    all_genes = sorted(expr.index.tolist())

    gene_input = st.text_input(
        "Type a gene symbol (e.g. TP53, KRAS, APC, MLH1, BRAF)",
        key="gene_lookup_input"
    )

    # Fuzzy match suggestions
    if gene_input:
        matches = [g for g in all_genes if gene_input.upper() in g.upper()]
        if len(matches) == 0:
            st.warning(f"No genes found matching '{gene_input}'. Try a different name.")
            return
        elif len(matches) > 50:
            st.info(f"{len(matches)} matches found. Showing top 20 — type more characters to narrow down.")
            matches = matches[:20]

        selected_gene = st.selectbox("Select gene", matches, key="gene_lookup_select")
    else:
        st.info("Start typing a gene name above.")
        return

    if selected_gene not in expr.index:
        st.error(f"Gene {selected_gene} not found in expression matrix.")
        return

    st.markdown(f"### Analysis for **{selected_gene}**")
    st.markdown("---")

    gene_expr = expr.loc[selected_gene]
    tumor_cols = [c for c in expr.columns if c.endswith("01A") or c.endswith("01B")]
    normal_cols = [c for c in expr.columns if c.endswith("11A") or c.endswith("11B")]

    tumor_vals = gene_expr[tumor_cols].dropna().values.astype(float)
    normal_vals = gene_expr[normal_cols].dropna().values.astype(float)

    # ── 1. Expression Boxplot ─────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=normal_vals, name="Normal", marker_color=COLORS["primary"],
            boxmean="sd"
        ))
        fig.add_trace(go.Box(
            y=tumor_vals, name="Tumor", marker_color=COLORS["secondary"],
            boxmean="sd"
        ))
        fig.update_layout(
            title=f"{selected_gene} Expression: Tumor vs Normal",
            yaxis_title="log2 Expression", showlegend=False,
            **THEME, height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # DE stats for this gene
        from scipy.stats import ttest_ind
        if len(tumor_vals) >= 2 and len(normal_vals) >= 2:
            t_stat, p_val = ttest_ind(tumor_vals, normal_vals, equal_var=False)
            logfc = tumor_vals.mean() - normal_vals.mean()

            st.markdown("#### Differential Expression")
            st.metric("log2 Fold Change", f"{logfc:.3f}")
            st.metric("p-value", f"{p_val:.2e}")
            st.metric("Mean (Tumor)", f"{tumor_vals.mean():.2f}")
            st.metric("Mean (Normal)", f"{normal_vals.mean():.2f}")

            if abs(logfc) > 1 and p_val < 0.05:
                direction = "UP-regulated" if logfc > 0 else "DOWN-regulated"
                st.success(f"Significantly {direction} in tumors")
            else:
                st.info("Not significantly differentially expressed")
        else:
            st.warning("Insufficient samples for statistical test.")

    # ── 2. Expression by Stage ────────────────────────────
    st.markdown("---")
    st.subheader(f"{selected_gene} Expression by Clinical Variables")

    merged = clinical.merge(
        pd.DataFrame({"sample_id": gene_expr.index, "expression": gene_expr.values}),
        on="sample_id", how="inner"
    )
    tumor_merged = merged[merged["sample_type"].str.contains("Tumor", case=False, na=False)]

    col1, col2 = st.columns(2)

    with col1:
        if "stage_coarse" in tumor_merged.columns and tumor_merged["stage_coarse"].notna().sum() > 10:
            fig = px.box(tumor_merged.dropna(subset=["stage_coarse"]),
                         x="stage_coarse", y="expression", color="stage_coarse",
                         color_discrete_sequence=[COLORS["primary"], COLORS["success"],
                                                  COLORS["warning"], COLORS["secondary"]],
                         title=f"{selected_gene} by AJCC Stage",
                         category_orders={"stage_coarse": ["Stage I", "Stage II", "Stage III", "Stage IV"]})
            apply_theme(fig)
            fig.update_layout(height=350, showlegend=False, yaxis_title="Expression")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "colon_side" in tumor_merged.columns and tumor_merged["colon_side"].notna().sum() > 10:
            fig = px.box(tumor_merged.dropna(subset=["colon_side"]),
                         x="colon_side", y="expression", color="colon_side",
                         color_discrete_map={"Right colon": COLORS["primary"],
                                             "Left colon": COLORS["secondary"],
                                             "Other/NOS": COLORS["muted"]},
                         title=f"{selected_gene} by Colon Side")
            apply_theme(fig)
            fig.update_layout(height=350, showlegend=False, yaxis_title="Expression")
            st.plotly_chart(fig, use_container_width=True)

    # ── 3. Survival Impact ────────────────────────────────
    st.markdown("---")
    st.subheader(f"Survival Impact of {selected_gene}")

    surv_merged = tumor_merged.merge(survival, left_on="sample_id", right_on="sample", how="inner")
    surv_merged = surv_merged.dropna(subset=["os_time", "os_event", "expression"])

    if len(surv_merged) >= 20:
        median_expr = surv_merged["expression"].median()
        surv_merged["expr_group"] = surv_merged["expression"].apply(
            lambda x: f"High {selected_gene}" if x >= median_expr else f"Low {selected_gene}"
        )

        from utils.survival_utils import km_plot_stratified
        fig, km_stats = km_plot_stratified(
            surv_merged, "expr_group", time_col="os_time", event_col="os_event",
            title=f"Survival by {selected_gene} Expression (median split)"
        )
        st.plotly_chart(fig, use_container_width=True)

        if "logrank_p" in km_stats:
            p = km_stats["logrank_p"]
            if p < 0.05:
                st.success(f"Log-rank p = {p:.4f} — {selected_gene} expression significantly stratifies survival")
            else:
                st.info(f"Log-rank p = {p:.4f} — no significant survival difference")
    else:
        st.warning("Insufficient samples with survival data for this gene.")

    # ── 4. Expression Distribution ────────────────────────
    st.markdown("---")
    with st.expander("Full Expression Distribution"):
        fig = px.histogram(
            pd.DataFrame({
                "Expression": np.concatenate([tumor_vals, normal_vals]),
                "Type": ["Tumor"] * len(tumor_vals) + ["Normal"] * len(normal_vals)
            }),
            x="Expression", color="Type", barmode="overlay", nbins=40,
            color_discrete_map={"Tumor": COLORS["secondary"], "Normal": COLORS["primary"]},
            title=f"{selected_gene} Expression Distribution", opacity=0.7
        )
        apply_theme(fig)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
