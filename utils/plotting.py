import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,17,27,0.95)",
    font=dict(color="#cdd6f4", family="monospace"),
)

COLORS = {
    "primary": "#89b4fa",
    "secondary": "#f38ba8",
    "success": "#a6e3a1",
    "warning": "#fab387",
    "accent": "#cba6f7",
    "muted": "#6c7086",
    "text": "#cdd6f4",
    "surface": "#313244",
}


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**THEME)
    return fig


def make_donut(labels, values, title, colors=None):
    if colors is None:
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"],
                  COLORS["warning"], COLORS["accent"], COLORS["muted"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors[:len(labels)]),
        textinfo="label+percent", textfont=dict(size=11)
    ))
    fig.update_layout(title=title, showlegend=False, **THEME, height=350, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def make_ma_plot(de_df: pd.DataFrame, top_n_labels: int = 12) -> go.Figure:
    sig_up = de_df[(de_df["significant"]) & (de_df["logFC"] > 0)]
    sig_down = de_df[(de_df["significant"]) & (de_df["logFC"] < 0)]
    not_sig = de_df[~de_df["significant"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=not_sig["avg_expr"], y=not_sig["logFC"], mode="markers",
        marker=dict(color=COLORS["muted"], size=3, opacity=0.4),
        name="Not significant", hovertext=not_sig["gene"], hoverinfo="text+x+y"
    ))
    fig.add_trace(go.Scatter(
        x=sig_up["avg_expr"], y=sig_up["logFC"], mode="markers",
        marker=dict(color="#ff6b6b", size=5, opacity=0.7),
        name="Up-regulated", hovertext=sig_up["gene"], hoverinfo="text+x+y"
    ))
    fig.add_trace(go.Scatter(
        x=sig_down["avg_expr"], y=sig_down["logFC"], mode="markers",
        marker=dict(color="#4ecdc4", size=5, opacity=0.7),
        name="Down-regulated", hovertext=sig_down["gene"], hoverinfo="text+x+y"
    ))

    # Label top genes
    top = de_df.nsmallest(top_n_labels, "padj")
    for _, row in top.iterrows():
        fig.add_annotation(
            x=row["avg_expr"], y=row["logFC"], text=row["gene"],
            showarrow=True, arrowhead=0, arrowcolor=COLORS["text"],
            font=dict(size=9, color=COLORS["text"]), ax=20, ay=-15
        )

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.5)
    fig.add_hline(y=1, line_dash="dot", line_color=COLORS["muted"], opacity=0.3)
    fig.add_hline(y=-1, line_dash="dot", line_color=COLORS["muted"], opacity=0.3)

    fig.update_layout(
        title="MA Plot — Differential Expression",
        xaxis_title="Average Expression", yaxis_title="log2 Fold Change",
        **THEME, height=550
    )
    return fig


def make_pca_scatter(pca_df: pd.DataFrame, color_col: str, pc_x="PC1", pc_y="PC2",
                     var_explained: tuple = (0, 0)) -> go.Figure:
    fig = px.scatter(
        pca_df, x=pc_x, y=pc_y, color=color_col,
        color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["success"],
                                 COLORS["warning"], COLORS["accent"], COLORS["muted"]],
        hover_data=["sample_id"] if "sample_id" in pca_df.columns else None,
    )
    fig.update_layout(
        title=f"PCA — colored by {color_col}",
        xaxis_title=f"{pc_x} ({var_explained[0]:.1f}% variance)",
        yaxis_title=f"{pc_y} ({var_explained[1]:.1f}% variance)",
        **THEME, height=500
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    return fig


def make_scree_plot(var_explained: np.ndarray, n: int = 20) -> go.Figure:
    var_exp = var_explained[:n] * 100
    fig = go.Figure(go.Bar(
        x=[f"PC{i+1}" for i in range(len(var_exp))], y=var_exp,
        marker_color=COLORS["primary"]
    ))
    fig.update_layout(
        title="Scree Plot — Variance Explained",
        xaxis_title="Principal Component", yaxis_title="% Variance Explained",
        **THEME, height=350
    )
    return fig


def make_heatmap(corr_matrix: pd.DataFrame, title="Gene Correlation Heatmap") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale="RdBu_r", zmid=0, text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}", textfont=dict(size=9)
    ))
    fig.update_layout(title=title, **THEME, height=max(400, len(corr_matrix) * 25))
    return fig


def make_roc_curve(fpr_dict, tpr_dict, auc_dict, title="ROC Curve") -> go.Figure:
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=COLORS["muted"]), showlegend=False
    ))
    for i, (name, fpr) in enumerate(fpr_dict.items()):
        tpr = tpr_dict[name]
        auc = auc_dict[name]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    fig.update_layout(
        title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        **THEME, height=450, legend=dict(x=0.55, y=0.05)
    )
    return fig


def make_confusion_matrix(cm, labels, title="Confusion Matrix") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, colorscale="Blues",
        text=cm, texttemplate="%{text}", textfont=dict(size=16)
    ))
    fig.update_layout(
        title=title, xaxis_title="Predicted", yaxis_title="Actual",
        **THEME, height=350
    )
    return fig


def make_feature_importance(names, values, title="Feature Importance", top_n=20) -> go.Figure:
    idx = np.argsort(np.abs(values))[::-1][:top_n]
    names = [names[i] for i in idx]
    values = [values[i] for i in idx]
    colors = [COLORS["secondary"] if v > 0 else COLORS["primary"] for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1], orientation="h",
        marker_color=colors[::-1]
    ))
    fig.update_layout(
        title=title, xaxis_title="Importance",
        **THEME, height=max(350, top_n * 25), margin=dict(l=150)
    )
    return fig
