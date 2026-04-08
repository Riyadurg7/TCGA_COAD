import streamlit as st
import os

st.set_page_config(
    page_title="TCGA-COAD Insight Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("TCGA-COAD Insight Engine")
st.sidebar.caption("Colon Adenocarcinoma Analysis & Prediction")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Overview", "Gene Expression", "Clinical Explorer", "Survival Analysis",
     "ML Prediction Lab", "Gene Lookup", "Patient Risk Calculator"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** TCGA-COAD (UCSC Xena)  \n"
    "**Samples:** 514 (tumor + normal)  \n"
    "**Genes:** 60,660"
)

# ── Load Data (cached, lazy) ─────────────────────────────
def _get_data():
    from utils.data_loader import load_expression, load_clinical, load_survival
    from utils.preprocessing import map_genes_to_symbols, clean_clinical

    @st.cache_data(show_spinner="Preparing data...")
    def prepare_data():
        expr_raw = load_expression()
        clinical = load_clinical()
        survival = load_survival()
        expr = map_genes_to_symbols(expr_raw)
        clinical = clean_clinical(clinical)
        return expr, clinical, survival

    if "data_loaded" not in st.session_state:
        expr, clinical, survival = prepare_data()
        st.session_state["expr"] = expr
        st.session_state["clinical"] = clinical
        st.session_state["survival"] = survival
        st.session_state["data_loaded"] = True
    return (st.session_state["expr"],
            st.session_state["clinical"],
            st.session_state["survival"])

# ── Page Routing ─────────────────────────────────────────
if page == "Home":
    from views.page_home import render
    render()
elif page == "Data Overview":
    expr, clinical, survival = _get_data()
    from views.page_overview import render
    render(expr, clinical, survival)
elif page == "Gene Expression":
    expr, clinical, survival = _get_data()
    from views.page_expression import render
    render(expr, clinical, survival)
elif page == "Clinical Explorer":
    expr, clinical, survival = _get_data()
    from views.page_clinical import render
    render(clinical, survival)
elif page == "Survival Analysis":
    expr, clinical, survival = _get_data()
    from views.page_survival import render
    render(expr, clinical, survival)
elif page == "ML Prediction Lab":
    expr, clinical, survival = _get_data()
    from views.page_prediction import render
    render(expr, clinical, survival)
elif page == "Gene Lookup":
    expr, clinical, survival = _get_data()
    from views.page_gene_lookup import render
    render(expr, clinical, survival)
elif page == "Patient Risk Calculator":
    expr, clinical, survival = _get_data()
    from views.page_risk_calculator import render
    render(expr, clinical, survival)
