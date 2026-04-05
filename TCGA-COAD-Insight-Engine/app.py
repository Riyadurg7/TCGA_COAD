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
    ["Data Overview", "Gene Expression", "Clinical Explorer", "Survival Analysis",
     "ML Prediction Lab", "Gene Lookup", "Patient Risk Calculator"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** TCGA-COAD (UCSC Xena)  \n"
    "**Samples:** 514 (tumor + normal)  \n"
    "**Genes:** 60,660"
)

# ── Load Data (cached) ───────────────────────────────────
from utils.data_loader import load_expression, load_clinical, load_survival
from utils.preprocessing import map_genes_to_symbols, filter_by_variance, clean_clinical
from utils.gene_mapping import strip_version

@st.cache_data(show_spinner="Preparing data...")
def prepare_data():
    expr_raw = load_expression()
    clinical = load_clinical()
    survival = load_survival()

    # Map Ensembl IDs to gene symbols
    expr = map_genes_to_symbols(expr_raw)

    # Clean clinical data
    clinical = clean_clinical(clinical)

    return expr, clinical, survival


if "data_loaded" not in st.session_state:
    expr, clinical, survival = prepare_data()
    st.session_state["expr"] = expr
    st.session_state["clinical"] = clinical
    st.session_state["survival"] = survival
    st.session_state["data_loaded"] = True
else:
    expr = st.session_state["expr"]
    clinical = st.session_state["clinical"]
    survival = st.session_state["survival"]

# ── Page Routing ─────────────────────────────────────────
if page == "Data Overview":
    from views.page_overview import render
    render(expr, clinical, survival)
elif page == "Gene Expression":
    from views.page_expression import render
    render(expr, clinical, survival)
elif page == "Clinical Explorer":
    from views.page_clinical import render
    render(clinical, survival)
elif page == "Survival Analysis":
    from views.page_survival import render
    render(expr, clinical, survival)
elif page == "ML Prediction Lab":
    from views.page_prediction import render
    render(expr, clinical, survival)
elif page == "Gene Lookup":
    from views.page_gene_lookup import render
    render(expr, clinical, survival)
elif page == "Patient Risk Calculator":
    from views.page_risk_calculator import render
    render(expr, clinical, survival)
