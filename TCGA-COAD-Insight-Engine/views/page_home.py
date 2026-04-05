import streamlit as st


def render():
    # ── Hero ─────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center; font-size:2.6rem; margin-bottom:0;'>"
        "TCGA-COAD Insight Engine</h1>"
        "<p style='text-align:center; color:#6c7086; font-size:1.1rem; margin-top:4px;'>"
        "Interactive analysis of colon adenocarcinoma using real patient data from "
        "The Cancer Genome Atlas</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Quick Start ──────────────────────────────────────
    st.subheader("Quick Start")
    st.markdown(
        "Use the **sidebar** on the left to navigate between pages. "
        "Each page focuses on a different aspect of the data. "
        "Everything is interactive — hover over charts for details, "
        "adjust sliders, and select variables from dropdowns."
    )

    # Page guide as a clean grid
    pages = [
        ("Data Overview", "Bird's-eye view of the dataset — sample counts, demographics, "
         "cancer stage distribution, data quality, and tumor locations."),
        ("Gene Expression", "Find genes altered in cancer. Includes differential expression "
         "(tumor vs normal), PCA dimensionality reduction, and gene correlation heatmap."),
        ("Clinical Explorer", "Pick any two clinical variables and instantly see their "
         "relationship with the right chart and statistical test, auto-selected for you."),
        ("Survival Analysis", "Kaplan-Meier survival curves, Cox regression to identify "
         "risk factors, and patient risk group stratification."),
        ("ML Prediction Lab", "Three pre-trained models: tumor/normal classification, "
         "cancer stage prediction, and survival risk scoring."),
        ("Gene Lookup", "Search any gene by name — get its expression profile, "
         "tumor vs normal comparison, stage breakdown, and survival impact."),
        ("Patient Risk Calculator", "Enter a patient's clinical features and gene values "
         "to get a predicted risk group with a breakdown of contributing factors."),
    ]

    for i in range(0, len(pages), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(pages):
                name, desc = pages[i + j]
                col.markdown(
                    f"<div style='background:#313244; border-radius:10px; padding:16px; "
                    f"border-left:3px solid #89b4fa; margin-bottom:8px; min-height:110px;'>"
                    f"<strong style='color:#89b4fa;'>{name}</strong><br>"
                    f"<span style='color:#a6adc8; font-size:0.9rem;'>{desc}</span></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Key Concepts ─────────────────────────────────────
    st.subheader("Key Concepts")
    st.caption("Expand any section below to learn the basics.")

    with st.expander("What is gene expression?"):
        st.markdown(
            "Every cell has the same DNA, but not every gene is active everywhere. "
            "**Gene expression** measures how active a gene is by detecting the amount "
            "of RNA it produces.\n\n"
            "- **High expression** = gene is very active\n"
            "- **Low expression** = gene is quiet\n\n"
            "In cancer, many genes become abnormally active or silent compared to "
            "healthy tissue. Detecting these changes reveals what drives the tumor."
        )

    with st.expander("What is TCGA-COAD?"):
        st.markdown(
            "**TCGA** (The Cancer Genome Atlas) is a landmark public project that "
            "collected genetic and clinical data from thousands of cancer patients "
            "across 33 cancer types.\n\n"
            "**COAD** = **Colon Adenocarcinoma**, the most common type of colon cancer. "
            "This dataset has **514 tissue samples** (tumor + matched normal), "
            "**60,660 genes** measured per sample, and clinical records for ~500 patients."
        )

    with st.expander("Tumor vs Normal samples"):
        st.markdown(
            "When a tumor is surgically removed, doctors also take a small piece of "
            "nearby **healthy tissue** from the same patient. Comparing gene expression "
            "in tumor vs normal reveals which genes are disrupted by cancer.\n\n"
            "Samples are identified by their TCGA barcode suffix:\n"
            "- Ending in **01A/01B** = Tumor\n"
            "- Ending in **11A/11B** = Normal"
        )

    with st.expander("Cancer stages (I - IV)"):
        st.markdown(
            "| Stage | Meaning |\n"
            "|-------|--------|\n"
            "| **I** | Small, confined to the inner colon lining |\n"
            "| **II** | Grown through the colon wall, no lymph node spread |\n"
            "| **III** | Spread to nearby lymph nodes |\n"
            "| **IV** | Spread to distant organs (metastatic) |\n\n"
            "Higher stage = more advanced = generally worse prognosis."
        )

    with st.expander("What is a p-value?"):
        st.markdown(
            "A p-value tells you the probability your result happened by **random chance**.\n\n"
            "- **p < 0.05** = statistically significant (less than 5% chance it's random)\n"
            "- **p > 0.05** = not significant (could be noise)\n\n"
            "Example: if comparing survival between two groups gives p = 0.003, "
            "there's only a 0.3% chance the difference is just luck."
        )

    with st.expander("What is log2 fold change?"):
        st.markdown(
            "When comparing gene expression between tumor and normal:\n\n"
            "- **logFC > 0** = gene is **up-regulated** (more active in tumor)\n"
            "- **logFC < 0** = gene is **down-regulated** (less active in tumor)\n"
            "- **|logFC| > 1** = at least a **2-fold** change (biologically meaningful)\n\n"
            "A logFC of +2 means the gene is 4x more active in tumors; "
            "-2 means 4x less active."
        )

    with st.expander("What is a Kaplan-Meier curve?"):
        st.markdown(
            "The standard way to visualize survival in medicine. It's a step-function where:\n\n"
            "- **X-axis** = time (days after diagnosis)\n"
            "- **Y-axis** = probability of being alive (starts at 100%)\n"
            "- The line drops each time a patient dies\n"
            "- Steeper drops = more deaths in that period\n\n"
            "When comparing groups, a **log-rank test** determines if their "
            "survival curves are significantly different."
        )

    with st.expander("What is AUC / ROC?"):
        st.markdown(
            "**ROC curve** plots True Positive Rate vs False Positive Rate at every "
            "decision threshold. **AUC** (Area Under the Curve) summarizes it:\n\n"
            "- **0.5** = random guessing (worthless)\n"
            "- **0.7 - 0.8** = acceptable\n"
            "- **0.9+** = excellent\n"
            "- **1.0** = perfect\n\n"
            "Used throughout the ML Prediction Lab to evaluate model quality."
        )

    with st.expander("What is a hazard ratio?"):
        st.markdown(
            "From Cox regression in survival analysis:\n\n"
            "- **HR = 1** = no effect on survival\n"
            "- **HR > 1** = increases risk of death (e.g., HR=2.0 means 2x the risk)\n"
            "- **HR < 1** = protective (decreases risk)\n\n"
            "Used in the Survival Analysis page and Patient Risk Calculator."
        )

    st.markdown("---")

    # ── Dataset at a glance ──────────────────────────────
    st.subheader("Dataset at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", "514")
    col2.metric("Genes", "60,660")
    col3.metric("Patients", "~500")
    col4.metric("Source", "UCSC Xena")

    st.markdown(
        "<p style='color:#6c7086; font-size:0.85rem; text-align:center; margin-top:30px;'>"
        "Built with Streamlit, scikit-learn, lifelines, and Plotly. "
        "Data from TCGA-COAD via UCSC Xena. For research and education only.</p>",
        unsafe_allow_html=True,
    )
