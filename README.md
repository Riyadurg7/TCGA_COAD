# ColonScope

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ppb-miniproj.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)


An interactive web application for exploring **Colon Adenocarcinoma (TCGA-COAD)** data — combining gene expression analysis, clinical exploration, survival modeling, and machine learning on real patient data from The Cancer Genome Atlas.

### [Live Demo](https://ppb-miniproj.streamlit.app/)

---

### Features

- **Differential Expression** — Identify genes significantly up/down-regulated in tumor vs. normal tissue
- **PCA & Correlation** — Visualize sample clustering and gene co-expression patterns
- **Clinical Explorer** — Cross-tabulate clinical variables with auto-selected statistical tests
- **Survival Analysis** — Kaplan-Meier curves, Cox regression, and risk stratification
- **ML Prediction Lab** — Pre-trained classifiers for tumor/normal, cancer stage, and patient risk
- **Gene Lookup** — Search any gene for its full expression and survival profile
- **Patient Risk Calculator** — Input clinical + molecular features to get a predicted risk score

---

## Table of Contents

1. [What Is This App?](#what-is-this-app)
2. [Key Concepts You Need to Know](#key-concepts-you-need-to-know)
3. [The Dataset](#the-dataset)
4. [Page-by-Page Walkthrough](#page-by-page-walkthrough)
   - [Page 1: Data Overview](#page-1-data-overview)
   - [Page 2: Gene Expression Analysis](#page-2-gene-expression-analysis)
   - [Page 3: Clinical Explorer](#page-3-clinical-explorer)
   - [Page 4: Survival Analysis](#page-4-survival-analysis)
   - [Page 5: ML Prediction Lab](#page-5-ml-prediction-lab)
   - [Page 6: Gene Lookup Tool](#page-6-gene-lookup-tool)
   - [Page 7: Patient Risk Calculator](#page-7-patient-risk-calculator)
5. [Every Statistic and ML Technique Explained](#every-statistic-and-ml-technique-explained)
6. [Project Structure](#project-structure)
7. [Setup and Installation](#setup-and-installation)
8. [Tech Stack](#tech-stack)

---

## What Is This App?

Imagine you have data from **514 real colon cancer patients** — their tumor tissue samples, normal tissue samples, clinical records (age, gender, cancer stage, treatments), and survival outcomes (how long they lived, whether they died). This app lets you explore all of that interactively.

**What can you do with it?**

- See which genes are turned "up" or "down" in cancer vs. healthy tissue
- Explore whether factors like age, gender, or tumor location affect outcomes
- View survival curves showing how long patients live based on different factors
- Use machine learning models that predict whether a tissue sample is cancerous, what stage the cancer is, and how high-risk a patient is
- Look up any individual gene and see its full profile across all patients
- Enter a hypothetical patient's information and get a predicted risk score

---

## Key Concepts You Need to Know

Before diving into the pages, here are the fundamental concepts this app uses. If you already know biology/statistics, skip ahead.

### What Is Gene Expression?

Every cell in your body has the same DNA (your genes). But not every gene is "active" in every cell. **Gene expression** is a measure of how active (how much "turned on") a specific gene is. It is measured by detecting how much RNA a gene produces.

- **High expression** = the gene is very active, producing a lot of RNA
- **Low expression** = the gene is quiet, producing little RNA

In cancer, many genes become abnormally active or inactive compared to healthy tissue. Detecting these changes helps us understand what drives the cancer.

### What Is TCGA?

**The Cancer Genome Atlas (TCGA)** is a massive, publicly funded project that collected genetic and clinical data from thousands of cancer patients across 33 cancer types. This app uses the **COAD** (Colon Adenocarcinoma) subset.

### What Is TCGA-COAD?

**COAD** stands for **Colon Adenocarcinoma** — the most common type of colon cancer. It starts in the glandular cells lining the inside of the colon. The TCGA-COAD dataset contains tumor samples, matched normal tissue samples, clinical records, and survival data for ~500 patients.

### Tumor vs. Normal Samples

When doctors remove a colon tumor, they also take a small piece of nearby **healthy (normal) tissue** from the same patient. By comparing gene expression in tumor vs. normal tissue, we can find which genes are disrupted by the cancer.

- **Tumor sample** = tissue from the cancerous growth
- **Normal sample** = tissue from healthy colon of the same patient

### Cancer Staging (AJCC Stages I-IV)

Cancer staging describes how far the cancer has spread:

| Stage | What It Means | Simplified |
|-------|---------------|------------|
| **Stage I** | Cancer is small and only in the inner lining of the colon | Early, localized |
| **Stage II** | Cancer has grown through the colon wall but hasn't reached lymph nodes | Still localized |
| **Stage III** | Cancer has spread to nearby lymph nodes | Regional spread |
| **Stage IV** | Cancer has spread to distant organs (liver, lungs, etc.) | Metastatic |

Higher stage = more advanced = generally worse prognosis.

### What Is a p-value?

A **p-value** is a number that tells you how likely your result would have occurred by random chance alone.

- **p < 0.05** = "statistically significant" — there's less than a 5% chance this result is just random noise. We consider this a real finding.
- **p > 0.05** = "not significant" — the result could easily be due to chance.

Example: If comparing survival between two groups gives p = 0.003, it means there's only a 0.3% chance the difference is random — so the groups truly have different survival.

### What Is log2 Fold Change?

When comparing gene expression between tumor and normal tissue, **log2 fold change (logFC)** tells you the direction and magnitude of change:

- **logFC > 0** = gene is **UP-regulated** (more active in tumor than normal)
- **logFC < 0** = gene is **DOWN-regulated** (less active in tumor)
- **|logFC| > 1** = the gene is at least **2x** more (or less) active. This is a common threshold for "biologically meaningful" change.

The "log2" part is a mathematical transformation that makes the numbers symmetric and easier to compare. A logFC of +2 means the gene is 4x more active in tumors; a logFC of -2 means it's 4x less active.

---

## The Dataset

All data comes from [UCSC Xena](https://xenabrowser.net/), a public portal for TCGA data.

| File | What It Contains | Size |
|------|-----------------|------|
| `data/COAD.tsv` | Gene expression matrix — measures how active each of 60,660 genes is in each of the 514 samples | 60,660 rows (genes) x 514 columns (samples) |
| `data/TCGA-COAD.clinical.tsv` | Clinical records — patient demographics, cancer stage, tumor location, treatments received | 561 patients x 86 data fields |
| `data/TCGA-COAD.survival.tsv` | Survival data — how many days each patient survived, and whether they had died at last follow-up | 529 patients |
| `data/ensembl_to_symbol.csv` | A lookup table to convert technical gene IDs (like "ENSG00000141510") to human-readable names (like "TP53") | 45,045 mappings |

### How Samples Are Identified

Each sample has a **TCGA barcode** like `TCGA-A6-2671-01A`. The ending tells you what type of tissue it is:
- Ends in **01A** or **01B** = **Tumor** sample
- Ends in **11A** or **11B** = **Normal** (healthy) tissue sample

---

## Page-by-Page Walkthrough

### Page 1: Data Overview

**Purpose:** Give you a bird's-eye view of the entire dataset before doing any analysis.

**What you see:**

1. **KPI Metrics (top row of numbers)**
   - **Total Samples:** How many tissue samples are in the dataset (514)
   - **Tumor Samples:** How many are from cancerous tissue
   - **Normal Samples:** How many are from healthy tissue
   - **Genes Measured:** The total number of genes tracked (60,660)

2. **Sample Type Distribution (donut chart)**
   - A pie/donut chart showing the proportion of tumor vs. normal samples. There are far more tumor samples than normal ones, which is typical — not every patient donates a matched normal sample.

3. **AJCC Stage Distribution (donut chart)**
   - Shows how many patients are in each cancer stage (I, II, III, IV). This tells you the severity profile of the patient population. Most patients are Stage II or III.

4. **Demographics**
   - **Gender Distribution (bar chart):** How many male vs. female patients. Colon cancer affects both genders roughly equally.
   - **Age Distribution (histogram):** A chart showing the age at which patients were diagnosed. Most colon cancer patients are diagnosed between ages 55-80.

5. **Data Quality — Non-missing Rate (horizontal bar chart)**
   - Not every patient record is complete. This chart shows, for each clinical variable (gender, age, stage, etc.), what percentage of patients have that data filled in. Variables with low completeness may be unreliable for analysis.

6. **Tumor Anatomical Location**
   - Shows where in the colon the tumor was found (cecum, ascending colon, sigmoid, etc.), color-coded by **right colon** vs. **left colon**. This matters because right-sided and left-sided colon cancers behave differently at the molecular level, respond differently to treatment, and have different prognoses.

7. **Raw Data Preview**
   - Expandable sections letting you peek at the actual data tables (expression matrix, clinical data, survival data).

---

### Page 2: Gene Expression Analysis

**Purpose:** Find genes that behave differently in cancer vs. healthy tissue, reduce the data's complexity, and see how genes relate to each other.

This page has **three tabs:**

#### Tab 1: Differential Expression

**What it does:** Compares gene activity in tumor tissue vs. normal tissue for every gene, then identifies which genes are significantly altered.

**What you see:**

- **Three KPI numbers:**
  - **Significant DEGs:** Total number of "differentially expressed genes" — genes that are significantly changed in cancer
  - **Up-regulated:** Genes more active in tumors
  - **Down-regulated:** Genes less active in tumors

- **MA Plot:** A scatter plot where:
  - **X-axis** = average expression level of each gene (how active it is overall)
  - **Y-axis** = log2 fold change (how much it changed between tumor and normal)
  - **Gray dots** = genes with no significant change
  - **Red dots** = significantly UP-regulated genes (more active in cancer)
  - **Teal dots** = significantly DOWN-regulated genes (less active in cancer)
  - **Labeled dots** = the most statistically significant genes, labeled by name
  - Horizontal dashed lines at logFC = +1 and -1 mark the "biologically meaningful change" threshold

- **Top DE Genes Table:** A downloadable table of the most significantly changed genes, with their fold change, p-value, and adjusted p-value.

**Statistics used:**
- **Welch's t-test:** Compares the mean expression of each gene between tumor and normal groups. It's a version of the t-test that doesn't assume both groups have equal variance (spread). This is important because tumor samples are often more variable than normal ones.
- **Benjamini-Hochberg (BH) FDR correction:** When you test 60,000 genes, some will appear "significant" just by random chance. The BH correction adjusts p-values to control the **false discovery rate** — the expected proportion of false positives. A gene must have an adjusted p-value (padj) < 0.05 AND |logFC| > 1 to be called "significant."

#### Tab 2: PCA (Principal Component Analysis)

**What it does:** Takes the expression data for thousands of genes and compresses it down to just 2 dimensions so you can visualize how similar or different samples are from each other.

**What you see:**

- **PCA Scatter Plot:** Each dot is a patient sample. Samples that are close together have similar gene expression patterns; samples far apart are very different. You can color the dots by:
  - **Sample type** (tumor vs. normal) — usually you'll see two distinct clusters, proving tumor and normal tissue have very different gene activity
  - **Stage** — to see if early-stage and late-stage cancers look different at the molecular level
  - **Gender** or **Colon side** — to see if these factors create molecular subtypes

- **Scree Plot:** A bar chart showing how much of the data's total variability each principal component captures. PC1 captures the most, PC2 the second most, etc. If PC1 captures 40% and PC2 captures 10%, together they show 50% of the data's structure.

**How PCA works (simplified):** Imagine you have data about each sample described by 5,000 numbers (gene expression values). PCA finds the "directions" in this high-dimensional space that capture the most variation. It's like taking a 3D object and finding the best angle to photograph it so you can see the most important features in a flat 2D photo. Each "principal component" is one of these optimal directions.

#### Tab 3: Correlation Heatmap

**What it does:** Shows how pairs of genes are related to each other — whether they tend to be active together or in opposite patterns.

**What you see:**

- A grid/matrix where both rows and columns are gene names. Each cell is colored by the **Pearson correlation** between the two genes:
  - **Red (+1)** = the two genes are highly correlated (when one is active, the other is too)
  - **Blue (-1)** = the two genes are anti-correlated (when one is active, the other is quiet)
  - **White (0)** = no relationship

This helps identify gene "modules" — groups of genes that work together, which can reveal biological pathways involved in the cancer.

---

### Page 3: Clinical Explorer

**Purpose:** Explore relationships between clinical variables (age, gender, stage, treatments, etc.) using interactive charts and statistical tests.

This page has **three tabs:**

#### Tab 1: Cross-tabulation

**What it does:** Pick ANY two clinical variables and the app automatically generates the right chart and statistical test.

**How the auto-detection works:**
| Variable A | Variable B | Chart Type | Statistical Test |
|-----------|-----------|------------|-----------------|
| Categorical | Categorical | Grouped bar chart | **Chi-square test** |
| Categorical | Numeric | Box plot | **Kruskal-Wallis test** |
| Numeric | Numeric | Scatter plot with trendline | **Pearson correlation** |

**Available variables:** Gender, Stage, Vital Status, Colon Side, Metastatic status, Node Positive, Age Group, Sample Type, Age (numeric), Stage (numeric).

**Example:** Select "Stage" and "Gender" to see if cancer stage distribution differs between males and females, with a chi-square test telling you if the difference is statistically significant.

#### Tab 2: Right vs Left Colon

**What it does:** Compares cancers found in the **right side** of the colon (cecum through transverse colon) vs. the **left side** (splenic flexure through sigmoid colon).

**Why this matters:** Right-sided and left-sided colon cancers are almost like different diseases:
- Right-sided: more common in older patients, often diagnosed later, different molecular profile
- Left-sided: often found earlier due to symptoms, different response to targeted therapies

**What you see:**
- Stage distribution by colon side
- Age distribution by colon side (box plot)
- Gender breakdown by colon side
- Overall survival time by colon side
- Statistical test results (chi-square for stage, Mann-Whitney U for age)

#### Tab 3: Treatment Overview

**What it does:** Shows what treatments patients received and how treatment patterns vary by cancer stage.

**What you see:**
- Bar chart of treatment modalities (chemotherapy, radiation, neither/unknown)
- Bar chart showing what percentage of patients at each stage received chemotherapy (spoiler: chemotherapy use increases dramatically at higher stages)

---

### Page 4: Survival Analysis

**Purpose:** Analyze how long patients survive and what factors affect survival. This is one of the most clinically important pages.

This page has **three tabs:**

#### Tab 1: Kaplan-Meier Curves

**What it is:** A **Kaplan-Meier (KM) curve** is the gold standard way to visualize survival in medical research. It's a step-function line chart where:
- **X-axis** = time (days after diagnosis)
- **Y-axis** = probability of being alive (starts at 1.0 = 100%)
- The line drops down each time a patient dies
- Flat portions = periods where nobody died
- Steeper drops = periods with more deaths

**What you can stratify by (split the curve into groups):**
- **AJCC Stage** — see how Stage I vs II vs III vs IV patients survive differently
- **Colon Side** — right vs. left colon survival
- **Age Group** — younger vs. older patients
- **Gender** — male vs. female survival
- **Metastatic Status** — M0 (no spread) vs. M1 (distant spread)
- **Node Status** — N0 (no lymph node involvement) vs. N+ (lymph nodes affected)
- **Gene Expression** — split patients into "high expression" and "low expression" for any gene based on the median value, then compare survival

**Statistics shown:**
- **Median survival time:** The time point where 50% of patients in a group have died. If the line never drops below 0.5, median survival is "Not reached" (a good sign).
- **Log-rank test (p-value):** Tests whether the survival curves of two groups are significantly different. A p < 0.05 means the groups truly have different survival — it's not just chance.

#### Tab 2: Cox Proportional Hazards Regression

**What it is:** While Kaplan-Meier compares groups one factor at a time, **Cox regression** analyzes multiple factors simultaneously. It answers: "Controlling for age, does stage still affect survival? Does gender matter after accounting for stage?"

**What you do:** Select which clinical variables (covariates) to include: Age, Stage, Gender, Node Positive, Metastatic status.

**What you see:**

- **Forest Plot:** A horizontal chart showing the **Hazard Ratio (HR)** for each factor:
  - **HR = 1** (dashed vertical line) = no effect on survival
  - **HR > 1** = increases the risk of death (bad)
  - **HR < 1** = decreases the risk (protective)
  - **Red dots** = statistically significant (p < 0.05)
  - **Gray dots** = not significant
  - **Horizontal lines** = 95% confidence interval (the range of plausible HR values)

- **Concordance Index (C-index):** A single number (0 to 1) measuring how well the model discriminates between patients who die sooner vs. later:
  - **0.5** = no better than random coin flip
  - **0.7-0.8** = good discrimination
  - **>0.8** = excellent
  - **1.0** = perfect (never achieved in practice)

- **Model Summary Table:** Shows each factor's hazard ratio, confidence interval, and p-value.

**How Cox Regression works (simplified):** It assumes each factor multiplies the baseline risk of death by a constant amount. For example, if Stage IV has HR=3.0, a Stage IV patient has 3x the death risk of a Stage I patient at any point in time. The model finds these multipliers from the data.

#### Tab 3: Risk Stratification

**What it does:** Takes the Cox model you just fitted and uses it to assign each patient a **risk score**, then splits patients into **Low, Medium, and High risk** groups based on tertiles (thirds) of the score distribution.

**What you see:**
- KM survival curves for the three risk groups (they should separate nicely — low-risk patients surviving longer)
- Histogram of risk score distributions for each group
- Summary table showing the number of patients, events (deaths), and mean survival for each risk group

---

### Page 5: ML Prediction Lab

**Purpose:** Showcase three pre-trained machine learning models that make predictions from the data. These models are trained offline (before the app starts) and their results are displayed interactively.

This page has **three tabs:**

#### Tab A: Tumor vs Normal Classifier

**The question:** "Given a tissue sample's gene expression profile, can a machine learning model correctly identify whether it's cancerous or healthy?"

**The approach:**
1. Take the top 5,000 most variable genes
2. Scale them (StandardScaler — subtract mean, divide by standard deviation so all genes are on the same scale)
3. Select the 200 most informative genes using ANOVA F-test (SelectKBest)
4. Split data 80% train / 20% test using a **patient-aware split** — ensuring that if a patient donated both a tumor and a normal sample, both go into the same set (prevents data leakage)
5. Train a Logistic Regression model with L1 (Lasso) penalty

**What you see:**
- **AUC (Area Under the ROC Curve):** The main performance metric. Ranges from 0 to 1:
  - **0.5** = worthless (random guessing)
  - **0.7-0.8** = acceptable
  - **0.9+** = excellent
  - **1.0** = perfect
  - This model achieves ~1.000, which seems suspiciously good but is explained: tumor and normal tissue have massively different gene expression (thousands of genes change), so it's genuinely easy for ML to distinguish them.

- **ROC Curve:** A plot of True Positive Rate (correctly identified tumors) vs. False Positive Rate (normals incorrectly called tumors) at every decision threshold. A perfect model hugs the top-left corner. The diagonal line represents random guessing.

- **Confusion Matrix:** A 2x2 grid showing:
  - True Positives (correctly identified tumors)
  - True Negatives (correctly identified normals)
  - False Positives (normals misclassified as tumors)
  - False Negatives (tumors misclassified as normals)

- **Classification Report:** Precision, Recall, and F1-score for each class:
  - **Precision** = Of all samples the model called "tumor," what fraction actually were tumors?
  - **Recall** = Of all actual tumors, what fraction did the model catch?
  - **F1-score** = The harmonic mean of precision and recall (a balanced single number)

- **Top Biomarker Genes:** A bar chart of the 20 genes that the model relies on most to make its predictions. These are potential biomarkers — genes whose expression level strongly indicates cancer.

**ML Techniques used:**
- **Logistic Regression (L1/Lasso):** A classification algorithm that predicts the probability of a sample being tumor vs. normal. The L1 penalty encourages the model to use only a few genes (sets most gene weights to zero), making it interpretable.
- **Random Forest:** An ensemble of hundreds of decision trees. Each tree makes a prediction, and the final answer is the majority vote. Included for comparison.
- **StandardScaler:** Subtracts the mean and divides by standard deviation for each gene, so all genes are on the same scale.
- **SelectKBest (ANOVA F-test):** Ranks all 5,000 genes by how well they separate the two classes (tumor vs. normal), then keeps only the top 200.
- **GroupShuffleSplit:** A splitting strategy that ensures all samples from the same patient end up in the same set (train or test), preventing the model from "cheating" by seeing related samples in both sets.

#### Tab B: Stage Prediction

**The question:** "Can we predict whether a cancer is Early (Stage I/II) or Late (Stage III/IV) from gene expression?"

**Why this is harder:** Unlike tumor vs. normal (which differ by thousands of genes), early and late stage cancers are both cancers — they share most molecular features. The differences are subtle.

**The approach:**
1. Select genes using **Mann-Whitney U test** (a non-parametric test that ranks genes by how well they separate early vs. late stage) — this is better than just picking high-variance genes, because a gene can have high variance without being related to stage
2. Add clinical features (age, gender, colon side, metastatic status, chemotherapy)
3. Use **Stratified 5-fold cross-validation** — splits data into 5 parts, trains on 4, tests on 1, rotates 5 times. Reports the average AUC across all 5 runs.
4. Feature selection is done **inside each fold** to prevent leakage (if you selected features on the full dataset before splitting, the test set would influence feature selection)

**What you see:**
- **Mean AUC (5-fold CV):** The average AUC across all 5 folds. Typically around 0.6-0.8 for stage prediction (much harder than tumor/normal).
- **ROC Curve** for the best model configuration
- **Top Predictive Features:** Which genes and clinical features best distinguish early from late stage

**ML Techniques used:**
- **Logistic Regression (L2/Ridge):** Similar to L1 but shrinks coefficients without forcing them to zero. Multiple regularization strengths tested (C=0.01, 0.1, 1.0).
- **Random Forest:** Tested with different maximum tree depths (3, 5, 8) to control overfitting.
- **Gradient Boosting:** Builds trees sequentially, where each new tree tries to correct the mistakes of the previous ones. Tested with learning rates 0.05 and 0.1.
- **SVM (RBF kernel):** Support Vector Machine with Radial Basis Function kernel — finds a non-linear boundary between classes.
- **Stratified K-Fold CV:** Ensures each fold has the same proportion of early/late stage patients as the full dataset.
- **Mann-Whitney U test:** A non-parametric test that doesn't assume gene expression follows a normal distribution. Ranks all values, then checks if one group's ranks are systematically higher.

#### Tab C: Risk Stratification (ML)

**The question:** "Can we predict which patients are at highest risk of dying, using both their gene expression and clinical data?"

**The approach:**
1. For each of the 5,000 genes, fit a simple **univariate Cox model** to see if that gene alone predicts survival
2. Keep the top 50 genes with the lowest p-values (most prognostic)
3. Add clinical features (age, stage, node status, metastasis)
4. Fit a **multivariate Cox Proportional Hazards model** with a penalizer (regularization) to prevent overfitting
5. Split 70/30 train/test
6. Use the model to assign each test patient a **risk score** (the model's "partial hazard")
7. Split patients into thirds: Low, Medium, High risk

**What you see:**
- **C-index (train):** How well the model discriminates (same interpretation as above)
- **KM curves by risk group on the test set:** The ultimate validation — do the predicted risk groups actually have different survival?
- **Forest plot of hazard ratios:** Which features increase or decrease death risk
- **Risk score distribution:** Histogram showing how risk scores are distributed across groups
- **Summary table** with group sizes, death counts, and mean survival times
- **Top Prognostic Genes:** Expandable list of genes most associated with survival

**ML Technique used:**
- **Cox Proportional Hazards (CoxPH):** The standard model for survival prediction. Unlike logistic regression (which predicts yes/no), Cox regression models the *hazard rate* — the instantaneous risk of dying at any given time point. It handles "censored" data (patients who were still alive at last follow-up — we know they survived at least X days, but don't know their final outcome).

---

### Page 6: Gene Lookup Tool

**Purpose:** A search engine for individual genes. Type any gene name and get its complete profile.

**What you do:** Type a gene name (e.g., TP53, KRAS, APC, BRAF, MLH1) and select it from the matching results.

**What you see for each gene:**

1. **Expression Boxplot (Tumor vs Normal)**
   - Side-by-side box plots showing the gene's expression distribution in tumor samples vs. normal samples
   - Box plots show the median (middle line), interquartile range (box), and outliers (dots)
   - The diamond inside shows the mean and standard deviation

2. **Differential Expression Statistics**
   - **log2 Fold Change:** How much the gene is up/down in tumors
   - **p-value:** Whether the difference is statistically significant
   - **Mean expression** in tumor and normal
   - A verdict: "Significantly UP-regulated in tumors" / "DOWN-regulated" / "Not significantly differentially expressed"

3. **Expression by Clinical Variables**
   - Box plots of the gene's expression across AJCC Stages (I-IV) — to see if the gene's activity changes as cancer progresses
   - Box plots by colon side (right vs. left) — to see if the gene behaves differently by tumor location

4. **Survival Impact**
   - Patients are split at the **median expression** of this gene into "High" and "Low" groups
   - A Kaplan-Meier curve compares survival between these two groups
   - A log-rank test determines if the gene significantly affects survival
   - This is how researchers identify **prognostic biomarkers** — genes whose expression level predicts patient outcome

5. **Full Expression Distribution**
   - An overlaid histogram showing the complete distribution of expression values for tumor and normal samples

---

### Page 7: Patient Risk Calculator

**Purpose:** An interactive tool where you input a patient's clinical and molecular profile and get a predicted risk score.

**What you input:**
- **Clinical features:** Age, gender, AJCC stage, colon side, lymph node involvement (N0 vs N+), distant metastasis (M0 vs M1)
- **Gene expression values:** The top 10 most prognostic genes. Defaults are set to the population median — you can adjust them to simulate different molecular profiles.

**What you get:**

1. **Risk Score:** A continuous number from the Cox model
2. **Risk Group:** Low, Medium, or High — determined by where your score falls relative to the training patients
3. **Gauge Chart:** A visual speedometer showing the risk score against the full range of possible scores, with green/yellow/red zones
4. **Contributing Factors:** A bar chart showing which factors pushed the risk score up (red = increases risk) or down (green = decreases risk), ranked by impact
5. **Full Factor Breakdown:** An expandable table with each feature's hazard ratio and direction of effect

**How it works behind the scenes:**
1. Your inputs are converted to the same format the model was trained on
2. Continuous features are **standardized** using the same mean/standard deviation from training
3. The Cox model calculates a **partial hazard** — your risk score
4. The score is compared to the distribution of scores in the test set to assign a risk group
5. Each feature's contribution is calculated as: coefficient x standardized value

**Important disclaimer:** This is a research/educational tool, NOT for clinical decision-making.

---

## Every Statistic and ML Technique Explained

### Statistical Tests

| Test | When It's Used | What It Does | Example in This App |
|------|---------------|--------------|---------------------|
| **Welch's t-test** | Comparing means of two groups | Tests if the average expression of a gene differs between tumor and normal tissue. "Welch's" variant doesn't assume equal variance. | Differential expression analysis |
| **Chi-square test** | Two categorical variables | Tests if there's an association between two categories (e.g., are gender and stage related?). Compares observed counts to what you'd expect if the variables were independent. | Clinical cross-tabulation (cat x cat) |
| **Kruskal-Wallis test** | One categorical, one numeric variable | Non-parametric alternative to ANOVA. Tests if a numeric variable (like age) differs across categories (like cancer stage) without assuming normality. | Clinical cross-tabulation (cat x num) |
| **Mann-Whitney U test** | Comparing two groups (non-parametric) | Like a t-test but doesn't assume normal distribution. Compares ranks instead of raw values. | Gene selection for stage prediction |
| **Pearson correlation** | Two numeric variables | Measures linear relationship (-1 to +1). Used for gene-gene correlations and numeric cross-tabulation. | Correlation heatmap, clinical explorer |
| **Log-rank test** | Comparing survival curves | Tests if two (or more) survival curves are significantly different. The standard test in survival analysis. | Kaplan-Meier survival comparisons |
| **Benjamini-Hochberg FDR** | Multiple hypothesis correction | When testing thousands of genes, adjusts p-values to control the proportion of false discoveries. | Differential expression (padj) |
| **ANOVA F-test** | Feature selection | Measures how well a feature separates classes (used by SelectKBest). | Tumor/normal classifier feature selection |

### Machine Learning Techniques

| Technique | Type | What It Does | Where It's Used |
|-----------|------|--------------|-----------------|
| **Logistic Regression** | Classification | Predicts the probability of a binary outcome (e.g., tumor vs normal) using a weighted sum of features passed through a sigmoid function. L1 (Lasso) penalty = pushes unimportant feature weights to exactly zero. L2 (Ridge) penalty = shrinks weights but keeps all features. | Tumor/normal classifier, Stage predictor |
| **Random Forest** | Classification | Creates hundreds of decision trees, each trained on a random subset of data and features. Final prediction = majority vote. Resistant to overfitting, handles non-linear relationships. | Tumor/normal, Stage prediction |
| **Gradient Boosting** | Classification | Builds trees sequentially — each new tree focuses on correcting mistakes from previous trees. Often more accurate than Random Forest but slower and more prone to overfitting. | Stage prediction |
| **SVM (RBF kernel)** | Classification | Finds the optimal boundary between classes by mapping data into a higher-dimensional space where a linear separator exists. RBF (Radial Basis Function) kernel allows non-linear boundaries. | Stage prediction |
| **Cox Proportional Hazards** | Survival regression | Models the relationship between covariates and the hazard (instantaneous death risk). Handles censored data (patients still alive at last follow-up). Output = hazard ratio per feature. | Survival analysis, Risk stratification |
| **PCA** | Dimensionality reduction | Finds the directions of maximum variance in high-dimensional data and projects onto them. Reduces thousands of gene dimensions to 2-3 for visualization. | Gene expression PCA tab |
| **StandardScaler** | Preprocessing | Subtracts mean, divides by standard deviation for each feature. Ensures all features are on the same scale (critical for algorithms like Logistic Regression and SVM). | All ML models |
| **SelectKBest** | Feature selection | Scores each feature by a statistical test (ANOVA F-test), keeps only the top K. Reduces noise and computation. | Tumor/normal classifier, Stage predictor |
| **Stratified K-Fold CV** | Evaluation | Splits data into K folds maintaining class proportions. Each fold takes a turn as the test set. Gives a more reliable performance estimate than a single train/test split. | Stage prediction |
| **GroupShuffleSplit** | Evaluation | Like train/test split, but ensures all samples from the same "group" (patient) stay together. Prevents data leakage from matched tumor/normal pairs. | Tumor/normal classifier |

### Key Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| **AUC (Area Under ROC Curve)** | 0.0 - 1.0 | Overall classification performance. 0.5 = random, 1.0 = perfect. |
| **Concordance Index (C-index)** | 0.0 - 1.0 | Survival model discrimination. How often does the model correctly rank who dies first? 0.5 = random, 1.0 = perfect. |
| **Precision** | 0.0 - 1.0 | Of positive predictions, how many were correct? |
| **Recall (Sensitivity)** | 0.0 - 1.0 | Of actual positives, how many were caught? |
| **F1 Score** | 0.0 - 1.0 | Harmonic mean of precision and recall. Balanced metric. |
| **Hazard Ratio (HR)** | 0 - infinity | >1 = increases death risk, <1 = protective, =1 = no effect. |
| **p-value** | 0.0 - 1.0 | <0.05 = statistically significant, >0.05 = could be chance. |
| **log2 Fold Change** | -inf to +inf | >0 = up-regulated in tumors, <0 = down-regulated. |logFC|>1 = biologically meaningful. |

---

## Project Structure

```
ColonScope/
|
|-- app.py                          # Main entry point. Loads data, sets up sidebar navigation, routes to pages.
|-- requirements.txt                # Python package dependencies
|
|-- data/                           # Raw datasets (not modified by the app)
|   |-- COAD.tsv                    # Gene expression matrix (60,660 genes x 514 samples)
|   |-- TCGA-COAD.clinical.tsv      # Patient clinical records
|   |-- TCGA-COAD.survival.tsv      # Patient survival data
|   |-- ensembl_to_symbol.csv       # Gene ID to gene name mapping
|
|-- models/                         # Pre-trained ML model artifacts (generated by training script)
|   |-- tumor_normal.joblib         # Tumor vs Normal classifier (Logistic Regression + Random Forest)
|   |-- stage_predictor.joblib      # Stage prediction models (multiple algorithms compared)
|   |-- risk_model.joblib           # Cox PH risk stratification model + feature statistics
|   |-- de_results.csv              # Pre-computed differential expression results
|   |-- meta.joblib                 # Dataset metadata (sample counts, gene lists)
|
|-- views/                          # One file per page — each exports a render() function
|   |-- page_overview.py            # Page 1: Data Overview
|   |-- page_expression.py          # Page 2: Gene Expression Analysis (DE, PCA, Correlation)
|   |-- page_clinical.py            # Page 3: Clinical Explorer (cross-tab, colon side, treatment)
|   |-- page_survival.py            # Page 4: Survival Analysis (KM, Cox, Risk groups)
|   |-- page_prediction.py          # Page 5: ML Prediction Lab (3 pre-trained models)
|   |-- page_gene_lookup.py         # Page 6: Gene Lookup Tool
|   |-- page_risk_calculator.py     # Page 7: Patient Risk Calculator
|
|-- utils/                          # Reusable backend modules
|   |-- data_loader.py              # Loads the 3 TSV/CSV data files with Streamlit caching
|   |-- gene_mapping.py             # Converts Ensembl IDs (ENSG...) to human-readable gene symbols
|   |-- preprocessing.py            # Data cleaning: stage coarsening, derived features, merging
|   |-- de_analysis.py              # Differential expression engine (t-test + BH FDR correction)
|   |-- survival_utils.py           # Kaplan-Meier fitting, Cox regression, risk score computation
|   |-- ml_models.py                # ML training pipelines (tumor/normal, stage, risk)
|   |-- plotting.py                 # Reusable Plotly chart factories (donut, MA, PCA, ROC, heatmap, etc.)
|
|-- scripts/
|   |-- train_all_models.py         # Offline script: trains all ML models and saves to models/
|
|-- assets/
    |-- style.css                   # Custom dark theme (Catppuccin Mocha color palette)
```

---

## Setup and Installation

### Prerequisites

- **Python 3.10+** installed on your system
- **pip** (Python package manager, comes with Python)

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

This installs: Streamlit (web framework), pandas/numpy (data), plotly (charts), scikit-learn (ML), lifelines (survival analysis), scipy (statistics), joblib (model saving).

### Step 2: Train ML models (one-time, before first run)

```bash
python scripts/train_all_models.py
```

This reads the raw data, runs differential expression analysis, trains all three ML models, and saves the results to the `models/` folder. Takes approximately 2-3 minutes.

### Step 3: Run the app

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. The app loads the data once (cached for speed) and you can navigate between pages using the sidebar.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Streamlit | Builds the interactive web UI with Python — no HTML/JS needed |
| **Data Manipulation** | pandas, numpy | Loading, cleaning, transforming tabular data |
| **Visualization** | Plotly | Interactive charts (hover, zoom, pan) with dark theme |
| **Statistics** | scipy | t-tests, chi-square, correlation, Mann-Whitney, Kruskal-Wallis |
| **Survival Analysis** | lifelines | Kaplan-Meier curves, Cox regression, log-rank tests |
| **Machine Learning** | scikit-learn | Logistic Regression, Random Forest, Gradient Boosting, SVM, PCA |
| **Model Persistence** | joblib | Saving/loading trained models to disk |
| **Theme** | Catppuccin Mocha | A popular dark color palette with pastel accent colors |

### Fully Offline

This app requires **no internet connection** at runtime. Gene ID mapping uses a bundled CSV file. No external API calls are made. All data and models are local.

---

### Live App

The hosted version is available at **[ppb-miniproj.streamlit.app](https://ppb-miniproj.streamlit.app/)** — no setup required.
