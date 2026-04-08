import streamlit as st
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@st.cache_data(show_spinner="Loading expression data...")
def load_expression(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, "COAD.tsv")
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.astype("float32")
    return df  # genes x samples


@st.cache_data(show_spinner="Loading clinical data...")
def load_clinical(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, "TCGA-COAD.clinical.tsv")
    df = pd.read_csv(path, sep="\t")

    rename = {
        "sample": "sample_id",
        "gender.demographic": "gender",
        "vital_status.demographic": "vital_status",
        "age_at_index.demographic": "age",
        "race.demographic": "race",
        "ethnicity.demographic": "ethnicity",
        "ajcc_pathologic_stage.diagnoses": "stage",
        "ajcc_pathologic_t.diagnoses": "tumor_t",
        "ajcc_pathologic_n.diagnoses": "tumor_n",
        "ajcc_pathologic_m.diagnoses": "tumor_m",
        "tissue_or_organ_of_origin.diagnoses": "anatomical_site",
        "primary_diagnosis.diagnoses": "diagnosis",
        "sample_type.samples": "sample_type",
        "tumor_grade.diagnoses": "tumor_grade",
        "treatment_type.treatments.diagnoses": "treatment_types",
        "treatment_or_therapy.treatments.diagnoses": "received_treatment",
        "morphology.diagnoses": "morphology",
        "year_of_diagnosis.diagnoses": "year_of_diagnosis",
    }
    df = df.rename(columns=rename)
    return df


@st.cache_data(show_spinner="Loading survival data...")
def load_survival(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, "TCGA-COAD.survival.tsv")
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"OS.time": "os_time", "OS": "os_event", "_PATIENT": "patient_id"})
    return df
