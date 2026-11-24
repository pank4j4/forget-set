"""Page: Overview"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from clinical_unlearning.config import ALL_FILES, RAW_DIR, REQUIRED_FILES, TARGET
from ui.components import page_header, plot, section_title


def render(df):
    page_header("RESEARCH COMMAND CENTER", "Overview Dashboard", "Cohort summary and source-file status for the loaded MIMIC-IV tables.")
    n_patients = df.subject_id.nunique()
    n_admissions = df.hadm_id.nunique()
    mortality = df[TARGET].mean() * 100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Unique patients", f"{n_patients:,}")
    c2.metric("Admissions", f"{n_admissions:,}")
    c3.metric("In-hospital mortality", f"{mortality:.2f}%")
    c4.metric("Source tables", f"{sum((RAW_DIR/f).exists() for f in ALL_FILES)}/5")
    left, right = st.columns(2)
    with left:
        counts = df[TARGET].value_counts().rename(index={0:"Survived",1:"Died"}).reset_index()
        counts.columns = ["Outcome","Admissions"]
        plot(px.pie(counts, names="Outcome", values="Admissions", hole=.58, title="Admission outcomes"), use_container_width=True)
    with right:
        if "age" in df.columns:
            plot(px.histogram(df, x="age", nbins=25, title="Patient age distribution"), use_container_width=True)
        else:
            st.info("Age distribution is unavailable because patients.csv was not found.")
    if "gender" in df.columns:
        g = df.groupby("gender")[TARGET].agg(["count","mean"]).reset_index()
        g["mortality_percent"] = g["mean"] * 100
        plot(px.bar(g, x="gender", y="mortality_percent", title="Mortality rate by gender", labels={"gender":"Gender","mortality_percent":"Mortality (%)"}), use_container_width=True)
    section_title("Source file status", "Which MIMIC-IV tables were found in data/raw")
    status_rows = []
    for name in ALL_FILES:
        path = RAW_DIR / name
        status_rows.append({"File":name, "Status":"✅ Loaded" if path.exists() else ("⚠️ Required" if name in REQUIRED_FILES else "➖ Optional / absent"),
                            "Size (MB)":round(path.stat().st_size/1024**2,2) if path.exists() else "—"})
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
