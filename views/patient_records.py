"""Page: Patient Records"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from clinical_unlearning.config import TARGET
from ui.components import page_header


def render(df):
    page_header("CLINICAL DATA WORKSPACE", "Patient Records", "Search anonymized patient profiles and inspect their admission-level records.")
    patient_summary = df.groupby("subject_id").agg(
        admissions=("hadm_id", "nunique"),
        age=("age", "first") if "age" in df.columns else (TARGET, "size"),
        mortality=(TARGET, "max")
    ).reset_index()
    if "age" not in df.columns:
        patient_summary["age"] = np.nan
    patient_summary["display_id"] = [f"Patient {i:05d}" for i in range(1, len(patient_summary)+1)]
    id_map = dict(zip(patient_summary.display_id, patient_summary.subject_id))
    patient_summary["gender"] = (
        patient_summary.subject_id.map(df.groupby("subject_id")["gender"].first())
        if "gender" in df.columns else "Not available"
    )
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search = st.text_input("Search patient", placeholder="e.g. Patient 00012")
    with f2:
        gender_options = ["All"] + sorted(patient_summary.gender.fillna("Unknown").astype(str).unique().tolist())
        gender_filter = st.selectbox("Gender", gender_options)
    with f3:
        outcome_filter = st.selectbox("Outcome", ["All", "Any mortality admission", "No mortality admission"])
    view = patient_summary.copy()
    if search.strip():
        view = view[view.display_id.str.contains(search.strip(), case=False, regex=False)]
    if gender_filter != "All":
        view = view[view.gender.astype(str) == gender_filter]
    if outcome_filter == "Any mortality admission":
        view = view[view.mortality == 1]
    elif outcome_filter == "No mortality admission":
        view = view[view.mortality == 0]
    page_size = st.selectbox("Rows per page", [10, 20, 50], index=0)
    max_pages = max(1, math.ceil(len(view) / page_size))
    page_no = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1)
    start = (page_no - 1) * page_size
    shown = view.iloc[start:start+page_size].copy()
    shown["Age"] = shown["age"].round(0).astype("Int64") if "age" in shown else pd.NA
    shown["Mortality"] = shown["mortality"].map({0: "No", 1: "Yes"})
    display_table = shown[["display_id", "Age", "gender", "admissions", "Mortality"]].rename(
        columns={"display_id":"Patient", "gender":"Gender", "admissions":"Admissions"}
    )
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(start+1, len(view)):,}–{min(start+page_size, len(view)):,} of {len(view):,} patients. Identifiers are anonymized in the interface.")
    if len(view):
        chosen = st.selectbox("Open patient profile", view.display_id.tolist())
        chosen_id = id_map[chosen]
        record = df[df.subject_id == chosen_id].copy()
        a, b, c = st.columns(3)
        a.metric("Admissions", f"{record.hadm_id.nunique():,}")
        b.metric("Age", str(int(record.age.dropna().iloc[0])) if "age" in record and record.age.notna().any() else "N/A")
        c.metric("Any in-hospital mortality", "Yes" if record[TARGET].max() == 1 else "No")
        safe_cols = [c for c in ["hadm_id", "admission_type", "admission_location", "insurance", "race", TARGET] if c in record.columns]
        profile = record[safe_cols].copy()
        profile.insert(0, "Patient", chosen)
        profile = profile.rename(columns={TARGET:"hospital_expire_flag"})
        st.markdown("#### Admission history")
        st.dataframe(profile, use_container_width=True, hide_index=True)
        if st.button("Create erasure request for this patient", type="primary"):
            st.session_state.pending_patient = int(chosen_id)
            st.success("Patient selected. Review and confirm the request in Erasure Requests.")
