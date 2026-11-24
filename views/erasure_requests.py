"""Page: Erasure Requests"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from clinical_unlearning.config import get_label
from clinical_unlearning.storage import audit
from ui.components import empty_state, page_header, task_short


def render(df):
    LABEL = get_label()
    page_header("PRIVACY OPERATIONS", "Erasure Requests", "Select training patients. All admissions belonging to each selected patient are included.")
    if st.session_state.train is None:
        empty_state("🧩", "Create a data split first", "Create a patient-level split in Model Training before selecting patients to forget.")
    else:
        tr = st.session_state.train
        patient_ids = sorted(tr.subject_id.unique().astype(int).tolist())
        rank = {pid: i + 1 for i, pid in enumerate(patient_ids)}   # anonymised display number
        pending = st.session_state.get("pending_patient", None)
        mode = st.radio("Request type", ["Individual patient", "Multiple patients", "Patient ID range", "Demographic group", "Random sample", "Outcome-balanced sample"], horizontal=True)
        selected = []
        if mode == "Individual patient":
            query = st.text_input("Search patient display label or ID", placeholder="Type a patient ID")
            filtered = patient_ids
            if query.strip():
                filtered = [x for x in patient_ids if query.strip() in str(x)]
            options = filtered[:200]
            select_all = False
            if len(filtered) > 200:
                select_all = st.checkbox(f"Select all {len(filtered):,} matching patients", value=False,
                                         help="The list below shows only the first 200 to keep the page fast. Tick this to include every match.")
            if select_all:
                selected = filtered
            elif options:
                default = [pending] if pending in options else []
                selected = st.multiselect("Select patient(s)", options, default=default,
                                          format_func=lambda x: f"Patient {rank[x]:05d}")
                if len(filtered) > 200:
                    st.caption(f"Showing the first 200 of {len(filtered):,} matches. Refine the search or tick 'Select all'.")
        elif mode == "Multiple patients":
            raw = st.text_input("Enter patient IDs separated by commas", placeholder="e.g. 10001234, 10004567")
            if raw.strip():
                try:
                    requested = {int(x.strip()) for x in raw.split(",") if x.strip()}
                    selected = [x for x in patient_ids if x in requested]
                    unknown = requested - set(patient_ids)
                    if unknown:
                        st.warning(f"{len(unknown)} IDs are not in the training split and were ignored.")
                except ValueError:
                    st.error("Use comma-separated numeric patient IDs.")
        elif mode == "Patient ID range":
            lo, hi = st.columns(2)
            start_id = lo.number_input("Start ID", min_value=min(patient_ids), max_value=max(patient_ids), value=min(patient_ids))
            end_id = hi.number_input("End ID", min_value=min(patient_ids), max_value=max(patient_ids), value=min(patient_ids))
            if start_id <= end_id:
                selected = [x for x in patient_ids if start_id <= x <= end_id]
            else:
                st.error("Start ID must not exceed end ID.")
        elif mode == "Random sample":
            total = len(patient_ids)
            rs1, rs2 = st.columns(2)
            size_by = rs1.radio("Size by", ["Number of patients", "Percent of training patients"], horizontal=True)
            if size_by == "Number of patients":
                n_pick = rs2.number_input("Patients to forget", min_value=1, max_value=total, value=min(1000, total), step=100)
            else:
                pct = rs2.slider("Percent of training patients", 1, 50, 5)
                n_pick = max(1, int(round(total * pct / 100)))
            selected = np.random.default_rng(21).choice(patient_ids, min(int(n_pick), total), replace=False).tolist()
            st.caption(f"{len(selected):,} of {total:,} training patients, chosen at random with a fixed seed (the same request always picks the same patients).")
        elif mode == "Outcome-balanced sample":
            died = sorted(tr.loc[tr[LABEL] == 1, "subject_id"].unique().astype(int).tolist())
            lived = sorted(set(patient_ids) - set(died))
            n_each = st.slider("Patients per outcome group", 5, max(5, min(5000, len(died), len(lived))), min(100, max(5, min(len(died), len(lived)))), step=5)
            rs = np.random.default_rng(11)
            selected = (rs.choice(died, min(n_each, len(died)), replace=False).tolist()
                        + rs.choice(lived, min(n_each, len(lived)), replace=False).tolist())
            st.caption("Half the patients have the positive outcome of the selected prediction task, half do not. For rare outcomes such as mortality (~1-2%), this guarantees the attack has memorised rare-class patients to find.")
        else:
            group_field = st.selectbox("Group field", [c for c in ["gender","race"] if c in tr.columns] + (["age"] if "age" in tr.columns else []))
            if group_field == "age":
                min_age = int(pd.to_numeric(tr.age, errors="coerce").min())
                max_age = int(pd.to_numeric(tr.age, errors="coerce").max())
                low, high = st.slider("Age range", min_age, max_age, (min_age, max_age))
                selected = tr.loc[pd.to_numeric(tr.age, errors="coerce").between(low, high), "subject_id"].unique().tolist()
            else:
                group_value = st.selectbox("Group value", sorted(tr[group_field].fillna("Unknown").astype(str).unique()))
                selected = tr.loc[tr[group_field].fillna("Unknown").astype(str) == group_value, "subject_id"].unique().tolist()
            st.caption("Group selection includes a patient if at least one of their training admissions matches the selected group.")
        selected = sorted(set(int(x) for x in selected))
        affected = tr[tr.subject_id.isin(selected)]
        a,b,c = st.columns(3)
        a.metric("Patients selected", f"{len(selected):,}")
        b.metric("Admissions to forget", f"{len(affected):,}")
        c.metric("Admissions retained", f"{len(tr)-len(affected):,}")
        if len(affected):
            st.caption(f"Positive outcome rate ({task_short()}) · forget set {affected[LABEL].mean():.1%} · whole training set {tr[LABEL].mean():.1%}")
        if len(affected):
            preview = affected[["subject_id","hadm_id",LABEL]].copy()
            preview["Patient"] = preview.subject_id.map(lambda x: f"Patient {rank[int(x)]:05d}")
            st.dataframe(preview[["Patient","hadm_id",LABEL]].head(100), use_container_width=True, hide_index=True)
        if st.button("Confirm Forget Set", type="primary", disabled=(len(selected)==0)):
            st.session_state.forget_ids = selected
            st.session_state.mia_before = None
            st.session_state.mia_after = None
            st.session_state.mia_attack_data = None
            audit("forget_set_confirmed", {"patients":len(selected),"admissions":len(affected)})
            st.success(f"Forget Set confirmed: {len(selected):,} patients, {len(affected):,} admissions. Next: run the attack before unlearning (MIA Simulation), then Unlearning & Audit.")
        if st.session_state.forget_ids:
            current = tr[tr.subject_id.isin(st.session_state.forget_ids)]
            st.info(f"Current Forget Set: {len(st.session_state.forget_ids):,} patients / {len(current):,} training admissions.")
            if st.button("Clear Forget Set"):
                st.session_state.forget_ids = []
                st.rerun()
