"""Streamlit session state: defaults, task switching and cached data loading."""
from __future__ import annotations

import streamlit as st

from clinical_unlearning.config import ALL_FILES, RAW_DIR
from clinical_unlearning.data import load_mimic_data

DEFAULTS = {
    "data": None, "train": None, "val": None, "test": None,
    "baseline": None, "baseline_metrics": None, "baseline_probs": None,
    "unlearned": None, "last_result": None, "forget_ids": [],
    "feature_cols": None, "model_ready": False,
    "mia_before": None, "mia_after": None, "mia_attack_data": None,
}

# Everything that depends on the chosen prediction task and must be cleared when it changes.
TASK_DEPENDENT = [
    "train", "val", "test", "baseline", "baseline_metrics", "baseline_probs", "unlearned",
    "last_result", "feature_cols", "mia_before", "mia_after", "mia_attack_data",
]


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_for_task_change(task_name: str) -> None:
    """Clear splits, models and attack results when the prediction task changes."""
    if st.session_state.get("task_prev") != task_name:
        for key in TASK_DEPENDENT:
            st.session_state[key] = None
        st.session_state.forget_ids = []
        st.session_state.task_prev = task_name


def _file_signature() -> tuple:
    return tuple(
        (name, (RAW_DIR / name).stat().st_mtime_ns if (RAW_DIR / name).exists() else None)
        for name in ALL_FILES
    )


@st.cache_data(show_spinner="Reading and preparing the local MIMIC-IV files...")
def _load_cached(file_signature):
    return load_mimic_data(file_signature)


def ensure_data():
    """Return the prepared dataset, loading it once per session. Shows a helpful message and stops on failure."""
    if st.session_state.data is None:
        try:
            with st.spinner("Loading MIMIC-IV from data/raw..."):
                st.session_state.data = _load_cached(_file_signature())
        except Exception as exc:
            st.error(f"Dataset loading failed: {exc}")
            st.markdown("**Expected folder layout**")
            st.code(
                "Clinical-Unlearning-Lab/\n"
                "├── app.py\n"
                "├── requirements.txt\n"
                "└── data/\n"
                "    └── raw/\n"
                "        ├── admissions.csv\n"
                "        ├── patients.csv\n"
                "        ├── diagnoses_icd.csv\n"
                "        ├── prescriptions.csv\n"
                "        └── procedures_icd.csv"
            )
            st.info("Only admissions.csv is mandatory. The other four files are optional, but your five files will all be used when present.")
            st.stop()
    return st.session_state.data
