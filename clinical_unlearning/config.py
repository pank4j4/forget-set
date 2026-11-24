"""Paths, constants and the selected prediction task."""
from __future__ import annotations

import threading
from pathlib import Path

# Project root = the folder that contains app.py
APP_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = APP_DIR / "data" / "raw"
MODEL_DIR = APP_DIR / "models"
RESULT_DIR = APP_DIR / "results"
DB_PATH = APP_DIR / "clinical_unlearning_lab.db"
for _folder in (RAW_DIR, MODEL_DIR, RESULT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

TARGET = "hospital_expire_flag"          # in-hospital mortality (kept for the dashboards)
TASKS = {
    "Prolonged hospital stay (longer than median)": "long_stay",
    "In-hospital mortality": TARGET,
}
DEFAULT_LABEL = "long_stay"

REQUIRED_FILES = ["admissions.csv"]
OPTIONAL_FILES = [
    "patients.csv", "diagnoses_icd.csv",
    "prescriptions.csv", "procedures_icd.csv"
]
ALL_FILES = REQUIRED_FILES + OPTIONAL_FILES

LEARNER_NAMES = {"forest": "Random forest", "mlp": "Neural network", "sgd": "Logistic regression"}

# The outcome column the models are trained on. It is chosen in the sidebar and stored per
# Streamlit session thread, so the backend stays free of any Streamlit import.
_selected = threading.local()


def set_label(name: str) -> None:
    _selected.value = name


def get_label() -> str:
    return getattr(_selected, "value", DEFAULT_LABEL)
