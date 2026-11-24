"""MIMIC-IV loading, feature construction, patient-level splits and preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from clinical_unlearning.config import RAW_DIR, TARGET, get_label


# ---------------------------------------------------------------------
# Data loading and feature construction
# ---------------------------------------------------------------------
COUNT_CHUNK = 2_000_000   # rows per chunk when counting rows per admission in very large tables


def read_local_csv(filename, required=False, columns=None):
    """Read a CSV from data/raw. `columns` limits what is loaded (saves memory on the big tables)."""
    path = RAW_DIR / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Place the CSV files inside the project's data/raw folder."
            )
        return None
    wanted = set(columns) if columns else None
    return pd.read_csv(path, low_memory=False, usecols=(lambda c: c in wanted) if wanted else None)

def _numeric_count_table(path_name, output_name):
    """Rows per admission (hadm_id), counted in chunks so huge files such as prescriptions.csv
    never have to fit in memory. Only the hadm_id column is read."""
    path = RAW_DIR / path_name
    if not path.exists():
        return None
    total = None
    for chunk in pd.read_csv(path, usecols=lambda c: c == "hadm_id", chunksize=COUNT_CHUNK):
        if "hadm_id" not in chunk.columns:
            return None
        part = chunk["hadm_id"].dropna().astype("int64").value_counts()
        total = part if total is None else total.add(part, fill_value=0)
    if total is None:
        return None
    out = total.astype("int64").rename(output_name).rename_axis("hadm_id").reset_index()
    return out

def load_mimic_data(file_signature):
    admissions = read_local_csv("admissions.csv", required=True)

    needed = ["subject_id", "hadm_id", TARGET]
    missing = [c for c in needed if c not in admissions.columns]
    if missing:
        raise ValueError(f"admissions.csv is missing required columns: {missing}")

    # Keep only admission-time/demographic fields. Do not use discharge or death times.
    keep = [
        "subject_id", "hadm_id", TARGET, "admission_type",
        "admission_location", "insurance", "language",
        "marital_status", "race", "admittime", "dischtime"
    ]
    admissions = admissions[[c for c in keep if c in admissions.columns]].copy()
    admissions[TARGET] = pd.to_numeric(admissions[TARGET], errors="coerce")
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    admissions = admissions.dropna(subset=["subject_id", "hadm_id", TARGET])
    admissions = admissions[admissions[TARGET].isin([0, 1])].copy()
    admissions[TARGET] = admissions[TARGET].astype(int)
    admissions["subject_id"] = admissions["subject_id"].astype("int64")
    admissions["hadm_id"] = admissions["hadm_id"].astype("int64")

    # Derive only features available at admission time.
    if "admittime" in admissions.columns:
        dt = pd.to_datetime(admissions["admittime"], errors="coerce")
        admissions["admission_month"] = dt.dt.month
        admissions["admission_dayofweek"] = dt.dt.dayofweek
        if "dischtime" in admissions.columns:
            # Label only (never a feature): prolonged stay = longer than the median stay.
            out = pd.to_datetime(admissions["dischtime"], errors="coerce")
            admissions["los_days"] = (out - dt).dt.total_seconds() / 86400.0
            admissions = admissions[admissions["los_days"].notna() & (admissions["los_days"] >= 0)].copy()
            admissions["long_stay"] = (admissions["los_days"] > admissions["los_days"].median()).astype(int)
        admissions = admissions.drop(columns=[c for c in ["admittime", "dischtime"] if c in admissions.columns])

    patients = read_local_csv("patients.csv", required=False, columns=["subject_id", "gender", "anchor_age"])
    if patients is not None and "subject_id" in patients.columns:
        cols = ["subject_id"] + [c for c in ["gender", "anchor_age"] if c in patients.columns]
        patient_small = patients[cols].drop_duplicates("subject_id").copy()
        if "anchor_age" in patient_small.columns:
            patient_small["age"] = pd.to_numeric(patient_small["anchor_age"], errors="coerce").clip(upper=89)
            patient_small = patient_small.drop(columns=["anchor_age"])
        admissions = admissions.merge(patient_small, on="subject_id", how="left", validate="many_to_one")

    # Aggregated diagnosis burden.
    diagnoses = read_local_csv("diagnoses_icd.csv", required=False, columns=["hadm_id", "icd_code"])
    if diagnoses is not None and {"hadm_id", "icd_code"}.issubset(diagnoses.columns):
        dx = diagnoses.groupby("hadm_id").agg(
            diagnosis_count=("icd_code", "count"),
            unique_diagnoses=("icd_code", "nunique")
        ).reset_index()
        admissions = admissions.merge(dx, on="hadm_id", how="left", validate="one_to_one")

    # Prescription/procedure counts. No raw drug or code text is exposed as a feature.
    for filename, output in [
        ("prescriptions.csv", "prescription_count"),
        ("procedures_icd.csv", "procedure_count")
    ]:
        counts = _numeric_count_table(filename, output)
        if counts is not None:
            admissions = admissions.merge(counts, on="hadm_id", how="left", validate="one_to_one")

    for col in ["diagnosis_count", "unique_diagnoses", "prescription_count", "procedure_count"]:
        if col in admissions.columns:
            admissions[col] = pd.to_numeric(admissions[col], errors="coerce").fillna(0)

    # Fill display fields only; model pipeline still performs train-fitted imputation.
    for col in admissions.select_dtypes(include=["object", "category"]).columns:
        admissions[col] = admissions[col].fillna("Unknown").astype(str)

    admissions = admissions.replace([np.inf, -np.inf], np.nan)
    if admissions.empty:
        raise ValueError("No valid admission rows remain after cleaning.")
    if admissions[TARGET].nunique() != 2:
        raise ValueError("Both mortality outcome classes (0 and 1) are needed.")
    return admissions

def get_feature_columns(df):
    excluded = {
        "subject_id", "hadm_id", TARGET, "long_stay", "los_days",
        "deathtime", "dischtime", "discharge_location",
        "dod", "dod_hosp", "dod_ssn"
    }
    return [c for c in df.columns if c not in excluded]

def split_by_patient(df, seed=42):
    LABEL = get_label()
    groups = df["subject_id"].values
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(first.split(df, df[LABEL], groups))
    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_idx, test_idx = next(second.split(temp, temp[LABEL], temp["subject_id"].values))
    val, test = temp.iloc[val_idx].copy(), temp.iloc[test_idx].copy()
    assert set(train.subject_id).isdisjoint(set(val.subject_id))
    assert set(train.subject_id).isdisjoint(set(test.subject_id))
    assert set(val.subject_id).isdisjoint(set(test.subject_id))
    return train, val, test

# ---------------------------------------------------------------------
# Robust preprocessing: numeric median is ONLY applied to numeric columns.
# ---------------------------------------------------------------------
def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No usable model features were found.")
    return ColumnTransformer(transformers, remainder="drop")
