"""SISA ensemble: sharded, sliced training, prediction and selective unlearning."""
from __future__ import annotations

import copy

import numpy as np

from clinical_unlearning.config import get_label
from clinical_unlearning.data import make_preprocessor
from clinical_unlearning.learners import make_forest, make_incremental, positive_proba


# ---------------------------------------------------------------------
# SISA implementation
# ---------------------------------------------------------------------
def train_sisa(train_df, feature_cols, n_shards=5, n_slices=4, epochs=3,
               model_type="sgd", n_trees=100):
    """
    Patient-level SISA prototype:
    - Assign whole patients to shards.
    - Sort each shard deterministically and divide into slices.
    - SGD shards: save a model checkpoint after each slice.
    - Forest shards: tree ensembles have no partial_fit, so each shard is a
      single slice; unlearning retrains the affected shard from scratch.
    - During unlearning, only shards containing forgotten patients are rebuilt.
    """
    LABEL = get_label()
    X_all = train_df[feature_cols].copy()
    preprocessor = make_preprocessor(X_all)
    X_all_t = preprocessor.fit_transform(X_all)
    X_all_t = np.asarray(X_all_t, dtype=np.float64)
    y_all = train_df[LABEL].to_numpy(dtype=int)

    # Deterministic patient-level shard assignment.
    patients = np.array(sorted(train_df.subject_id.unique()))
    rng = np.random.default_rng(42)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    shard_patient_ids = np.array_split(shuffled, min(n_shards, len(shuffled)))

    models, shard_meta = [], []
    for shard_index, patient_ids in enumerate(shard_patient_ids):
        row_idx = np.flatnonzero(train_df.subject_id.isin(patient_ids).to_numpy())
        # Stable ordering within shard; slices are contiguous groups of rows.
        row_idx = row_idx[np.argsort(train_df.iloc[row_idx].subject_id.to_numpy(), kind="stable")]

        if model_type == "forest":
            clf = make_forest(n_trees)
            clf.fit(X_all_t[row_idx], y_all[row_idx])
            slice_indices = [row_idx]
            checkpoints = [{
                "slice_id": 0,
                "last_patient_id": int(train_df.iloc[row_idx].subject_id.max()),
                "model": None  # the shard model itself lives in `models`
            }]
        else:
            # Slice by PATIENT (never split one patient's admissions across slices).
            shard_subjects = train_df.iloc[row_idx].subject_id.to_numpy()
            ordered_patients = np.unique(shard_subjects)
            groups = np.array_split(ordered_patients, min(n_slices, max(1, len(ordered_patients))))
            slice_indices = [row_idx[np.isin(shard_subjects, g)] for g in groups if len(g)]
            clf = make_incremental(model_type, y_all[row_idx])
            checkpoints = []
            first_call = True
            for slice_id, indices in enumerate(slice_indices):
                for _ in range(epochs):
                    if first_call:
                        clf.partial_fit(X_all_t[indices], y_all[indices], classes=np.array([0, 1]))
                        first_call = False
                    else:
                        clf.partial_fit(X_all_t[indices], y_all[indices])
                checkpoints.append({
                    "slice_id": slice_id,
                    "last_patient_id": int(train_df.iloc[indices].subject_id.max()),
                    "model": copy.deepcopy(clf)
                })
        models.append(clf)
        shard_meta.append({
            "shard_id": shard_index,
            "patient_ids": [int(x) for x in patient_ids],
            "row_indices": row_idx.tolist(),
            "slice_indices": [x.tolist() for x in slice_indices],
            "checkpoints": checkpoints
        })
    return {
        "preprocessor": preprocessor,
        "models": models,
        "shards": shard_meta,
        "feature_cols": feature_cols,
        "n_shards": len(shard_meta),
        "n_slices": 1 if model_type == "forest" else n_slices,
        "epochs": epochs,
        "model_type": model_type,
        "n_trees": int(n_trees),
        "class_weight_strategy": "balanced_per_shard_explicit"
    }

def sisa_predict(model_bundle, frame):
    X = np.asarray(model_bundle["preprocessor"].transform(frame[model_bundle["feature_cols"]]), dtype=float)
    shard_probs = [positive_proba(clf, X) for clf in model_bundle["models"]]
    return np.mean(np.vstack(shard_probs), axis=0)

def unlearn_sisa(bundle, train_df, forget_ids, epochs=None):
    """
    Selective SISA unlearning.
      * Shards that contain no forgotten patient are reused unchanged.
      * Linear / neural shards (sliced): find the FIRST slice containing a forgotten patient,
        restore the saved checkpoint taken just before it, and replay only the remaining slices
        (without the forgotten rows). Earlier slices are never retrained.
      * Forest shards (no incremental training, one slice): the shard is retrained.
    `unlearn_log` records, per shard, where replay resumed and how many slices were retrained.
    """
    LABEL = get_label()
    model_type = bundle.get("model_type", "sgd")
    n_trees = bundle.get("n_trees", 100)
    updated = {
        "preprocessor": bundle["preprocessor"],
        "models": list(bundle["models"]),
        "shards": [dict(s) for s in bundle["shards"]],
        "feature_cols": list(bundle["feature_cols"]),
        "n_shards": bundle["n_shards"],
        "n_slices": bundle["n_slices"],
        "epochs": bundle["epochs"],
        "model_type": model_type,
        "n_trees": n_trees,
        "class_weight_strategy": bundle.get("class_weight_strategy", "balanced_per_shard_explicit"),
        "original_patient_shard_map": dict(bundle.get("original_patient_shard_map", {})),
        "unlearn_log": {}
    }
    X_all = np.asarray(bundle["preprocessor"].transform(train_df[bundle["feature_cols"]]), dtype=float)
    y_all = train_df[LABEL].to_numpy(dtype=int)
    subj_all = train_df["subject_id"].to_numpy()
    forget_set = set(int(x) for x in forget_ids)
    affected, replayed = [], []
    repeat = epochs or bundle["epochs"]

    for i, shard in enumerate(bundle["shards"]):
        if not forget_set.intersection(shard["patient_ids"]):
            continue
        affected.append(i)
        old_rows = np.asarray(shard["row_indices"])
        keep_mask = ~np.isin(subj_all[old_rows], list(forget_set))
        kept_global = old_rows[keep_mask]
        kept_global = kept_global[np.argsort(subj_all[kept_global], kind="stable")]
        if len(kept_global) == 0:
            raise ValueError(f"Shard {i} has no retained training records after forgetting.")
        remaining_patients = [int(x) for x in shard["patient_ids"] if int(x) not in forget_set]
        old_slices = [np.asarray(x) for x in shard["slice_indices"]]

        if model_type == "forest":
            clf = make_forest(n_trees)
            clf.fit(X_all[kept_global], y_all[kept_global])
            new_checkpoints = [{"slice_id": 0, "last_patient_id": int(subj_all[kept_global].max()), "model": None}]
            new_slices = [kept_global]
            resume = 0
            replayed.append((i, 0))
        else:
            kept_slices = [ix[~np.isin(subj_all[ix], list(forget_set))] for ix in old_slices]
            first_hit = next((j for j, ix in enumerate(old_slices)
                              if np.isin(subj_all[ix], list(forget_set)).any()), 0)
            ckpt = shard["checkpoints"][first_hit - 1].get("model") if first_hit > 0 else None
            if ckpt is None:
                clf = make_incremental(model_type, y_all[kept_global])
                first_call, resume = True, 0
                new_checkpoints, new_slices = [], []
            else:
                # Restore the model exactly as it was before the first affected slice.
                clf = copy.deepcopy(ckpt)
                first_call, resume = False, first_hit
                new_checkpoints = [dict(c) for c in shard["checkpoints"][:first_hit]]
                new_slices = [ix for ix in old_slices[:first_hit]]
            for j in range(resume, len(old_slices)):
                indices = kept_slices[j]
                if len(indices) == 0:
                    continue
                for _ in range(repeat):
                    if first_call:
                        clf.partial_fit(X_all[indices], y_all[indices], classes=np.array([0, 1]))
                        first_call = False
                    else:
                        clf.partial_fit(X_all[indices], y_all[indices])
                new_checkpoints.append({
                    "slice_id": len(new_slices),
                    "last_patient_id": int(subj_all[indices].max()),
                    "model": copy.deepcopy(clf)
                })
                new_slices.append(indices)
                replayed.append((i, j))

        updated["models"][i] = clf
        updated["shards"][i]["checkpoints"] = new_checkpoints
        updated["shards"][i]["patient_ids"] = remaining_patients
        updated["shards"][i]["row_indices"] = kept_global.tolist()
        updated["shards"][i]["slice_indices"] = [x.tolist() for x in new_slices]
        updated["unlearn_log"][i] = {
            "resume_from": int(resume),
            "slices_retrained": int(sum(1 for (si, _) in replayed if si == i)),
            "slices_total": int(len(old_slices)),
            "rows_removed": int(len(old_rows) - len(kept_global)),
            "patients_removed": int(len(shard["patient_ids"]) - len(remaining_patients)),
        }
    return updated, affected, replayed
