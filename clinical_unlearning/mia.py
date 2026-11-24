"""Shadow-model membership inference attack (Shokri et al., 2017)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from clinical_unlearning.config import get_label
from clinical_unlearning.data import make_preprocessor
from clinical_unlearning.learners import make_forest, make_incremental, positive_proba


# ---------------------------------------------------------------------
# MIA: Shokri-style shadow-model membership inference.
#
# Following Shokri et al. (2017), the attacker never sees the target's
# training set. Instead it:
#   1. trains shadow models (same learner type) on its own auxiliary data
#      (here: the validation patients, disjoint from train and test),
#   2. records each shadow model's output on its own members vs non-members,
#   3. trains a binary attack classifier on those (output -> in/out) pairs,
#   4. applies the classifier to the target model's outputs.
# A separate held-out shadow model calibrates a fixed three-state decision
# band. Attacker, candidates, shard routes and band are NOT changed after
# unlearning, so any drop in detection comes from the model itself.
# ---------------------------------------------------------------------
def patient_shard_map(model_bundle):
    """Original patient -> SISA shard assignment, preserved after unlearning."""
    mapping = {}
    for shard in model_bundle["shards"]:
        for patient_id in shard["patient_ids"]:
            mapping[int(patient_id)] = int(shard["shard_id"])
    # The original assignment is also retained separately so forgotten
    # patients can still be routed to their original shard after erasure.
    mapping.update(model_bundle.get("original_patient_shard_map", {}))
    return mapping


def mia_shard_routes(model_bundle, frame, nonmember=False):
    """
    Route each record to exactly one shard for the membership test.
    Members use their actual original training shard. Non-members are
    assigned deterministically and evenly across shards as matched controls.
    """
    n_shards = int(model_bundle["n_shards"])
    if nonmember:
        return np.arange(len(frame), dtype=int) % n_shards
    mapping = patient_shard_map(model_bundle)
    missing = [int(x) for x in frame["subject_id"].unique() if int(x) not in mapping]
    if missing:
        raise ValueError(
            "No original SISA shard assignment exists for one or more member "
            "candidates. Re-train the original model before running MIA."
        )
    return frame["subject_id"].astype(int).map(mapping).to_numpy(dtype=int)


def attack_features(p_pos, y):
    """Per-record attack features from a model's output and the true label."""
    p_pos = np.clip(np.asarray(p_pos, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    p_true = np.where(y == 1, p_pos, 1.0 - p_pos)
    entropy = -(p_pos * np.log(p_pos) + (1 - p_pos) * np.log(1 - p_pos))
    return np.column_stack([np.log(p_true), p_true, entropy, y.astype(float), p_pos])


def _fit_shadow(model_type, X, y, n_trees, epochs):
    if model_type == "forest":
        clf = make_forest(n_trees)
        clf.fit(X, y)
        return clf
    clf = make_incremental(model_type, y)
    for e in range(max(1, int(epochs))):
        if e == 0:
            clf.partial_fit(X, y, classes=np.array([0, 1]))
        else:
            clf.partial_fit(X, y)
    return clf


def _matched_eval_rows(subj, y, in_ids, out_ids, first_row, rng, per_class_cap=400):
    """Equal numbers of y=1 and y=0 admissions for members and non-members."""
    in_rows = np.flatnonzero(np.isin(subj, in_ids))
    out_rows = np.flatnonzero(np.isin(subj, out_ids))
    n1 = min(int(np.sum(y[in_rows] == 1)), int(np.sum(y[out_rows] == 1)), per_class_cap)
    if n1 < 5:  # too few positives: fall back to one admission per patient
        return first_row.loc[in_ids].to_numpy(), first_row.loc[out_ids].to_numpy()
    def pick(rows):
        pos = rng.choice(rows[y[rows] == 1], n1, replace=False)
        neg = rng.choice(rows[y[rows] == 0], n1, replace=False)
        return np.concatenate([pos, neg])
    return pick(in_rows), pick(out_rows)


def build_shokri_attacker(model_bundle, attacker_df, n_shadow=3, seed=2024, confidence=0.75):
    """
    Train shadow models on the attacker's auxiliary patients, fit the attack
    classifier, and calibrate a fixed decision band on one extra held-out
    shadow model. Returns a dict used as the attacker everywhere else.
    """
    LABEL = get_label()
    feature_cols = model_bundle["feature_cols"]
    model_type = model_bundle.get("model_type", "sgd")
    n_trees = model_bundle.get("n_trees", 100)
    epochs = model_bundle.get("epochs", 3)

    attacker_df = attacker_df.reset_index(drop=True)
    subj = attacker_df["subject_id"].to_numpy(dtype=np.int64)
    y_att = attacker_df[LABEL].to_numpy(dtype=int)
    pids = np.array(sorted(np.unique(subj)))
    if len(pids) < 40:
        raise ValueError(
            "The attacker's auxiliary (validation) data has fewer than 40 patients. "
            "Use a larger dataset to build shadow models."
        )

    # The attacker fits its own preprocessing on its own data.
    prep = make_preprocessor(attacker_df[feature_cols])
    X_att = np.asarray(prep.fit_transform(attacker_df[feature_cols]), dtype=np.float64)
    first_row = pd.Series(np.arange(len(subj)), index=subj).groupby(level=0).first()

    # Match the shadow training size to a target shard where possible.
    avg_shard = int(np.mean([max(1, len(s["patient_ids"])) for s in model_bundle["shards"]]))
    shadow_n = max(10, min(avg_shard, len(pids) // 2))

    train_feats, train_labels = [], []
    cal_feats, cal_labels = [], []
    for s in range(n_shadow + 1):  # the last shadow is only used for calibration
        rng = np.random.default_rng(seed + s)
        perm = rng.permutation(pids)
        in_ids = perm[:shadow_n]
        out_ids = perm[shadow_n:2 * shadow_n]
        in_rows = np.flatnonzero(np.isin(subj, in_ids))
        shadow = _fit_shadow(model_type, X_att[in_rows], y_att[in_rows], n_trees, epochs)

        # Class-balanced, size-matched evaluation sets. With ~1-2% mortality a random
        # sample is almost all survivors, which hides the (rare-class) memorisation signal
        # and lets the attacker use the outcome label as a proxy for membership.
        in_eval, out_eval = _matched_eval_rows(subj, y_att, in_ids, out_ids, first_row, rng)
        f_in = attack_features(positive_proba(shadow, X_att[in_eval]), y_att[in_eval])
        f_out = attack_features(positive_proba(shadow, X_att[out_eval]), y_att[out_eval])
        feats = np.vstack([f_in, f_out])
        labels = np.concatenate([np.ones(len(f_in)), np.zeros(len(f_out))])
        if s < n_shadow:
            train_feats.append(feats)
            train_labels.append(labels)
        else:
            cal_feats, cal_labels = feats, labels

    attack_model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.1, random_state=42
    )
    attack_model.fit(np.vstack(train_feats), np.concatenate(train_labels))

    cal_scores = attack_model.predict_proba(cal_feats)[:, 1]
    member_scores = cal_scores[cal_labels == 1]
    nonmember_scores = cal_scores[cal_labels == 0]
    if min(len(member_scores), len(nonmember_scores)) < 5:
        raise ValueError("At least five calibration members and non-members are required.")
    auc = float(roc_auc_score(cal_labels, cal_scores))

    # Fixed three-state band on the attacker's membership posterior P(member | output),
    # estimated with a balanced 50/50 prior. The band is set by how confident the attacker
    # must be BEFORE it makes a claim, never by looking at the results:
    #   score >  confidence      : "Likely member"
    #   score <  1 - confidence  : "Likely non-member"
    #   otherwise                : "Inconclusive"  (no usable evidence either way)
    # A score near 0.5 is a coin flip, so it is never reported as a detection.
    confidence = float(min(max(confidence, 0.55), 0.95))
    high = confidence
    low = 1.0 - confidence
    return {
        "attack_model": attack_model,
        "low": low, "high": high, "calibration_auc": auc,
        "calibration_member_scores": member_scores,
        "calibration_nonmember_scores": nonmember_scores,
        "shadow_model_type": model_type,
        "n_shadow": n_shadow,
        "shadow_train_patients": int(shadow_n)
    }


def mia_scores(model_bundle, frame, shard_routes, attacker):
    """
    Score each candidate with the attacker. A training member is queried
    against the exact shard that trained it; after unlearning the same
    candidate is queried against that same (rebuilt) shard. Non-members are
    routed to matched shards. Higher score = stronger evidence of membership.
    """
    LABEL = get_label()
    if len(frame) == 0:
        return np.array([], dtype=float)
    X = np.asarray(
        model_bundle["preprocessor"].transform(frame[model_bundle["feature_cols"]]),
        dtype=float
    )
    y = frame[LABEL].to_numpy(dtype=int)
    shard_routes = np.asarray(shard_routes, dtype=int)
    if len(shard_routes) != len(frame):
        raise ValueError("Each candidate must have exactly one shard route.")

    probs = np.empty(len(frame), dtype=float)
    for shard_id in np.unique(shard_routes):
        mask = shard_routes == shard_id
        clf = model_bundle["models"][int(shard_id)]
        probs[mask] = positive_proba(clf, X[mask])
    return attacker["attack_model"].predict_proba(attack_features(probs, y))[:, 1]


def classify_mia_scores(scores, thresholds):
    scores = np.asarray(scores, dtype=float)
    low, high = thresholds["low"], thresholds["high"]
    labels = np.full(len(scores), "Inconclusive", dtype=object)
    # Strict inequalities: many records share the attacker's "no evidence" score,
    # which can sit exactly on a band edge. Ties are NOT evidence of membership.
    labels[scores < low] = "Likely non-member"
    labels[scores > high] = "Likely member"
    return labels


def evaluate_mia_attack(model_bundle, forgotten_candidates, nonmember_controls,
                        retained_controls, thresholds, forget_routes,
                        nonmember_routes, retained_routes):
    """Evaluate identical records with the attacker's unchanged calibration."""
    forget_scores = mia_scores(model_bundle, forgotten_candidates, forget_routes, thresholds)
    nonmember_scores = mia_scores(model_bundle, nonmember_controls, nonmember_routes, thresholds)
    retained_scores = (
        mia_scores(model_bundle, retained_controls, retained_routes, thresholds)
        if len(retained_controls) else np.array([])
    )
    n = min(len(forget_scores), len(nonmember_scores))
    if n < 5:
        raise ValueError("At least five forgotten candidates and non-member controls are required.")
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    scores = np.concatenate([forget_scores[:n], nonmember_scores[:n]])
    forget_inference = classify_mia_scores(forget_scores, thresholds)
    nonmember_inference = classify_mia_scores(nonmember_scores, thresholds)
    retained_inference = (
        classify_mia_scores(retained_scores, thresholds)
        if len(retained_scores) else np.array([])
    )
    return {
        "forget_scores": forget_scores,
        "nonmember_scores": nonmember_scores,
        "retained_scores": retained_scores,
        "forget_inference": forget_inference,
        "nonmember_inference": nonmember_inference,
        "retained_inference": retained_inference,
        "auc": float(roc_auc_score(labels, scores)),
        "forget_detected_rate": float(np.mean(forget_inference == "Likely member")),
        "forget_nonmember_rate": float(np.mean(forget_inference == "Likely non-member")),
        "forget_inconclusive_rate": float(np.mean(forget_inference == "Inconclusive")),
        "nonmember_false_positive_rate": float(np.mean(nonmember_inference == "Likely member")),
        "retained_detected_rate": float(np.mean(retained_inference == "Likely member")) if len(retained_inference) else None,
        "threshold_low": float(thresholds["low"]),
        "threshold_high": float(thresholds["high"]),
        "n_forget": len(forget_scores),
        "n_nonmember": len(nonmember_scores),
        "n_retained": len(retained_scores)
    }
