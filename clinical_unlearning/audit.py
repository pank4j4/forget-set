"""Unlearning audit: utility, memorisation gap, pass/fail verdict and report."""
from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

from clinical_unlearning.config import get_label
from clinical_unlearning.metrics import metrics_for
from clinical_unlearning.sisa import sisa_predict


# ---------------------------------------------------------------------
# Audit helpers: utility, memorisation gap, verdict and report
# ---------------------------------------------------------------------
def _auc_or_none(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def audit_model(bundle, train_df, test_df, forget_ids, retain_cap=5000):
    """
    Utility on held-out test patients, plus the classic unlearning check: how well does the model
    predict the FORGOTTEN patients' own records versus unseen patients? A model that memorised the
    forgotten patients scores much higher on them than on the test set (the memorisation gap).
    """
    LABEL = get_label()
    forget_set = set(int(x) for x in forget_ids)
    forget_rows = train_df[train_df.subject_id.isin(forget_set)]
    retain_rows = train_df[~train_df.subject_id.isin(forget_set)]
    if len(retain_rows) > retain_cap:
        retain_rows = retain_rows.sample(n=retain_cap, random_state=5)
    p_test = sisa_predict(bundle, test_df)
    test_m = metrics_for(test_df[LABEL], p_test)
    f_auc = _auc_or_none(forget_rows[LABEL], sisa_predict(bundle, forget_rows))
    r_auc = _auc_or_none(retain_rows[LABEL], sisa_predict(bundle, retain_rows))
    return {
        "Test AUROC": test_m["AUROC"], "Test F1": test_m["F1"],
        "Forget-set AUROC": f_auc, "Retain-set AUROC": r_auc,
        "Memorisation gap": (f_auc - test_m["AUROC"]) if (f_auc is not None and test_m["AUROC"] is not None) else None,
        "test_probs": p_test
    }


def build_verdict(res):
    """Pass/fail checks with thresholds fixed in advance (not tuned on the results)."""
    A, M, T = res["audits"], res["mia"], res["times"]
    orig, sisa = A["Original"], A["SISA unlearned"]
    checks = []
    def add(name, status, detail):
        checks.append({"Check": name, "Result": status, "Evidence": detail})

    if "Original" in M and "SISA unlearned" in M:
        cal = res["calibration_auc"]
        mb, ma = M["Original"], M["SISA unlearned"]
        valid = cal is not None and cal > 0.55 and mb["auc"] > 0.60
        add("Attack works before unlearning", "PASS" if valid else "INCONCLUSIVE",
            f"calibration AUC {cal:.2f}, attack AUC on forgotten vs non-members {mb['auc']:.2f} (needs > 0.60)")
        if valid:
            add("Forgotten patients indistinguishable afterwards",
                "PASS" if ma["auc"] <= 0.60 else "FAIL",
                f"attack AUC {mb['auc']:.2f} -> {ma['auc']:.2f} (needs <= 0.60)")
            add("Detection falls to the false-positive floor",
                "PASS" if ma["forget_detected_rate"] <= ma["nonmember_false_positive_rate"] + 0.05 else "FAIL",
                f"forgotten flagged {mb['forget_detected_rate']:.1%} -> {ma['forget_detected_rate']:.1%}; non-members falsely flagged {ma['nonmember_false_positive_rate']:.1%}")
    else:
        add("Privacy attack", "SKIPPED", "Run the attack BEFORE unlearning (MIA Simulation, Stage 1) to include it")

    og, sg = orig["Memorisation gap"], sisa["Memorisation gap"]
    if og is None or sg is None:
        add("Memorisation gap closed", "SKIPPED", "forgotten records contain a single outcome class")
    elif og <= 0.05:
        add("Memorisation gap closed", "INCONCLUSIVE", f"original model did not memorise the forgotten patients (gap {og:+.3f})")
    else:
        add("Memorisation gap closed", "PASS" if sg <= 0.05 else "FAIL",
            f"forget-set AUROC minus test AUROC: {og:+.3f} -> {sg:+.3f} (needs <= +0.050)")

    if orig["Test AUROC"] is not None and sisa["Test AUROC"] is not None:
        drop = orig["Test AUROC"] - sisa["Test AUROC"]
        add("Utility preserved", "PASS" if drop <= 0.02 else "FAIL",
            f"test AUROC {orig['Test AUROC']:.3f} -> {sisa['Test AUROC']:.3f} (drop {drop:+.3f}, allowed 0.020)")

    if "Retrain from scratch" in A and A["Retrain from scratch"]["Test AUROC"] is not None and sisa["Test AUROC"] is not None:
        diff = abs(A["Retrain from scratch"]["Test AUROC"] - sisa["Test AUROC"])
        add("Matches retraining from scratch", "PASS" if diff <= 0.02 else "FAIL",
            f"test AUROC difference vs gold standard {diff:.3f} (allowed 0.020)")

    done, total = res["slices_replayed"], res["slices_total"]
    add("Compute saved vs retraining everything", "PASS" if done < total else "NO SAVING",
        f"replayed {done} of {total} slice-trainings ({done / max(1, total):.0%})"
        + (f"; {T['SISA unlearned']:.1f}s vs {T['Retrain from scratch']:.1f}s" if "Retrain from scratch" in T else ""))
    return checks


def build_report(res, cfg):
    def f(v, kind="num"):
        if v is None: return "n/a"
        return f"{v:.1%}" if kind == "pct" else f"{v:.3f}"
    lines = ["# Machine unlearning audit report", "",
             f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
             "## Configuration",
             f"- Prediction task: {cfg['task']}",
             f"- Shard learner: {cfg['model_type']} · {cfg['n_shards']} shards · {cfg['n_slices']} slices · {cfg['epochs']} passes",
             f"- Forget Set: {res['forget_patients']} patients / {res['forget_admissions']} admissions",
             f"- Shards touched: {len(res['affected_shards'])} of {cfg['n_shards']}", "",
             "## Verdict", "", "| Check | Result | Evidence |", "|---|---|---|"]
    for c in res["verdict"]:
        lines.append(f"| {c['Check']} | {c['Result']} | {c['Evidence']} |")
    lines += ["", "## Model comparison", "", "| Metric | " + " | ".join(res["audits"].keys()) + " |",
              "|---|" + "---|" * len(res["audits"])]
    for label, key in [("Test AUROC", "Test AUROC"), ("Test F1", "Test F1"), ("Forget-set AUROC", "Forget-set AUROC"),
                       ("Retain-set AUROC", "Retain-set AUROC"), ("Memorisation gap", "Memorisation gap")]:
        lines.append(f"| {label} | " + " | ".join(f(a[key]) for a in res["audits"].values()) + " |")
    if res["mia"]:
        lines.append("| Attack AUC (forgotten vs non-members) | " + " | ".join(f(res['mia'][k]['auc']) if k in res['mia'] else "n/a" for k in res["audits"]) + " |")
        lines.append("| Forgotten flagged as members | " + " | ".join(f(res['mia'][k]['forget_detected_rate'], 'pct') if k in res['mia'] else "n/a" for k in res["audits"]) + " |")
        lines.append("| Non-member false-positive rate | " + " | ".join(f(res['mia'][k]['nonmember_false_positive_rate'], 'pct') if k in res['mia'] else "n/a" for k in res["audits"]) + " |")
    lines += ["", "## Limitations",
              "- The privacy result describes one shadow-model attack, not every possible attack.",
              "- Retraining affected shards gives exact unlearning for the ensemble, but does not remove data from source files, backups or logs.",
              "- Detection after unlearning cannot go below the attacker's false-positive rate on true non-members.",
              "- Results come from a single run; repeat with different Forget Sets before drawing conclusions.", ""]
    return "\n".join(lines)
