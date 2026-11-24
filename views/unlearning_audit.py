"""Page: Unlearning Audit"""
from __future__ import annotations

import html as htmllib
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from clinical_unlearning.audit import audit_model, build_report, build_verdict
from clinical_unlearning.config import LEARNER_NAMES, TASKS, get_label
from clinical_unlearning.mia import evaluate_mia_attack, mia_shard_routes
from clinical_unlearning.sisa import train_sisa, unlearn_sisa
from clinical_unlearning.storage import audit, save_experiment
from ui.components import empty_state, page_header, plot, section_title, task_short


def render(df):
    LABEL = get_label()
    page_header("VERIFICATION WORKSPACE", "Unlearning & Audit", "Run selective SISA unlearning, then audit it against the gold standard: a model retrained from scratch without the forgotten patients.")
    if st.session_state.baseline is None:
        empty_state("🧠", "Train the original model first", "Train the original SISA model in Model Training to continue.")
    elif not st.session_state.forget_ids:
        empty_state("🗑️", "No Forget Set yet", "Confirm a Forget Set in Erasure Requests first.")
    else:
        base = st.session_state.baseline
        tr, te = st.session_state.train, st.session_state.test
        forget_ids = [int(x) for x in st.session_state.forget_ids]
        forget_rows = tr[tr.subject_id.isin(forget_ids)]
        retain_rows = tr[~tr.subject_id.isin(forget_ids)]
        touched = sum(1 for sh in base["shards"] if set(forget_ids).intersection(sh["patient_ids"]))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients to forget", f"{len(forget_ids):,}")
        c2.metric("Admissions to forget", f"{len(forget_rows):,}")
        c3.metric("Shards touched", f"{touched} of {base['n_shards']}")
        c4.metric("Privacy attack locked", "Yes" if st.session_state.mia_attack_data is not None else "No")
        if st.session_state.mia_attack_data is None:
            st.info("Tip: run the attack BEFORE unlearning (MIA Simulation, Stage 1) so this audit can include the privacy test. Without it only utility and memorisation checks are shown.")
        _lrn = LEARNER_NAMES.get(base.get("model_type", "sgd"), "model")
        _ep = "" if base.get("model_type") == "forest" else f" · {base['epochs']} epochs/slice"
        st.caption(f"Configuration: {_lrn} · {base['n_shards']} shards · {base['n_slices']} slice(s) per shard{_ep} · task: {task_short()}")
        do_retrain = st.checkbox(
            "Also train a retrain-from-scratch reference (gold standard)", value=True,
            help="Trains a fresh ensemble on the retained patients only. Unlearning is judged by how closely it matches this model, and by how much cheaper it is.")

        if st.button("Run SISA unlearning and audit", type="primary"):
            try:
                if forget_rows.empty or retain_rows.empty:
                    raise ValueError("Forget Set or Retain Set is empty.")
                with st.spinner("Unlearning: replaying only the affected shards and slices..."):
                    t0 = time.time()
                    updated, affected, replayed = unlearn_sisa(base, tr, forget_ids)
                    unlearn_seconds = time.time() - t0
                retrain_bundle, retrain_seconds = None, None
                if do_retrain:
                    with st.spinner("Training the retrain-from-scratch reference..."):
                        t0 = time.time()
                        retrain_bundle = train_sisa(retain_rows, base["feature_cols"], base["n_shards"], base["n_slices"],
                                                    base["epochs"], base["model_type"], base.get("n_trees", 100))
                        retrain_seconds = time.time() - t0
                with st.spinner("Auditing utility, memorisation and privacy..."):
                    models = {"Original": base, "SISA unlearned": updated}
                    if retrain_bundle is not None:
                        models["Retrain from scratch"] = retrain_bundle
                    audits = {name: audit_model(m, tr, te, forget_ids) for name, m in models.items()}
                    mia, cal_auc = {}, None
                    ad = st.session_state.mia_attack_data
                    if ad is not None:
                        cal_auc = ad["thresholds"]["calibration_auc"]
                        args = (ad["forgotten_candidates"], ad["nonmember_controls"], ad["retained_controls"], ad["thresholds"])
                        mia["Original"] = st.session_state.mia_before
                        mia["SISA unlearned"] = evaluate_mia_attack(
                            updated, *args, ad["forget_routes"], ad["nonmember_routes"], ad["retained_routes"])
                        if retrain_bundle is not None:
                            # The retrained model never saw the forgotten patients: route them like non-members.
                            mia["Retrain from scratch"] = evaluate_mia_attack(
                                retrain_bundle, *args,
                                mia_shard_routes(retrain_bundle, ad["forgotten_candidates"], nonmember=True),
                                ad["nonmember_routes"],
                                mia_shard_routes(retrain_bundle, ad["retained_controls"]))
                total_slices = int(sum(len(sh["slice_indices"]) for sh in base["shards"]))
                times = {"Original": (st.session_state.baseline_metrics or {}).get("Training_seconds"),
                         "SISA unlearned": unlearn_seconds}
                if retrain_seconds is not None:
                    times["Retrain from scratch"] = retrain_seconds
                shard_table = []
                for sh in base["shards"]:
                    sid = sh["shard_id"]
                    lg = updated["unlearn_log"].get(sid)
                    shard_table.append({
                        "Shard": sid + 1,
                        "Patients forgotten": lg["patients_removed"] if lg else 0,
                        "Admissions removed": lg["rows_removed"] if lg else 0,
                        "Replay resumed from slice": ("from scratch" if lg["resume_from"] == 0 else f"{lg['resume_from'] + 1}") if lg else "not touched",
                        "Slices retrained": f"{lg['slices_retrained']} of {lg['slices_total']}" if lg else f"0 of {len(sh['slice_indices'])}",
                    })
                res = {"audits": audits, "mia": mia, "times": times, "calibration_auc": cal_auc,
                       "affected_shards": affected, "slices_replayed": len(replayed), "slices_total": total_slices,
                       "forget_patients": len(forget_ids), "forget_admissions": len(forget_rows),
                       "shard_table": shard_table}
                res["verdict"] = build_verdict(res)
                st.session_state.unlearned = updated
                st.session_state.mia_after = mia.get("SISA unlearned")
                st.session_state.last_result = res
                sisa_metrics = {k: v for k, v in audits["SISA unlearned"].items() if k != "test_probs"}
                sisa_metrics.update({"Unlearning_seconds": unlearn_seconds, "Slices_replayed": len(replayed), "Slices_total": total_slices})
                save_experiment("SISA selective replay + audit", len(forget_ids), len(forget_rows), "Completed", sisa_metrics)
                audit("sisa_unlearning_completed", {"affected_shards": affected, "slices_replayed": len(replayed),
                                                    "slices_total": total_slices, "verdict": [(c["Check"], c["Result"]) for c in res["verdict"]]})
                st.success("Unlearning and audit completed.")
            except Exception as exc:
                st.error(f"Unlearning failed: {exc}")

        res = st.session_state.last_result
        if res is not None:
            verdict = res["verdict"]
            n_pass = sum(c["Result"] == "PASS" for c in verdict)
            n_fail = sum(c["Result"] == "FAIL" for c in verdict)
            n_eval = sum(c["Result"] in ("PASS", "FAIL") for c in verdict)
            if n_fail == 0 and n_pass > 0:
                vcls, vbig = "ok", "✅ Unlearning verified"
                vtxt = f"{n_pass} of {n_eval} evaluated checks passed. Unlearning behaved like retraining without the forgotten patients, as far as these tests can tell."
            elif n_fail > 0:
                vcls, vbig = "bad", "❌ Needs attention"
                vtxt = f"{n_fail} of {n_eval} evaluated checks failed. See the evidence below."
            else:
                vcls, vbig = "neutral", "⚠️ Not evaluated"
                vtxt = "No check could be evaluated. See the evidence below."
            st.markdown(f'<div class="vbanner {vcls}"><div class="big">{vbig}</div><div class="txt">{htmllib.escape(vtxt)}</div></div>', unsafe_allow_html=True)

            tab_v, tab_c, tab_k, tab_x = st.tabs(["Verdict", "Model comparison", "Cost of unlearning", "Export"])

            with tab_v:
                chip = {"PASS": ("pass", "PASS"), "FAIL": ("fail", "FAIL"), "INCONCLUSIVE": ("warn", "INCONCLUSIVE"),
                        "SKIPPED": ("skip", "SKIPPED"), "NO SAVING": ("info", "NO SAVING")}
                body = "".join(
                    f'<tr><td><b>{htmllib.escape(c["Check"])}</b></td>'
                    f'<td><span class="chip {chip.get(c["Result"], ("skip", c["Result"]))[0]}">{htmllib.escape(c["Result"])}</span></td>'
                    f'<td>{htmllib.escape(c["Evidence"])}</td></tr>' for c in verdict)
                st.markdown(f'<table class="vt"><thead><tr><th>Check</th><th>Result</th><th>Evidence</th></tr></thead><tbody>{body}</tbody></table>',
                            unsafe_allow_html=True)
                st.caption("Thresholds are fixed in advance and are not tuned on these results. 'No saving' is informational: a request that touches every shard cannot skip any retraining.")

            with tab_c:
                section_title("Original vs SISA unlearned vs retrain from scratch")
                def cell(v, kind="num"):
                    if v is None: return "—"
                    return f"{v:.1%}" if kind == "pct" else (f"{v:.1f}s" if kind == "sec" else f"{v:.3f}")
                names = list(res["audits"].keys())
                labels = ["Test AUROC", "Test F1", "Forget-set AUROC", "Retain-set AUROC", "Memorisation gap (forget − test)",
                          "Attack AUC (forgotten vs non-members)", "Forgotten flagged as members", "Non-member false-positive rate",
                          "Time (training / unlearning / retraining)"]
                table = {"Metric": labels}
                for n in names:
                    a_, m_ = res["audits"][n], res["mia"].get(n)
                    table[n] = [cell(a_["Test AUROC"]), cell(a_["Test F1"]), cell(a_["Forget-set AUROC"]), cell(a_["Retain-set AUROC"]),
                                cell(a_["Memorisation gap"]), cell(m_["auc"] if m_ else None),
                                cell(m_["forget_detected_rate"] if m_ else None, "pct"),
                                cell(m_["nonmember_false_positive_rate"] if m_ else None, "pct"), cell(res["times"].get(n), "sec")]
                st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
                st.caption("Memorisation gap: how much better the model predicts the forgotten patients' own records than unseen patients. A model that memorised them has a large gap; after real unlearning it should match the retrained model (≈ 0).")
                chart_rows = []
                for n in names:
                    for k in ["Test AUROC", "Forget-set AUROC", "Retain-set AUROC"]:
                        if res["audits"][n][k] is not None:
                            chart_rows.append({"Model": n, "Evaluated on": k.replace(" AUROC", ""), "AUROC": res["audits"][n][k]})
                plot(px.bar(pd.DataFrame(chart_rows), x="Evaluated on", y="AUROC", color="Model", barmode="group",
                            title="Predictive performance on forgotten, retained and unseen patients",
                            color_discrete_map={"Original": "#8a99b0", "SISA unlearned": "#0f7f86", "Retrain from scratch": "#1f4e8c"},
                            range_y=[0.4, 1.0]), use_container_width=True)

            with tab_k:
                section_title("Cost of unlearning", "What SISA saved compared with retraining everything")
                k1, k2, k3 = st.columns(3)
                k1.metric("Shards retrained", f"{len(res['affected_shards'])} of {len(res['shard_table'])}")
                k2.metric("Slice-trainings replayed", f"{res['slices_replayed']} of {res['slices_total']}",
                          delta=f"{res['slices_replayed'] / max(1, res['slices_total']) - 1:+.0%} vs full retrain", delta_color="inverse")
                if "Retrain from scratch" in res["times"]:
                    k3.metric("Speed-up vs retraining", f"{res['times']['Retrain from scratch'] / max(1e-9, res['times']['SISA unlearned']):.1f}×")
                st.dataframe(pd.DataFrame(res["shard_table"]), use_container_width=True, hide_index=True)
                st.caption("SISA saves work in two ways: shards without forgotten patients are not touched, and sliced learners resume from the checkpoint just before the first affected slice. A request that touches every shard (a large Forget Set) saves little; a single-patient request typically retrains one shard.")

            with tab_x:
                section_title("Export", "Take the results into your report")
                cfg = {"task": next((k for k, v in TASKS.items() if v == LABEL), LABEL),
                       "model_type": {"forest": "random forest", "mlp": "neural network", "sgd": "logistic regression (SGD)"}.get(base.get("model_type", "sgd")),
                       "n_shards": base["n_shards"], "n_slices": base["n_slices"], "epochs": base["epochs"]}
                report_md = build_report(res, cfg)
                d1, d2 = st.columns(2)
                d1.download_button("Download audit report (Markdown)", report_md.encode("utf-8"), "unlearning_audit_report.md", "text/markdown")
                d2.download_button("Download comparison table (CSV)", pd.DataFrame(table).to_csv(index=False).encode("utf-8"), "unlearning_comparison.csv", "text/csv")
                st.caption("Retraining the affected shards gives exact unlearning for the ensemble but does not remove data from source CSVs, backups, logs or external copies.")
