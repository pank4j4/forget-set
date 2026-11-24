"""Page: Mia Simulation"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clinical_unlearning.config import get_label
from clinical_unlearning.mia import build_shokri_attacker, evaluate_mia_attack, mia_shard_routes
from clinical_unlearning.storage import audit
from ui.components import empty_state, page_header, plot, stage_header, style_inference


def render(df):
    LABEL = get_label()
    page_header("PRIVACY ATTACK WORKSPACE", "Membership Inference Attack", "Test whether an attacker can infer that a candidate patient record was used to train the model.")

    if st.session_state.baseline is None:
        empty_state("🧠", "Train the original model first", "Train the original SISA model in Model Training to continue.")
    elif not st.session_state.forget_ids:
        empty_state("🗑️", "No Forget Set yet", "First confirm a Forget Set in Erasure Requests. The attack will use those same patients before and after unlearning.")
    else:
        tr, va, te = st.session_state.train, st.session_state.val, st.session_state.test
        forget_ids = set(int(x) for x in st.session_state.forget_ids)
        forget_rows_all = tr[tr.subject_id.isin(forget_ids)].copy()
        retain_rows_all = tr[~tr.subject_id.isin(forget_ids)].copy()
        with st.expander("How this attack works"):
            st.markdown(
                "- **Shadow models (Shokri et al., 2017):** the attacker trains its own models on auxiliary patients it owns "
                "(the validation split, never overlapping with training or test) and learns what outputs look like for members versus non-members.\n"
                "- **Black-box:** it never sees the target's training set; it infers membership from model responses only.\n"
                "- **SISA-aware routing:** a known member is queried against the exact shard that trained it, and the same shard after unlearning. "
                "Non-member controls are spread evenly across shards.\n"
                "- **Fixed between stages:** attacker, candidates, routes and decision band do not change after unlearning.")
        if st.session_state.baseline.get("model_type", "sgd") != "forest":
            st.warning("The current shard learner (linear or neural network) generalises instead of memorising individual patients, "
                       "so the attacker cannot tell most forgotten patients from unseen ones even BEFORE unlearning, and their scores will not "
                       "change afterwards. Retrain with the random-forest shard learner for a clear before/after result.")

        max_candidates = min(forget_rows_all.subject_id.nunique(), te.subject_id.nunique())
        if max_candidates < 5:
            st.error("Choose at least five forgotten patients and ensure at least five held-out test patients are available for a meaningful demonstration.")
        else:
            sample_n = st.slider("Forgotten patient candidates to test", 5, min(1000, max_candidates), min(100, max_candidates), step=5)
            st.caption("One admission per patient is sampled so that repeated admissions from one patient do not dominate the attack.")
            confidence = st.slider("Attacker confidence required to claim membership", 0.60, 0.95, 0.75, step=0.05,
                                   help="The attacker only says 'Likely member' if its membership probability exceeds this, "
                                        "and 'Likely non-member' if it is below 1 minus this. Scores in between are Inconclusive.")

            if st.button("1 · Run attack BEFORE unlearning", type="primary"):
                try:
                    baseline = st.session_state.baseline
                    with st.spinner("Training shadow models and the attack classifier..."):
                        # One admission per patient; prefer the admission with the death outcome
                        # so the rare class is represented.
                        def one_per_patient(frame):
                            return (frame.sort_values(["subject_id", LABEL], ascending=[True, False])
                                    .groupby("subject_id", as_index=False).head(1))
                        cand_all = one_per_patient(forget_rows_all)
                        test_one = one_per_patient(te)
                        retain_one = one_per_patient(retain_rows_all)

                        # Matched design: non-member controls have EXACTLY the same outcome mix
                        # as the forgotten candidates (otherwise the comparison is confounded).
                        parts_c, parts_n = [], []
                        for cls in (0, 1):
                            c_pool = cand_all[cand_all[LABEL] == cls]
                            n_pool = test_one[test_one[LABEL] == cls]
                            want = int(round(sample_n * (len(c_pool) / max(1, len(cand_all)))))
                            k = min(want, len(c_pool), len(n_pool))
                            if k > 0:
                                parts_c.append(c_pool.sample(n=k, random_state=31 + cls))
                                parts_n.append(n_pool.sample(n=k, random_state=73 + cls))
                        if not parts_c:
                            raise ValueError("No matched candidates and controls could be built.")
                        forgotten_candidates = pd.concat(parts_c)
                        nonmember_controls = pd.concat(parts_n)
                        n_eval = len(forgotten_candidates)
                        # Retained controls: same outcome mix, patients that stay in training.
                        parts_r = []
                        for cls, grp in forgotten_candidates.groupby(LABEL):
                            pool = retain_one[retain_one[LABEL] == cls]
                            if len(pool):
                                parts_r.append(pool.sample(n=min(len(grp), len(pool)), random_state=47 + int(cls)))
                        retained_controls = pd.concat(parts_r) if parts_r else retain_one.head(0)
                        if n_eval < 5:
                            raise ValueError("Not enough independent patients for the attack. Increase the Forget Set or use a larger test split.")

                        # Attacker is built from the attacker's own auxiliary data only.
                        attacker = build_shokri_attacker(baseline, va, confidence=confidence)

                        # Actual shard for every training member; balanced fixed routes
                        # for non-members. These exact routes are reused after unlearning.
                        forget_routes = mia_shard_routes(baseline, forgotten_candidates)
                        retained_routes = mia_shard_routes(baseline, retained_controls) if len(retained_controls) else np.array([], dtype=int)
                        nonmember_routes = mia_shard_routes(baseline, nonmember_controls, nonmember=True)

                        attack_data = {
                            "forgotten_candidates": forgotten_candidates,
                            "nonmember_controls": nonmember_controls,
                            "retained_controls": retained_controls,
                            "thresholds": attacker,
                            "forget_routes": forget_routes,
                            "nonmember_routes": nonmember_routes,
                            "retained_routes": retained_routes
                        }
                        before = evaluate_mia_attack(
                            baseline, forgotten_candidates, nonmember_controls,
                            retained_controls, attacker,
                            forget_routes, nonmember_routes, retained_routes
                        )
                    st.session_state.mia_attack_data = attack_data
                    st.session_state.mia_before = before
                    st.session_state.mia_after = None
                    audit("mia_before_unlearning", {
                        "forget_patients_tested": len(forgotten_candidates),
                        "attack_auc": before["auc"],
                        "detected_rate": before["forget_detected_rate"],
                        "calibration_auc": attacker["calibration_auc"]
                    })
                    st.success("Attack completed against the original model. The same attacker, candidates and decision band are saved for the post-unlearning attack.")
                except Exception as exc:
                    st.error(f"Before-unlearning attack failed: {exc}")

            attack_data = st.session_state.mia_attack_data
            before = st.session_state.mia_before
            if before is not None and attack_data is not None:
                stage_header(1, "Before unlearning", "The fixed attacker is run against the original model.")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Forgotten candidates flagged as members", f'{before["forget_detected_rate"]:.1%}')
                m2.metric("Attack ROC-AUC", f'{before["auc"]:.3f}')
                m3.metric("Non-member false-positive rate", f'{before["nonmember_false_positive_rate"]:.1%}')
                m4.metric("Candidates tested", f'{before["n_forget"]:,}')
                with st.expander("Attacker details"):
                    st.markdown(
                        f'- **Decision band:** score < {before["threshold_low"]:.2f} → likely non-member · > {before["threshold_high"]:.2f} → likely member · otherwise inconclusive\n'
                        f'- **Shadow calibration AUC:** {attack_data["thresholds"]["calibration_auc"]:.3f} '
                        f'({attack_data["thresholds"]["n_shadow"]} shadow models, {attack_data["thresholds"]["shadow_train_patients"]:,} patients each)\n'
                        '- The band is not changed after unlearning.')
                if attack_data["thresholds"]["calibration_auc"] <= 0.55:
                    st.warning("The independently calibrated attacker is weak (AUC is close to chance). Do not interpret patient labels as reliable evidence of membership; this run should be treated as inconclusive.")
                else:
                    st.success("The attacker shows above-chance separation on a held-out shadow model. Its classifier, thresholds and shard-routing rules are now locked for the before/after test.")

                candidates = attack_data["forgotten_candidates"].copy()
                # Generate stable anonymized labels without exposing raw MIMIC identifiers.
                all_patient_ids = sorted(tr.subject_id.unique().tolist())
                label_map = {pid: f"Patient {i+1:05d}" for i, pid in enumerate(all_patient_ids)}
                candidate_view = pd.DataFrame({
                    "Candidate patient": candidates.subject_id.map(label_map),
                    "Age": candidates["age"].round(0).astype("Int64") if "age" in candidates else "N/A",
                    "Gender": candidates["gender"].astype(str) if "gender" in candidates else "N/A",
                    "Admission type": candidates["admission_type"].astype(str) if "admission_type" in candidates else "N/A",
                    "Attacker inference": before["forget_inference"],
                    "Membership score (higher = stronger evidence)": np.round(before["forget_scores"], 4)
                })
                st.markdown("**Candidate records visible to the simulated attacker**")
                st.dataframe(style_inference(candidate_view, ["Attacker inference"]), use_container_width=True, hide_index=True)
                st.caption("These candidate records are supplied to the attacker; they are not extracted from model weights. Membership scores are attack signals, not probabilities. The three-state result depends on the model scores and the fixed calibration band.")

                if st.session_state.unlearned is None:
                    st.warning("Now run SISA unlearning in Unlearning & Audit. Then return here and run the same attack against the unlearned model.")
                else:
                    if st.button("2 · Run same attack AFTER unlearning", type="primary"):
                        try:
                            after = evaluate_mia_attack(
                                st.session_state.unlearned,
                                attack_data["forgotten_candidates"],
                                attack_data["nonmember_controls"],
                                attack_data["retained_controls"],
                                attack_data["thresholds"],
                                attack_data["forget_routes"],
                                attack_data["nonmember_routes"],
                                attack_data["retained_routes"]
                            )
                            st.session_state.mia_after = after
                            audit("mia_after_unlearning", {
                                "forget_patients_tested": after["n_forget"],
                                "attack_auc": after["auc"],
                                "detected_rate": after["forget_detected_rate"]
                            })
                            st.success("Post-unlearning attack completed using the same candidate records, controls and fixed attacker.")
                        except Exception as exc:
                            st.error(f"After-unlearning attack failed: {exc}")

                after = st.session_state.mia_after
                if after is not None:
                    stage_header(2, "After unlearning", "The same attacker, candidates, shard routes and decision band are run against the unlearned model.")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Forgotten candidates flagged as members", f'{after["forget_detected_rate"]:.1%}', delta=f'{after["forget_detected_rate"]-before["forget_detected_rate"]:+.1%}', delta_color="inverse")
                    a2.metric("Attack ROC-AUC", f'{after["auc"]:.3f}', delta=f'{after["auc"]-before["auc"]:+.3f}', delta_color="inverse")
                    was = before["forget_inference"] == "Likely member"
                    still = after["forget_inference"] == "Likely member"
                    if was.sum() > 0:
                        st.metric("Records detected BEFORE that are no longer detected AFTER",
                                  f"{(was & ~still).sum()} of {int(was.sum())} ({(was & ~still).sum() / was.sum():.0%})")
                    else:
                        st.info("The attacker did not confidently detect any forgotten record before unlearning, so there is nothing to un-detect. Lower the confidence slider or use the random-forest shard learner.")
                    a3.metric("Retained-control detection rate", "N/A" if after["retained_detected_rate"] is None else f'{after["retained_detected_rate"]:.1%}')

                    comparison = pd.DataFrame({
                        "Attack stage": ["Before unlearning", "After unlearning"],
                        "Forgotten records flagged as members": [before["forget_detected_rate"], after["forget_detected_rate"]],
                        "Attack ROC-AUC": [before["auc"], after["auc"]],
                        "Non-member false-positive rate": [before["nonmember_false_positive_rate"], after["nonmember_false_positive_rate"]]
                    })
                    plot(px.bar(
                        comparison.melt("Attack stage", var_name="Measure", value_name="Value"),
                        x="Measure", y="Value", color="Attack stage", barmode="group",
                        color_discrete_map={"Before unlearning": "#c8465a", "After unlearning": "#0f7f86"},
                        title="Same attacker: membership signal before vs after unlearning",
                        range_y=[0, 1]
                    ), use_container_width=True)
                    st.dataframe(comparison, use_container_width=True, hide_index=True)

                    # Score distributions: forgotten records should move toward the non-member controls.
                    dist = go.Figure()
                    dist.add_trace(go.Histogram(x=before["forget_scores"], name="Forgotten · before", opacity=.65, nbinsx=30, marker_color="#c8465a"))
                    dist.add_trace(go.Histogram(x=after["forget_scores"], name="Forgotten · after", opacity=.65, nbinsx=30, marker_color="#0f7f86"))
                    dist.add_trace(go.Histogram(x=after["nonmember_scores"], name="Non-member controls", opacity=.5, nbinsx=30, marker_color="#8a99b0"))
                    dist.update_layout(barmode="overlay", title="Membership score distributions",
                                       xaxis_title="Membership score", yaxis_title="Patients")
                    plot(dist, use_container_width=True)

                    after_candidates = attack_data["forgotten_candidates"].copy()
                    after_view = pd.DataFrame({
                        "Candidate patient": after_candidates.subject_id.map(label_map),
                        "Before inference": before["forget_inference"],
                        "After inference": after["forget_inference"],
                        "Before membership score": np.round(before["forget_scores"], 4),
                        "After membership score": np.round(after["forget_scores"], 4)
                    })
                    st.markdown("**What changed for the same forgotten candidates?**")
                    st.dataframe(style_inference(after_view, ["Before inference", "After inference"]), use_container_width=True, hide_index=True)
                    if attack_data["thresholds"]["calibration_auc"] <= 0.55:
                        st.warning("Because the attacker did not validate reliably before unlearning, this before/after result cannot support a strong forgetting claim.")
                    else:
                        st.caption("Good result: forgotten scores move toward the non-member controls while retained controls stay member-like. Detection cannot drop below the false-positive rate. This is evidence against this attack, not proof of deletion.")
                    transition_counts = after_view.groupby(["Before inference", "After inference"]).size().reset_index(name="Patients")
                    st.markdown("**Membership-inference transitions**")
                    st.dataframe(style_inference(transition_counts, ["Before inference", "After inference"]), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download before/after MIA results",
                        comparison.to_csv(index=False).encode("utf-8"),
                        "mia_before_after_comparison.csv", "text/csv"
                    )
            else:
                st.info("Run Stage 1 to establish the attacker, candidate records and fixed decision threshold.")
