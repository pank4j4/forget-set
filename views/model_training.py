"""Page: Model Training"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from clinical_unlearning.config import get_label
from clinical_unlearning.data import get_feature_columns, split_by_patient
from clinical_unlearning.metrics import metrics_for
from clinical_unlearning.sisa import sisa_predict, train_sisa
from clinical_unlearning.storage import audit, save_experiment
from ui.components import card, empty_state, page_header, plot, section_title, task_short


def render(df):
    LABEL = get_label()
    page_header("MODEL WORKSPACE", "Model Training", f"Train a patient-level SISA ensemble. Prediction task: {task_short()}.")
    if st.session_state.train is None:
        empty_state("🧩", "No data split yet", "Create the patient-level train / validation / test split first. A patient never appears in more than one partition.")
        if st.button("Create patient-level split", type="primary"):
            tr, va, te = split_by_patient(df)
            st.session_state.train, st.session_state.val, st.session_state.test = tr, va, te
            st.session_state.baseline = st.session_state.unlearned = None
            st.session_state.forget_ids = []
            st.session_state.mia_before = st.session_state.mia_after = st.session_state.mia_attack_data = None
            audit("patient_split_created", {"train":len(tr),"validation":len(va),"test":len(te)})
            st.rerun()
    else:
        tr, va, te = st.session_state.train, st.session_state.val, st.session_state.test
        a,b,c = st.columns(3)
        a.metric("Training patients", f"{tr.subject_id.nunique():,}")
        b.metric("Validation patients", f"{va.subject_id.nunique():,}")
        c.metric("Test patients", f"{te.subject_id.nunique():,}")
        st.caption(f"Positive outcome rate ({task_short()}) · train {tr[LABEL].mean():.1%} · validation {va[LABEL].mean():.1%} · test {te[LABEL].mean():.1%}")

        with card():
            section_title("1 · Choose the shard learner", "Each shard trains its own independent model; predictions are averaged.")
            model_type_label = st.selectbox(
                "Shard learner",
                ["Random forest (memorising, recommended for MIA demo)",
                 "Neural network (parametric, high capacity)",
                 "Logistic regression (SGD, low capacity)"],
                label_visibility="collapsed"
            )
            model_type = ("forest" if model_type_label.startswith("Random")
                          else "mlp" if model_type_label.startswith("Neural") else "sgd")
            section_title("2 · Configure the ensemble")
            cs1, cs2, cs3 = st.columns(3)
            n_shards = cs1.slider("SISA shards", 2, 10, 5)
            if model_type == "forest":
                n_trees = cs2.slider("Trees per shard", 20, 300, 100, step=10)
                n_slices, epochs = 1, 1
                learner_note = "Trees are grown to purity, so there are no epochs. Each shard is one slice and is retrained when it holds a forgotten patient."
            elif model_type == "mlp":
                n_trees = 100
                n_slices = cs2.slider("Slices per shard", 2, 8, 4)
                epochs = cs3.slider("Epochs per slice", 10, 200, 100, step=10)
                learner_note = (f"Two wide layers, almost no regularisation. Each slice is trained for {epochs} epochs and checkpointed; "
                                f"a full shard sees ≈{n_slices * epochs:,} epochs. Unlearning resumes from the checkpoint before the first affected slice. "
                                "Training time grows with epochs, so start lower on the full dataset.")
            else:
                n_trees = 100
                n_slices = cs2.slider("Slices per shard", 2, 8, 4)
                epochs = cs3.slider("Epochs per slice", 10, 200, 100, step=10)
                learner_note = f"Linear model: {epochs} epochs per slice. It generalises rather than memorises, so the attack finds little to detect."
            st.caption(learner_note)
        feature_cols = get_feature_columns(tr)
        st.caption(f"{len(feature_cols)} model features · identifiers and outcome fields are excluded.")
        if st.button("Train original SISA model", type="primary"):
            try:
                start_time = time.time()
                with st.spinner("Training shards..."):
                    bundle = train_sisa(tr, feature_cols, n_shards, n_slices, epochs, model_type, n_trees)
                    bundle["original_patient_shard_map"] = {
                        int(pid): int(shard["shard_id"])
                        for shard in bundle["shards"]
                        for pid in shard["patient_ids"]
                    }
                    probs = sisa_predict(bundle, te)
                    metrics = metrics_for(te[LABEL], probs)
                metrics["Training_seconds"] = time.time() - start_time
                st.session_state.baseline = bundle
                st.session_state.baseline_probs = probs
                st.session_state.baseline_metrics = metrics
                st.session_state.feature_cols = feature_cols
                st.session_state.unlearned = None
                st.session_state.last_result = None
                st.session_state.mia_before = st.session_state.mia_after = st.session_state.mia_attack_data = None
                audit("original_sisa_trained", {**metrics, "model_type": model_type, "n_trees": n_trees})
                save_experiment(f"Original SISA training ({model_type})", 0, len(tr), "Completed", metrics)
                st.success("Original SISA ensemble trained.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        if st.session_state.baseline is not None:
            metrics = st.session_state.baseline_metrics
            cols = st.columns(5)
            for col, (name, value) in zip(cols, list(metrics.items())[:5]):
                col.metric(name.replace("_"," "), f"{value:.3f}")
            y = te[LABEL].to_numpy()
            p = st.session_state.baseline_probs
            if len(np.unique(y)) > 1:
                fpr, tpr, _ = roc_curve(y, p)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name="SISA model"))
                fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Chance",line=dict(dash="dash")))
                fig.update_layout(title="Held-out test ROC curve",xaxis_title="False positive rate",yaxis_title="True positive rate")
                plot(fig, use_container_width=True)
