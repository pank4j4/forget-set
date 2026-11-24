"""Sidebar: brand, page navigation, prediction task and the workflow tracker."""
from __future__ import annotations

import html as htmllib

import streamlit as st

from clinical_unlearning.config import TASKS
from ui.state import reset_for_task_change

PAGE_ICONS = {
    "Patient Records": "👥", "Overview Dashboard": "📊", "Model Training": "🧠", "SISA Architecture": "🧩",
    "Erasure Requests": "🗑️", "MIA Simulation": "🕵️", "Unlearning & Audit": "✅", "Experiment History": "🗂️",
}


def render_sidebar(pages):
    """Draw the sidebar. Returns (selected page, container for the workflow tracker)."""
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">🧬</div><div><div class="brand-name">Clinical<br>Unlearning Lab</div>'
                    '<div class="brand-sub">MIMIC-IV · Privacy-aware clinical AI</div></div></div>', unsafe_allow_html=True)
        st.divider()
        page = st.radio("WORKSPACE", pages, label_visibility="collapsed",
                        format_func=lambda p: f"{PAGE_ICONS.get(p, '')}  {p}")
        st.divider()
        task_name = st.radio("Prediction task", list(TASKS), key="task_name", format_func=lambda n: n.split(" (")[0],
                             help="Prolonged stay is balanced (~50/50), which makes the membership attack and the unlearning effect measurable. Mortality is very imbalanced.")
        reset_for_task_change(task_name)
        st.divider()
        workflow_box = st.container()   # filled by render_workflow() once the page has run
        st.divider()
        st.caption("Research prototype · Not for clinical decisions")
    return page, workflow_box


def render_workflow(box) -> None:
    """Progress tracker with a hint for the next step."""
    with box:
        steps = [
            ("Patient-level split", st.session_state.train is not None, "Create the split in Model Training."),
            ("Original SISA model trained", st.session_state.baseline is not None, "Train the SISA model in Model Training."),
            ("Forget Set confirmed", bool(st.session_state.forget_ids), "Choose patients to forget in Erasure Requests."),
            ("Attack run before unlearning", st.session_state.mia_before is not None, "Run Stage 1 in MIA Simulation."),
            ("SISA unlearning + audit run", st.session_state.unlearned is not None, "Run unlearning in Unlearning & Audit."),
            ("Attack run after unlearning", st.session_state.mia_after is not None, "Run Stage 2 in MIA Simulation."),
        ]
        done = sum(1 for _, d, _ in steps if d)
        nxt = next((i for i, (_, d, _) in enumerate(steps) if not d), None)
        rows = "".join(
            f'<div class="wf-step {"done" if d else ("next" if i == nxt else "")}"><span class="dot">{"✓" if d else i + 1}</span>{htmllib.escape(label)}</div>'
            for i, (label, d, _) in enumerate(steps))
        hint = (f'<div class="wf-hint"><b>Next:</b> {htmllib.escape(steps[nxt][2])}</div>' if nxt is not None
                else '<div class="wf-hint"><b>All steps complete.</b> Download the audit report from Unlearning & Audit.</div>')
        st.markdown(f'<div class="wf-head"><span>WORKFLOW</span><span>{done}/{len(steps)}</span></div>'
                    f'<div class="wf-bar"><div style="width:{int(100 * done / len(steps))}%"></div></div>{rows}{hint}',
                    unsafe_allow_html=True)
