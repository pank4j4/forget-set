"""Clinical Unlearning Lab: entry point.  Run with:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Clinical Unlearning Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from clinical_unlearning.config import DEFAULT_LABEL, TARGET, TASKS, get_label, set_label
from ui.sidebar import render_sidebar, render_workflow
from ui.state import ensure_data, init_state
from ui.theme import inject_css
from views import VIEWS

FOOTER = ('<div class="foot">Clinical Unlearning Lab · MIMIC-IV research prototype · '
          'No clinical decisions should be made from this application.</div>')

inject_css()
init_state()

# The prediction task chosen in the sidebar decides which outcome column every model is trained on.
set_label(TASKS.get(st.session_state.get("task_name"), DEFAULT_LABEL))

page, workflow_box = render_sidebar(list(VIEWS))
df = ensure_data()

if get_label() not in df.columns:
    st.warning("dischtime is missing from admissions.csv, so the prolonged-stay task is unavailable. Using mortality.")
    set_label(TARGET)

VIEWS[page](df)

st.markdown(FOOTER, unsafe_allow_html=True)
render_workflow(workflow_box)
