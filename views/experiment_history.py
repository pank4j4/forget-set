"""Page: Experiment History"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from clinical_unlearning.storage import get_db
from ui.components import page_header


def render(df):
    page_header("RECORDS", "Experiment History", "Every training run, unlearning operation and attack is logged here for reproducibility.")
    con = get_db()
    experiments = pd.read_sql_query("SELECT * FROM experiments ORDER BY id DESC", con)
    audits = pd.read_sql_query("SELECT * FROM audit ORDER BY id DESC", con)
    con.close()
    st.markdown("#### Experiments")
    if experiments.empty:
        st.info("No experiments recorded yet.")
    else:
        st.dataframe(experiments.drop(columns=["metrics_json"]), use_container_width=True, hide_index=True)
        st.download_button("Export experiment history", experiments.to_csv(index=False).encode(), "experiment_history.csv", "text/csv")
    st.markdown("#### Audit trail")
    if audits.empty:
        st.info("No audit events recorded yet.")
    else:
        st.dataframe(audits, use_container_width=True, hide_index=True)
        st.download_button("Export audit log", audits.to_csv(index=False).encode(), "audit_log.csv", "text/csv")
    st.markdown("#### Research notes")
    st.warning("MIA values describe the implemented shadow-model attack only. The attacker's classifier and three-state decision band are fixed before unlearning. Unlearning does not remove rows from source CSV files, backups, logs, or external systems. This prototype is for research demonstration, not clinical use.")
