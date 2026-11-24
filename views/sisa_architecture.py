"""Page: Sisa Architecture"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components import empty_state, page_header, section_title


def render(df):
    page_header("MODEL INTERNALS", "SISA Architecture", "Sharded, Isolated, Sliced and Aggregated training.")
    if st.session_state.baseline is None:
        empty_state("🧠", "No trained model yet", "Train the original SISA model to inspect its shard and slice assignments.")
    else:
        bundle = st.session_state.baseline
        cols = st.columns(4)
        cols[0].metric("Shards", bundle["n_shards"])
        cols[1].metric("Slices / shard", bundle["n_slices"])
        cols[2].metric("Patients", f"{st.session_state.train.subject_id.nunique():,}")
        cols[3].metric("Checkpoints", sum(len(s["checkpoints"]) for s in bundle["shards"]))
        rows = []
        for shard in bundle["shards"]:
            rows.append({
                "Shard": shard["shard_id"] + 1,
                "Patients assigned": len(shard["patient_ids"]),
                "Admissions": len(shard["row_indices"]),
                "Slices created": len(shard["slice_indices"]),
                "Checkpoints": len(shard["checkpoints"])
            })
        shard_cards = "".join(
            f'<div class="shard"><div class="n">Shard {r["Shard"]}</div><div class="m">{r["Patients assigned"]:,} patients<br>'
            f'{r["Admissions"]:,} admissions<br>{r["Slices created"]} slice(s) · {r["Checkpoints"]} checkpoint(s)</div></div>'
            for r in rows)
        st.markdown(f'<div class="shards">{shard_cards}</div>', unsafe_allow_html=True)
        with st.expander("Table view"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        section_title("Inspect a shard", "Slice boundaries are cut by patient, so a patient never spans two slices.")
        selected_shard = st.selectbox("Inspect shard", list(range(bundle["n_shards"])), format_func=lambda x:f"Shard {x+1}")
        shard = bundle["shards"][selected_shard]
        slice_rows = [{"Slice":i+1,"Admissions":len(ix),"First patient (internal)":int(st.session_state.train.iloc[ix].subject_id.min()),
                       "Last patient (internal)":int(st.session_state.train.iloc[ix].subject_id.max())}
                      for i,ix in enumerate(shard["slice_indices"])]
        st.dataframe(pd.DataFrame(slice_rows), use_container_width=True, hide_index=True)
        st.caption("Raw IDs are shown only in this technical architecture view; patient-facing pages use anonymized labels.")
