"""Reusable presentation helpers: page header, section titles, empty states, charts, styled tables."""
from __future__ import annotations

import html as htmllib

import streamlit as st

from clinical_unlearning.config import LEARNER_NAMES, TASKS, get_label


# ---------------------------------------------------------------------
# UI helpers (presentation only; no effect on data or models)
# ---------------------------------------------------------------------
PALETTE = ["#0f7f86", "#1f4e8c", "#c8465a", "#c98a1b", "#6b5bd6", "#2a8f5c"]


def task_short():
    LABEL = get_label()
    name = next((k for k, v in TASKS.items() if v == LABEL), str(LABEL))
    return name.split(" (")[0]


def page_header(kicker, title, subtitle=""):
    badges = [f"Task: {task_short()}"]
    base = st.session_state.get("baseline")
    if base:
        badges.append(f"{LEARNER_NAMES.get(base.get('model_type', 'sgd'), 'Model')} · {base['n_shards']} shards")
    fids = st.session_state.get("forget_ids")
    if fids:
        badges.append(f"Forget Set: {len(fids):,} patients")
    chips = "".join(f'<span class="badge">{htmllib.escape(b)}</span>' for b in badges)
    sub = f"<p>{htmllib.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="hero"><div class="kick">{htmllib.escape(kicker)}</div><h1>{htmllib.escape(title)}</h1>{sub}'
                f'<div class="badges">{chips}</div></div>', unsafe_allow_html=True)


def section_title(title, subtitle=""):
    sub = f'<div class="s">{htmllib.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="sec"><div class="t">{htmllib.escape(title)}</div>{sub}</div>', unsafe_allow_html=True)


def stage_header(number, title, subtitle=""):
    sub = f'<div class="s">{htmllib.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="stage"><div class="n">{number}</div><div><div class="t">{htmllib.escape(title)}</div>{sub}</div></div>',
                unsafe_allow_html=True)


def empty_state(icon, title, body):
    st.markdown(f'<div class="empty"><div class="ico">{icon}</div><div class="t">{htmllib.escape(title)}</div>'
                f'<div class="b">{htmllib.escape(body)}</div></div>', unsafe_allow_html=True)


def card():
    try:
        return st.container(border=True)
    except TypeError:  # older Streamlit without bordered containers
        return st.container()


def plot(fig, use_container_width=True, **kwargs):
    """Apply one consistent chart theme, then render."""
    fig.update_layout(
        font=dict(family="DM Sans, sans-serif", color="#0f1f35", size=13),
        title_font=dict(family="Manrope, sans-serif", size=16), title_x=0, title_xanchor="left",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=PALETTE,
        margin=dict(l=10, r=10, t=56, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d5deea", zeroline=False)
    fig.update_yaxes(gridcolor="#e7edf5", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)


_INFERENCE_STYLE = {
    "Likely member": "background-color:#fde8eb;color:#b3263d;font-weight:600",
    "Likely non-member": "background-color:#e2f5ea;color:#1c7a47;font-weight:600",
    "Inconclusive": "background-color:#edf1f7;color:#5d6d84;font-weight:600",
}


def style_inference(frame, cols):
    """Colour the three attacker verdicts. Falls back to the plain table if styling is unavailable."""
    try:
        styler = frame.style
        apply = getattr(styler, "map", None) or styler.applymap
        return apply(lambda v: _INFERENCE_STYLE.get(v, ""), subset=cols)
    except Exception:
        return frame
