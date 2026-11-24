"""Global stylesheet."""
from __future__ import annotations

import streamlit as st


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--ink:#0f1f35;--muted:#66768d;--line:#e3e9f2;--brand:#0f7f86;--brand-dark:#0b666c;--rose:#c8465a;--green:#2a8f5c;--amber:#c98a1b;--bg:#f4f7fb;}
html, body, [class*="css"], .stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(1100px 480px at 88% -8%, #dff0f2 0%, rgba(223,240,242,0) 62%), var(--bg);}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.1rem;padding-bottom:3rem;max-width:1500px;}
h1,h2,h3,h4,h5 {font-family:'Manrope',sans-serif;color:var(--ink);letter-spacing:-.01em;}
h4 {font-size:1.08rem;margin-top:1.3rem;}
h5 {font-size:.98rem;margin:.2rem 0 .5rem 0;}

/* sidebar */
§ {background:linear-gradient(180deg,#0d1b30 0%,#112544 100%);border-right:1px solid #1c3558;}
§ * {color:#e6eefb;}
§ hr {border-color:rgba(255,255,255,.09);}
§ [data-testid="stMarkdownContainer"] p {color:#b9c9df;}
§ div[role="radiogroup"] {gap:2px;}
§ div[role="radiogroup"] > label {padding:9px 12px;border-radius:10px;width:100%;margin:0;transition:background .15s;}
§ div[role="radiogroup"] > label:hover {background:rgba(255,255,255,.07);}
§ div[role="radiogroup"] > label > div:first-child {display:none;}
§ div[role="radiogroup"] > label:has(input:checked) {background:rgba(37,170,178,.22);box-shadow:inset 3px 0 0 #3cc6cd;}
§ div[role="radiogroup"] > label:has(input:checked) p {font-weight:700;color:#fff;}
§ [data-testid="stWidgetLabel"] p {font-size:.72rem;letter-spacing:.12em;font-weight:800;text-transform:uppercase;color:#8fb3d9 !important;}
§ .brand {display:flex;gap:12px;align-items:center;padding:2px 2px 4px;}
§ .brand-mark {width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,#25aab2,#1f4e8c);display:flex;align-items:center;justify-content:center;font-size:1.35rem;flex:none;}
§ .brand-name {font-family:'Manrope',sans-serif;font-weight:800;color:#fff;line-height:1.15;font-size:1.02rem;}
§ .brand-sub {color:#93a9c6;font-size:.72rem;margin-top:2px;}
§ .wf-head {display:flex;justify-content:space-between;font-size:.7rem;letter-spacing:.14em;font-weight:800;color:#8fb3d9;margin-bottom:7px;}
§ .wf-bar {height:6px;background:rgba(255,255,255,.1);border-radius:99px;overflow:hidden;margin-bottom:10px;}
§ .wf-bar > div {height:100%;background:linear-gradient(90deg,#25aab2,#7fe3e8);border-radius:99px;}
§ .wf-step {display:flex;gap:10px;align-items:center;font-size:.84rem;padding:4px 0;color:#9fb3cf;}
§ .wf-step .dot {width:18px;height:18px;border-radius:50%;border:1.5px solid #46628a;display:inline-flex;align-items:center;justify-content:center;font-size:.65rem;flex:none;color:#9fb3cf;}
§ .wf-step.done, § .wf-step.done .dot {color:#e6eefb;}
§ .wf-step.done .dot {background:#25aab2;border-color:#25aab2;color:#06242a;font-weight:800;}
§ .wf-step.next, § .wf-step.next .dot {color:#fff;font-weight:700;}
§ .wf-step.next .dot {border-color:#7fe3e8;box-shadow:0 0 0 3px rgba(127,227,232,.2);}
§ .wf-hint {margin-top:10px;background:rgba(127,227,232,.1);border:1px solid rgba(127,227,232,.25);border-radius:10px;padding:8px 10px;font-size:.78rem;color:#cfeff2;line-height:1.4;}

/* page header */
.hero {background:linear-gradient(120deg,#0d1b30 0%,#143a5a 58%,#0f7f86 135%);border-radius:18px;padding:22px 26px 20px;margin:0 0 18px 0;box-shadow:0 8px 24px rgba(13,27,48,.14);}
.hero .kick {font-size:.7rem;letter-spacing:.16em;font-weight:800;color:#7fe3e8;}
.hero h1 {color:#fff;font-size:1.85rem;margin:.15rem 0 .25rem 0;padding:0;font-weight:800;}
.hero p {color:#cfe0f2;margin:0;font-size:.95rem;max-width:900px;line-height:1.45;}
.hero .badges {margin-top:13px;display:flex;gap:8px;flex-wrap:wrap;}
.hero .badge {background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);color:#eaf4ff;padding:3px 11px;border-radius:999px;font-size:.76rem;font-weight:600;}

/* cards, metrics, controls */
div[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-left:4px solid var(--brand);padding:14px 16px;border-radius:14px;box-shadow:0 1px 2px rgba(15,31,53,.04);}
div[data-testid="stMetricLabel"] p {color:var(--muted);font-size:.76rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;}
div[data-testid="stMetricValue"] {color:var(--ink);font-family:'Manrope',sans-serif;font-weight:800;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
div.stButton > button, div.stDownloadButton > button {border-radius:10px;border:1px solid #d3deec;font-weight:650;min-height:2.6rem;transition:transform .12s, box-shadow .12s;}
div.stButton > button:hover, div.stDownloadButton > button:hover {transform:translateY(-1px);box-shadow:0 4px 12px rgba(15,31,53,.1);border-color:var(--brand);color:var(--brand-dark);}
div.stButton > button[kind="primary"] {background:linear-gradient(135deg,#12979e,#0b666c);color:#fff;border:0;}
div.stButton > button[kind="primary"]:hover {color:#fff;box-shadow:0 6px 16px rgba(15,127,134,.32);}
div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
div[data-testid="stAlert"] {border-radius:12px;}
div[data-testid="stExpander"] {background:#fff;border:1px solid var(--line);border-radius:12px;}
div[data-baseweb="tab-list"] {gap:6px;border-bottom:1px solid var(--line);}
button[data-baseweb="tab"] {font-weight:600;padding:8px 16px;}
button[data-baseweb="tab"][aria-selected="true"] {color:var(--brand-dark);}
div[data-baseweb="tab-highlight"] {background-color:var(--brand);}
.stSlider [data-baseweb="slider"] div[role="slider"] {background-color:var(--brand);}

/* reusable blocks */
.sec {margin:1.5rem 0 .6rem 0;}
.sec .t {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.08rem;color:var(--ink);}
.sec .s {color:var(--muted);font-size:.86rem;margin-top:1px;}
.stage {display:flex;align-items:center;gap:12px;margin:1.6rem 0 .7rem 0;}
.stage .n {width:30px;height:30px;border-radius:50%;background:var(--brand);color:#fff;font-family:'Manrope',sans-serif;font-weight:800;display:flex;align-items:center;justify-content:center;flex:none;}
.stage .t {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.12rem;color:var(--ink);line-height:1.15;}
.stage .s {color:var(--muted);font-size:.84rem;}
.empty {background:#fff;border:1.5px dashed #c6d4e6;border-radius:16px;padding:34px 24px;text-align:center;margin:10px 0 18px;}
.empty .ico {font-size:2rem;}
.empty .t {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.15rem;color:var(--ink);margin:.3rem 0 .2rem;}
.empty .b {color:var(--muted);max-width:560px;margin:0 auto;line-height:1.5;}
.chip {display:inline-block;padding:2px 10px;border-radius:999px;font-size:.76rem;font-weight:700;white-space:nowrap;}
.chip.pass {background:#e2f5ea;color:#1c7a47;} .chip.fail {background:#fde8eb;color:#b3263d;}
.chip.warn {background:#fff3da;color:#9a6a08;} .chip.skip {background:#edf1f7;color:#5d6d84;} .chip.info {background:#e4eefb;color:#25589c;}
table.vt {width:100%;border-collapse:separate;border-spacing:0;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden;font-size:.9rem;}
table.vt th {background:#f3f6fb;text-align:left;padding:10px 14px;color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;}
table.vt td {padding:11px 14px;border-top:1px solid var(--line);vertical-align:top;color:var(--ink);line-height:1.45;}
.vbanner {display:flex;gap:16px;align-items:center;border-radius:16px;padding:18px 22px;margin:6px 0 14px;border:1px solid;}
.vbanner.ok {background:#e9f7ef;border-color:#bfe3cd;} .vbanner.bad {background:#fdecef;border-color:#f3c3cb;} .vbanner.neutral {background:#fff6e2;border-color:#f0dba6;}
.vbanner .big {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.45rem;color:var(--ink);white-space:nowrap;}
.vbanner .txt {color:#33475f;line-height:1.45;}
.shards {display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:8px 0 14px;}
.shard {background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 14px;box-shadow:0 1px 2px rgba(15,31,53,.04);}
.shard .n {font-family:'Manrope',sans-serif;font-weight:800;color:var(--brand);}
.shard .m {color:var(--muted);font-size:.82rem;margin-top:5px;line-height:1.55;}
.foot {text-align:center;color:#8a99b0;font-size:.78rem;margin-top:2.2rem;padding-top:1rem;border-top:1px solid var(--line);}
</style>
""".replace("§", 'section[data-testid="stSidebar"]')


def inject_css():
    """Inject the global stylesheet."""
    st.markdown(_CSS, unsafe_allow_html=True)
