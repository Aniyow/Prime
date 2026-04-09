import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
import plotly.graph_objects as go
import plotly.express as px
from math import comb
from sklearn.linear_model import LinearRegression

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Prime Numbers",
    page_icon="∞",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("page",        "overview")
_init("theme",       "dark")
_init("csv_path",    "./Prime_code/primes_with_gaps.csv")
_init("prime_limit", 50_000)

DARK = dict(
    bg="#07080d", surface="#0e1017", surface2="#131620",
    border="#1e2130", text="#e8eaf0", muted="#6b7280",
    accent="#c8ff57", accent2="#57d4ff", accent3="#ff6b6b",
    yellow="#ffca28", paper_bg="rgba(0,0,0,0)", grid="#1e2130",
)
LIGHT = dict(
    bg="#ffffff", surface="#f8fafc", surface2="#ffffff",
    border="#d1d5db", text="#111827", muted="#4b5563",
    accent="#2563eb", accent2="#059669", accent3="#dc2626",
    yellow="#ca8a04", paper_bg="rgba(255,255,255,1)", grid="#d1d5db",
)

T  = DARK if st.session_state.theme == "dark" else LIGHT
A  = T["accent"]
A2 = T["accent2"]
A3 = T["accent3"]
AY = T["yellow"]

# ══════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {{
  --bg:   {T['bg']};    --surf:  {T['surface']};  --surf2: {T['surface2']};
  --bdr:  {T['border']}; --txt:  {T['text']};     --muted: {T['muted']};
  --acc:  {T['accent']}; --acc2: {T['accent2']};  --acc3:  {T['accent3']};
  --yel:  {T['yellow']}; --mono: 'Space Mono', monospace; --sans: 'Syne', sans-serif;
}}

.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
  background: var(--bg) !important;
  color: var(--txt) !important;
  font-family: var(--sans);
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

#MainMenu {{ visibility: hidden !important; }}
footer    {{ visibility: hidden !important; }}

.block-container {{ padding: 1.6rem 2.2rem !important; max-width: 1440px !important; }}

[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {{
  background: {T['surface']} !important;
  border-right: 1px solid {T['border']} !important;
}}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-testid]),
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div.stMarkdown,
[data-testid="stSidebar"] [data-testid="stText"] {{
  color: {T['text']} !important;
}}

[data-testid="stSidebar"] input {{
  background: {T['surface2']} !important;
  color: {T['text']} !important;
  border-color: {T['border']} !important;
  font-family: var(--mono);
  font-size: 12px;
}}
[data-testid="stSidebar"] label {{
  color: {T['muted']} !important;
  font-family: var(--mono);
  font-size: 11px;
}}

[data-testid="stSidebar"] .stButton > button {{
  background: transparent !important;
  border: 1px solid {T['border']} !important;
  color: {T['text']} !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  border-radius: 8px !important;
  transition: all .2s !important;
  text-align: left !important;
  padding: 8px 14px !important;
  width: 100% !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
  border-color: {T['accent']} !important;
  color: {T['accent']} !important;
  background: rgba(200,255,87,0.05) !important;
}}

[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {{
  visibility: visible !important;
  display: block !important;
  opacity: 1 !important;
}}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {{
  visibility: visible !important;
  opacity: 1 !important;
  background: {T['surface']} !important;
  border: 1px solid {T['border']} !important;
}}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg {{
  fill: {T['accent']} !important;
  stroke: {T['accent']} !important;
}}

[data-testid="stSlider"] [role="progressbar"] {{ background: var(--acc) !important; }}
[data-testid="stSlider"] [role="slider"]      {{ background: var(--acc) !important; border-color: var(--acc) !important; }}

.stButton > button {{
  background: transparent !important;
  border: 1px solid var(--bdr) !important;
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  border-radius: 8px !important;
  transition: all .2s !important;
  text-align: left !important;
  padding: 8px 14px !important;
}}
.stButton > button:hover {{ border-color: var(--acc) !important; color: var(--acc) !important; }}

.hero {{
  background: linear-gradient(135deg, var(--surf) 0%, var(--surf2) 60%, var(--bg) 100%);
  border: 1px solid var(--bdr); border-radius: 16px;
  padding: 36px 44px; margin-bottom: 22px; position: relative; overflow: hidden;
}}
.hero::before {{
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(200,255,87,.04) 0%, transparent 70%);
}}
.hero-tag   {{ font-family:var(--mono); font-size:10px; letter-spacing:3px; text-transform:uppercase; color:var(--acc); margin-bottom:8px; }}
.hero-title {{ font-family:var(--sans); font-weight:800; font-size:36px; line-height:1.05; color:var(--txt); margin:0 0 8px; }}
.hero-title span {{ color:var(--acc); }}
.hero-sub   {{ font-family:var(--mono); font-size:12px; color:var(--muted); }}

.stat-row {{ display:flex; gap:12px; margin-bottom:22px; flex-wrap:wrap; }}
.stat-card {{
  flex:1; min-width:130px; background:var(--surf);
  border:1px solid var(--bdr); border-radius:12px;
  padding:18px 20px; position:relative; overflow:hidden;
}}
.stat-card::after {{ content:''; position:absolute; bottom:0; left:0; right:0; height:2px; }}
.stat-card.g::after {{ background:var(--acc); }}
.stat-card.b::after {{ background:var(--acc2); }}
.stat-card.r::after {{ background:var(--acc3); }}
.stat-card.y::after {{ background:var(--yel); }}
.stat-lbl {{ font-family:var(--mono); font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:5px; }}
.stat-val {{ font-family:var(--mono); font-size:22px; font-weight:700; }}
.stat-val.g {{ color:var(--acc); }}  .stat-val.b {{ color:var(--acc2); }}
.stat-val.r {{ color:var(--acc3); }} .stat-val.y {{ color:var(--yel); }}
.stat-hint {{ font-family:var(--mono); font-size:10px; color:var(--muted); margin-top:3px; }}

.sec-head {{
  font-family:var(--mono); font-size:10px; letter-spacing:3px;
  text-transform:uppercase; color:var(--muted);
  margin:0 0 14px; padding-bottom:10px; border-bottom:1px solid var(--bdr);
}}
.sec-head span {{ color:var(--acc); margin-right:8px; }}

.info-box {{
  background:rgba(87,212,255,.05); border:1px solid rgba(87,212,255,.2);
  border-radius:10px; padding:13px 17px;
  font-family:var(--mono); font-size:12px; color:var(--acc2); line-height:1.7; margin-bottom:16px;
}}

.sb-logo {{
  font-family:var(--mono); font-weight:700; font-size:17px; color:var(--acc);
  padding:18px 18px 24px; border-bottom:1px solid var(--bdr); margin-bottom:18px;
}}
.sb-logo small {{ display:block; font-size:9px; letter-spacing:3px; color:var(--muted); margin-top:3px; font-weight:400; }}

/* FIX: chart title/desc styles were missing — added here */
.chart-title {{
  font-family: var(--sans); font-size: 14px; font-weight: 600;
  color: var(--txt); margin-bottom: 2px;
}}
.chart-desc {{
  font-family: var(--mono); font-size: 11px; color: var(--muted);
}}

::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:var(--bg); }}
::-webkit-scrollbar-thumb {{ background:var(--bdr); border-radius:3px; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PLOTLY LAYOUT HELPER
# ══════════════════════════════════════════════
def plot_layout(h=380, **kw):
    return dict(
        title="",
        paper_bgcolor=T["paper_bg"],
        plot_bgcolor=T["paper_bg"],
        font=dict(family="Space Mono, monospace", color=T["muted"], size=11),
        xaxis=dict(gridcolor=T["grid"], linecolor=T["grid"], zerolinecolor=T["grid"],
                   tickfont=dict(color=T["muted"])),
        yaxis=dict(gridcolor=T["grid"], linecolor=T["grid"], zerolinecolor=T["grid"],
                   tickfont=dict(color=T["muted"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=T["border"],
                    borderwidth=1, font=dict(color=T["text"])),
        margin=dict(l=10, r=10, t=20, b=10),
        height=h, **kw,
    )

def sfig(h=380):
    fig = go.Figure()
    fig.update_layout(**plot_layout(h))
    return fig

def chart(fig, title_txt, desc_txt="", h=None):
    st.markdown(
        f'<div style="background:{T["surface"]};border:1px solid {T["border"]};'
        f'border-radius:14px 14px 0 0;padding:16px 20px 8px;">'
        f'<div class="chart-title">{title_txt}</div>'
        f'<div class="chart-desc">{desc_txt}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<div style="background:{T["surface"]};border:1px solid {T["border"]};'
        f'border-top:none;border-radius:0 0 14px 14px;height:8px;margin-bottom:18px;"></div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(path: str):
    df     = pd.read_csv(path)
    primes = df["prime"].values.astype(np.int64)
    gaps   = df["gap"].values.astype(np.float64)
    gaps   = gaps[gaps > 0]
    return primes, gaps

# ══════════════════════════════════════════════
#  MERSENNE PRIME DATA  (all 52 known, as of 2024)
#  columns: rank, exponent p, digit count, year discovered
# ══════════════════════════════════════════════
MERSENNE_DATA = [
    (1,  2,       1,        "antiquity"),
    (2,  3,       1,        "antiquity"),
    (3,  5,       2,        "antiquity"),
    (4,  7,       3,        "antiquity"),
    (5,  13,      4,        "1461"),
    (6,  17,      6,        "1588"),
    (7,  19,      6,        "1588"),
    (8,  31,      10,       "1772"),
    (9,  61,      19,       "1883"),
    (10, 89,      27,       "1911"),
    (11, 107,     33,       "1914"),
    (12, 127,     39,       "1876"),
    (13, 521,     157,      "1952"),
    (14, 607,     183,      "1952"),
    (15, 1279,    386,      "1952"),
    (16, 2203,    664,      "1952"),
    (17, 2281,    687,      "1952"),
    (18, 3217,    969,      "1957"),
    (19, 4253,    1281,     "1961"),
    (20, 4423,    1332,     "1961"),
    (21, 9689,    2917,     "1963"),
    (22, 9941,    2993,     "1963"),
    (23, 11213,   3376,     "1963"),
    (24, 19937,   6002,     "1971"),
    (25, 21701,   6533,     "1978"),
    (26, 23209,   6987,     "1979"),
    (27, 44497,   13395,    "1979"),
    (28, 86243,   25962,    "1982"),
    (29, 110503,  33265,    "1988"),
    (30, 132049,  39751,    "1983"),
    (31, 216091,  65050,    "1985"),
    (32, 756839,  227832,   "1992"),
    (33, 859433,  258716,   "1994"),
    (34, 1257787, 378632,   "1996"),
    (35, 1398269, 420921,   "1996"),
    (36, 2976221, 895932,   "1997"),
    (37, 3021377, 909526,   "1998"),
    (38, 6972593, 2098960,  "1999"),
    (39, 13466917,4053946,  "2001"),
    (40, 20996011,6320430,  "2003"),
    (41, 24036583,7235733,  "2004"),
    (42, 25964951,7816230,  "2005"),
    (43, 30402457,9152052,  "2005"),
    (44, 32582657,9808358,  "2006"),
    (45, 37156667,11185272, "2008"),
    (46, 42643801,12837064, "2009"),
    (47, 43112609,12978189, "2008"),
    (48, 57885161,17425170, "2013"),
    (49, 74207281,22338618, "2016"),
    (50, 77232917,23249425, "2017"),
    (51, 82589933,24862048, "2018"),
    (52, 136279841,41024320,"2024"),
]

df_m = pd.DataFrame(MERSENNE_DATA, columns=["rank", "exponent", "digits", "year_str"])
df_m["year"]    = df_m["year_str"].apply(lambda y: int(y) if y != "antiquity" else 0)
df_m["exp_gap"] = df_m["exponent"].diff().fillna(0).astype(int)

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-logo">∞ PRIME<small>INTELLIGENCE</small></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("☀  Light", key="btn_light", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()
    with c2:
        if st.button("◑  Dark",  key="btn_dark",  use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()

    st.markdown("---")

    new_path = st.text_input(
        "Dataset path",
        value=str(st.session_state.csv_path),
        key="csv_path_input",
        placeholder="path/to/primes_with_gaps.csv",
    )
    if new_path and new_path != st.session_state.csv_path:
        st.session_state.csv_path = new_path
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        f'<p style="font-family:\'Space Mono\',monospace;font-size:10px;'
        f'letter-spacing:2px;text-transform:uppercase;color:{T["muted"]};margin-bottom:10px;">'
        f'NAVIGATE</p>',
        unsafe_allow_html=True,
    )

    PAGES = [
        ("📊", "overview",  "Overview"),
        ("📈", "pnt",       "Prime Number Thm"),
        ("🔬", "gaps",      "Gap Analysis"),
        ("🌀", "ulam",      "Ulam Spiral"),
        ("🧮", "aks",       "AKS Visualization"),
        ("🔢", "density",   "Density & Digits"),
        ("🤖", "ml",        "ML Regression"),
        ("🔮", "mersenne",  "Mersenne Primes"),
    ]
    for icon, key, label in PAGES:
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown(
        f'<p style="font-family:\'Space Mono\',monospace;font-size:10px;color:{T["border"]};">'
        f'v2.2 · Streamlit + Plotly</p>',
        unsafe_allow_html=True,
    )

# ── Read page AFTER sidebar ──
page = st.session_state.page

# ══════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════
try:
    with st.spinner("Loading dataset…"):
        primes_all, gaps_all = load_data(st.session_state.csv_path)
except FileNotFoundError:
    st.error(f"**File not found:** `{st.session_state.csv_path}`")
    st.info("Update the dataset path in the sidebar ←")
    st.stop()
except Exception as e:
    st.error(f"**Could not load CSV:** {e}")
    st.info("Update the dataset path in the sidebar ←")
    st.stop()

total = len(primes_all)

# ══════════════════════════════════════════════
#  DATASET SLICE SLIDER  (skip on mersenne page)
# ══════════════════════════════════════════════
max_show     = min(total, 200_000)
safe_default = min(st.session_state.prime_limit, max_show)

if page != "mersenne":
    st.markdown(
        f'<p style="font-family:\'Space Mono\',monospace;font-size:10px;letter-spacing:2px;'
        f'text-transform:uppercase;color:{T["muted"]};margin-bottom:2px;">⚙ DATASET SLICE</p>',
        unsafe_allow_html=True,
    )
    prime_limit = st.slider(
        "primes_to_analyse",
        min_value=1_000, max_value=max_show, value=safe_default,
        step=1_000, format="%d primes",
        label_visibility="collapsed", key="prime_limit_slider",
    )
    st.session_state.prime_limit = prime_limit
else:
    prime_limit = safe_default

primes        = primes_all[:prime_limit]
gaps          = gaps_all[:min(prime_limit, len(gaps_all))]
# FIX: primes_subset was capped at 50k, causing Ulam spiral to miss primes
# beyond that range. Now derived from the active slice so the slider controls it.
primes_subset = primes[:50_000] if len(primes) > 50_000 else primes
primes_small  = primes[:1_000]
gaps_small    = gaps[:1_000]

# ══════════════════════════════════════════════
#  STAT CARDS  (skip on mersenne page — it draws its own)
# ══════════════════════════════════════════════
if page != "mersenne":
    mean_gap = float(np.mean(gaps))
    max_gap  = int(np.max(gaps))
    skew_gap = float(scipy.stats.skew(gaps))
    corr_val = float(np.corrcoef(primes_small[1:], gaps_small[1:])[0, 1])

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card g">
        <div class="stat-lbl">Primes Loaded</div>
        <div class="stat-val g">{len(primes):,}</div>
        <div class="stat-hint">of {total:,} total</div>
      </div>
      <div class="stat-card b">
        <div class="stat-lbl">Mean Gap</div>
        <div class="stat-val b">{mean_gap:.2f}</div>
        <div class="stat-hint">avg spacing</div>
      </div>
      <div class="stat-card r">
        <div class="stat-lbl">Max Gap</div>
        <div class="stat-val r">{max_gap}</div>
        <div class="stat-hint">largest gap</div>
      </div>
      <div class="stat-card y">
        <div class="stat-lbl">Skewness</div>
        <div class="stat-val y">{skew_gap:.3f}</div>
        <div class="stat-hint">right-skewed</div>
      </div>
      <div class="stat-card b">
        <div class="stat-lbl">P–G Corr</div>
        <div class="stat-val b">{corr_val:.4f}</div>
        <div class="stat-hint">prime vs gap</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

META = {
    "overview":  ("PRIME NUMBERS",        "Statistical overview"),
    "pnt":       ("PRIME NUMBER THEOREM",  "π(x) vs x / log(x) — convergence validation"),
    "gaps":      ("GAP ANALYSIS",          "Distribution · KDE · CDF of prime gaps"),
    "ulam":      ("ULAM SPIRAL",           "Polar embedding with animated reveal"),
    "aks":       ("AKS PRIMALITY TEST",    "Binomial coefficients mod n — prime witness"),
    "density":   ("DENSITY & LAST DIGITS", "Density histogram + Dirichlet digit distribution"),
    "ml":        ("ML REGRESSION",         "Linear model — prime magnitude vs gap size"),
    "mersenne":  ("MERSENNE PRIMES",       "2^p − 1 · exponents · digit growth · gap distributions · discovery timeline"),
}
title, subtitle = META.get(page, ("PRIME NUMBERS", ""))
st.markdown(f"""
<div class="hero">
  <div class="hero-tag">∞ Prime Numbers · {len(primes):,} primes loaded</div>
  <div class="hero-title"><span>{title[:5]}</span>{title[5:]}</div>
  <div class="hero-sub">{subtitle}</div>
</div>
""", unsafe_allow_html=True)


if page == "overview":
    st.markdown('<p class="sec-head"><span>01</span>DATASET SUMMARY</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = sfig()
        fig.add_trace(go.Scatter(
            x=primes, y=np.arange(1, len(primes) + 1),
            mode="lines", line=dict(color=A, width=2), name="π(x)"))
        chart(fig, "Cumulative Prime Count π(x)", "How many primes exist up to x")

    with c2:
        fig = sfig()
        fig.add_trace(go.Scatter(
            x=np.arange(len(gaps)), y=gaps,
            mode="lines", line=dict(color=A2, width=1), name="gap"))
        chart(fig, "Prime Gap Over Index", "Gap between consecutive primes")

    st.markdown(f"""
    <div class="info-box">
      Variance: <strong>{float(np.var(gaps)):,.2f}</strong> &nbsp;|&nbsp;
      Std Dev: <strong>{float(np.std(gaps)):.2f}</strong> &nbsp;|&nbsp;
      Largest prime: <strong>{int(primes[-1]):,}</strong> &nbsp;|&nbsp;
      Most common gap: <strong>{int(pd.Series(gaps).mode()[0])}</strong>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  ② PNT
# ══════════════════════════════════════════════
elif page == "pnt":
    st.markdown('<p class="sec-head"><span>02</span>PRIME NUMBER THEOREM</p>', unsafe_allow_html=True)
    approx = primes / np.log(primes)

    fig = sfig(460)
    fig.add_trace(go.Scatter(x=primes, y=np.arange(1, len(primes) + 1),
                             mode="lines", name="π(x)", line=dict(color=A, width=2)))
    fig.add_trace(go.Scatter(x=primes, y=approx, mode="lines", name="x / log(x)",
                             line=dict(color=A3, width=2, dash="dash")))
    chart(fig, "π(x) vs x / log(x)", "PNT states π(x) ~ x / ln(x) as x → ∞", h=460)

    c1, c2 = st.columns(2)
    with c1:
        ratio = np.arange(1, len(primes) + 1) / approx
        fig = sfig()
        fig.add_trace(go.Scatter(x=primes, y=ratio, mode="lines",
                                 line=dict(color=A2, width=1.5), name="ratio"))
        fig.add_hline(y=1, line_dash="dot", line_color=A3,
                      annotation_text="y=1", annotation_font_color=T["muted"])
        chart(fig, "Ratio π(x) / (x/log x)", "Converges to 1 by PNT")
    with c2:
        fig = sfig()
        fig.add_trace(go.Scatter(x=primes, y=np.arange(1, len(primes) + 1) - approx,
                                 mode="lines", line=dict(color=AY, width=1.5), name="error"))
        chart(fig, "Error: π(x) − x/log(x)", "Deviation from the approximation")

# ══════════════════════════════════════════════
#  ③ GAPS
# ══════════════════════════════════════════════
elif page == "gaps":
    st.markdown('<p class="sec-head"><span>03</span>GAP ANALYSIS</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(gaps, nbins=100, color_discrete_sequence=[A])
        fig.update_layout(**plot_layout())
        fig.update_layout(showlegend=False)
        chart(fig, "Gap Distribution — Histogram", "Frequency of each gap size")

    with c2:
        kde   = scipy.stats.gaussian_kde(gaps)
        xs    = np.linspace(float(gaps.min()), float(gaps.max()), 300)
        alpha = "0.08" if st.session_state.theme == "dark" else "0.15"
        fig   = sfig()
        fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode="lines", fill="tozeroy",
                                 line=dict(color=A2, width=2),
                                 fillcolor=f"rgba(87,212,255,{alpha})", name="KDE"))
        fig.update_xaxes(title_text="Prime Gap")
        fig.update_yaxes(title_text="Prob. Density")
        chart(fig, "Kernel Density Estimate", "Smooth probability density of gap sizes")

    sorted_g = np.sort(gaps)
    cdf_y    = np.arange(len(sorted_g)) / len(sorted_g)
    fig = sfig(300)
    fig.add_trace(go.Scatter(x=sorted_g, y=cdf_y, mode="lines",
                             line=dict(color=A3, width=2), name="CDF"))
    chart(fig, "Cumulative Distribution Function (CDF)", "Probability that a random gap ≤ x")

# ══════════════════════════════════════════════
#  ④ ULAM SPIRAL
# ══════════════════════════════════════════════
elif page == "ulam":
    st.markdown('<p class="sec-head"><span>04</span>ULAM SPIRAL</p>',
                unsafe_allow_html=True)
    n = st.slider("Integers to embed", 5_000, min(100_000, total), 30_000, 5_000, key="ulam_n")

    # FIX: was using primes_subset (capped at 50k globally), which meant
    # numbers above the 50,000th prime were never marked as prime when
    # the slider was set beyond that range. Now we build a local set
    # from the active slice so coverage always matches the slider.
    ulam_prime_set = set(primes_subset.tolist())

    numbers    = np.arange(1, n + 1)
    theta_s    = numbers * 0.08
    r_s        = np.sqrt(numbers)
    x_s        = r_s * np.cos(theta_s)
    y_s        = r_s * np.sin(theta_s)
    prime_mask = np.array([num in ulam_prime_set for num in numbers])

    step_size  = 3_000
    frames, steps = [], []
    for i in range(step_size, n + 1, step_size):
        m = (numbers <= i) & prime_mask
        frames.append(go.Frame(
            data=[go.Scatter(x=x_s[m], y=y_s[m], mode="markers",
                             marker=dict(size=2.5, color=np.log(numbers[m]),
                                         colorscale="Turbo", line=dict(width=0), opacity=0.85,
                                         showscale=False),
                             text=numbers[m],
                             hovertemplate="n = %{text}<extra>Prime</extra>")],
            name=str(i),
        ))
        steps.append({"method": "animate",
                      "args": [[str(i)], {"mode": "immediate"}],
                      "label": f"{i:,}"})

    im  = (numbers <= step_size) & prime_mask
    bg  = "#000000" if st.session_state.theme == "dark" else "#ffffff"
    fig = go.Figure(
        data=[go.Scatter(x=x_s[im], y=y_s[im], mode="markers",
                         marker=dict(size=2.5, color=np.log(numbers[im]), colorscale="Inferno"),
                         text=numbers[im],
                         hovertemplate="n = %{text}<extra>Prime</extra>")],
        frames=frames,
    )
    fig.update_layout(
        title="",
        paper_bgcolor=bg, plot_bgcolor=bg,
        xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
        height=640, margin=dict(l=0, r=0, t=0, b=0),
        sliders=[{"active": 0, "steps": steps,
                  "currentvalue": {"prefix": "n = ", "font": {"size": 13, "color": A}},
                  "bgcolor": T["surface"], "bordercolor": T["border"]}],
        updatemenus=[{"type": "buttons", "bgcolor": T["surface"], "bordercolor": T["border"],
                      "font": {"color": A},
                      "buttons": [
                          {"label": "▶ Play", "method": "animate",
                           "args": [None, {"frame": {"duration": 80, "redraw": True}}]},
                          {"label": "⏸ Pause", "method": "animate",
                           "args": [[None], {"frame": {"duration": 0}}]},
                      ]}],
    )
    # FIX: was `width="stretch"` which is not a valid Streamlit argument.
    # Replaced with use_container_width=True to match all other chart calls.
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
#  ⑤ AKS
# ══════════════════════════════════════════════
elif page == "aks":
    st.markdown('<p class="sec-head"><span>05</span>AKS PRIMALITY TEST</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      For a prime p, all interior binomial coefficients C(p,k) mod p = 0 for 0 &lt; k &lt; p.
      <strong>Red bars</strong> are composite witnesses.
    </div>""", unsafe_allow_html=True)
    n_val  = st.slider("Select n", 5, 40, 17, key="aks_n")
    coeffs = [comb(n_val, k) % n_val for k in range(n_val + 1)]
    colors = [A3 if (c != 0 and k not in [0, n_val]) else A for k, c in enumerate(coeffs)]
    nz     = sum(1 for k, c in enumerate(coeffs) if c != 0 and k not in [0, n_val])
    passed = nz == 0
    verdict_color = A if passed else A3
    verdict_text  = "✓ PRIME — all interior coefficients are 0 mod n" if passed \
                    else f"✗ COMPOSITE — {nz} non-zero coefficient(s) detected"
    st.markdown(f"""
    <div class="info-box">
      n = <strong>{n_val}</strong> →
      <span style="color:{verdict_color}">{verdict_text}</span>
    </div>""", unsafe_allow_html=True)
    fig = sfig(420)
    fig.add_trace(go.Bar(x=list(range(n_val + 1)), y=coeffs,
                         marker_color=colors, name=f"C({n_val},k) mod {n_val}"))
    fig.update_xaxes(title_text="k", title_font_color=T["muted"])
    fig.update_yaxes(title_text=f"C(n,k) mod {n_val}", title_font_color=T["muted"])
    chart(fig, f"C({n_val}, k) mod {n_val}", "Interior non-zero bars = composite witness")


elif page == "density":
    st.markdown('<p class="sec-head"><span>06</span>DENSITY &amp; LAST DIGITS</p>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        bins          = np.arange(0, int(primes_subset.max()) + 1000, 1000)
        counts, edges = np.histogram(primes_subset, bins=bins)
        fig = sfig()
        fig.add_trace(go.Bar(x=edges[:-1], y=counts, marker_color=A2, name="density"))
        chart(fig, "Prime Density per 1,000", "Count of primes in each 1,000-wide bin")

    with c2:
        last_d = primes % 10
        digs   = [1, 3, 7, 9]
        dcols  = [A, A2, A3, AY]
        fig    = sfig()
        for d, col in zip(digs, dcols):
            fig.add_trace(go.Bar(x=[str(d)], y=[int(np.sum(last_d == d))],
                                 marker_color=col, name=f"…{d}"))
        fig.update_layout(barmode="group")
        chart(fig, "Last Digit Distribution", "Primes > 5 end only in 1, 3, 7, 9")

elif page == "ml":
    st.markdown('<p class="sec-head"><span>07</span>ML REGRESSION</p>', unsafe_allow_html=True)
    X      = primes_small[1:].reshape(-1, 1)
    y_reg  = gaps_small[1:]
    model  = LinearRegression().fit(X, y_reg)
    y_pred = model.predict(X)
    resid  = y_reg - y_pred
    c1, c2 = st.columns(2)

    with c1:
        fig = sfig()
        fig.add_trace(go.Scatter(x=primes_small[1:], y=y_reg, mode="markers", name="Actual",
                                 marker=dict(size=4, color=A2, opacity=0.5)))
        fig.add_trace(go.Scatter(x=primes_small[1:], y=y_pred, mode="lines", name="Fit",
                                 line=dict(color=A3, width=2)))
        chart(fig, "Linear Regression: Prime vs Gap",
              "Slope shows slow growth of gaps with prime magnitude")

    with c2:
        fig = sfig()
        fig.add_trace(go.Scatter(x=primes_small[1:], y=resid, mode="markers",
                                 marker=dict(size=4, color=A, opacity=0.5), name="residual"))
        fig.add_hline(y=0, line_dash="dot", line_color=A3)
        chart(fig, "Residual Plot", "Actual − Predicted; should be random noise")

    st.markdown(f"""
    <div class="info-box">
      Slope: <strong>{model.coef_[0]:.6f}</strong> &nbsp;|&nbsp;
      Intercept: <strong>{model.intercept_:.4f}</strong> &nbsp;|&nbsp;
      R² (train): <strong>{model.score(X, y_reg):.4f}</strong>
    </div>""", unsafe_allow_html=True)

    df_c = pd.DataFrame({"prime": primes_small[1:], "gap": gaps_small[1:]}).corr()
    fig  = px.imshow(df_c, text_auto=True, color_continuous_scale="RdYlGn", zmin=-1, zmax=1)
    fig.update_layout(**plot_layout(300))
    chart(fig, "Correlation Matrix", "Pearson r between prime and gap")

# ══════════════════════════════════════════════
#  ⑧ MERSENNE PRIMES
# ══════════════════════════════════════════════
elif page == "mersenne":

    largest_exp    = int(df_m["exponent"].max())
    largest_digits = int(df_m["digits"].max())
    # FIX: iloc[1:] correctly skips the diff=0 row for rank 1
    exp_gaps_all   = df_m["exp_gap"].iloc[1:].values
    mean_exp_gap   = float(exp_gaps_all.mean())
    max_exp_gap    = int(exp_gaps_all.max())

    # FIX: stat card hint previously said "M51 digit count" but M52 is the
    # largest known Mersenne prime (2^136,279,841 − 1, discovered 2024).
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card g">
        <div class="stat-lbl">Known Mersennes</div>
        <div class="stat-val g">{len(df_m)}</div>
        <div class="stat-hint">as of 2024</div>
      </div>
      <div class="stat-card b">
        <div class="stat-lbl">Largest Exponent</div>
        <div class="stat-val b">{largest_exp:,}</div>
        <div class="stat-hint">p in 2^p − 1</div>
      </div>
      <div class="stat-card r">
        <div class="stat-lbl">Largest Digits</div>
        <div class="stat-val r">{largest_digits:,}</div>
        <div class="stat-hint">M52 digit count</div>
      </div>
      <div class="stat-card y">
        <div class="stat-lbl">Mean Exp Gap</div>
        <div class="stat-val y">{mean_exp_gap:,.0f}</div>
        <div class="stat-hint">avg between exponents</div>
      </div>
      <div class="stat-card b">
        <div class="stat-lbl">Max Exp Gap</div>
        <div class="stat-val b">{max_exp_gap:,}</div>
        <div class="stat-hint">largest jump</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sec-head"><span>08</span>MERSENNE PRIMES — 2^p − 1</p>',
                unsafe_allow_html=True)

    # FIX: info box previously said "all 51 known" but there are 52 known as of 2024
    st.markdown("""
    <div class="info-box">
      A <strong>Mersenne prime</strong> has the form 2^p − 1 where p must itself be prime.
      Not every prime p works — e.g. 2^11 − 1 = 2047 = 23 × 89 (composite).
      All 52 known Mersenne primes are shown here. The search continues through
      <strong>GIMPS</strong> (Great Internet Mersenne Prime Search, est. 1996).
    </div>""", unsafe_allow_html=True)

    # ── Row 1: Exponent growth (log scale) + Digit count bar ──
    c1, c2 = st.columns(2)

    with c1:
        fig = sfig(400)
        fig.add_trace(go.Scatter(
            x=df_m["rank"], y=df_m["exponent"],
            mode="markers+lines",
            marker=dict(size=7, color=A, line=dict(color=T["border"], width=1)),
            line=dict(color=A, width=1.5, dash="dot"),
            name="exponent p",
            hovertemplate="M%{x}: p = %{y:,}<extra></extra>",
        ))
        fig.update_xaxes(title_text="Mersenne rank")
        fig.update_yaxes(title_text="Exponent p", type="log")
        chart(fig, "Mersenne Exponents (log scale)",
              "Each point is p where 2^p − 1 is prime. Growth is super-exponential.")

    with c2:
        fig = sfig(400)
        fig.add_trace(go.Bar(
            x=df_m["rank"], y=df_m["digits"],
            marker=dict(color=df_m["digits"],
                        colorscale=[[0, A2], [1, A]], showscale=False),
            name="digits",
            hovertemplate="M%{x}: %{y:,} digits<extra></extra>",
        ))
        fig.update_xaxes(title_text="Mersenne rank")
        fig.update_yaxes(title_text="Number of digits")
        # FIX: previously said "M51 has over 24 million digits" but M52 has ~41 million
        chart(fig, "Digit Count per Mersenne Prime",
              "Digit count ≈ p × log₁₀(2). M52 has over 41 million digits.")

    # ── Row 2: Exponent gap bar + log-log digit vs exponent ──
    c1, c2 = st.columns(2)

    with c1:
        df_gp    = df_m.iloc[1:].copy()
        gap_cols = [A3 if g > mean_exp_gap else A2 for g in df_gp["exp_gap"]]
        fig = sfig(400)
        fig.add_trace(go.Bar(
            x=df_gp["rank"], y=df_gp["exp_gap"],
            marker_color=gap_cols, name="exponent gap",
            hovertemplate="M%{x}: gap = %{y:,}<extra></extra>",
        ))
        fig.add_hline(
            y=mean_exp_gap, line_dash="dot", line_color=AY,
            annotation_text=f"mean {mean_exp_gap:,.0f}",
            annotation_font_color=AY, annotation_position="top right",
        )
        fig.update_xaxes(title_text="Mersenne rank")
        fig.update_yaxes(title_text="Gap between consecutive exponents")
        chart(fig, "Gaps Between Consecutive Mersenne Exponents",
              "Red bars exceed the mean gap. Large jumps reflect hard-to-find primes.")

    with c2:
        p_range = np.linspace(2, largest_exp, 300)
        fig = sfig(400)
        fig.add_trace(go.Scatter(
            x=df_m["exponent"], y=df_m["digits"],
            mode="markers",
            marker=dict(size=8, color=AY, opacity=0.85,
                        line=dict(color=T["border"], width=1)),
            name="observed",
            hovertemplate="p=%{x:,} → %{y:,} digits<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=p_range, y=p_range * np.log10(2),
            mode="lines", line=dict(color=A3, width=2, dash="dash"),
            name="p · log₁₀(2)",
        ))
        fig.update_xaxes(title_text="Exponent p", type="log")
        fig.update_yaxes(title_text="Digits", type="log")
        chart(fig, "Digit Count vs Exponent (log–log)",
              "Points hug the line digits = p·log₁₀(2) almost perfectly.")

    # ── Row 3: Discovery timeline + Exponent gap histogram + KDE ──
    c1, c2 = st.columns(2)

    with c1:
        df_disc   = df_m[df_m["year"] > 0].sort_values("year").reset_index(drop=True)
        cumul     = np.arange(1, len(df_disc) + 1)
        fig = sfig(400)
        fig.add_vrect(
            x0=1996, x1=int(df_disc["year"].max()) + 2,
            fillcolor=A, opacity=0.04, line_width=0,
            annotation_text="GIMPS era", annotation_position="top left",
            annotation_font_color=A, annotation_font_size=10,
        )
        fig.add_vline(x=1996, line_dash="dot", line_color=A,
                      annotation_text="1996 GIMPS",
                      annotation_font_color=A, annotation_position="top right")
        fig.add_trace(go.Scatter(
            x=df_disc["year"], y=cumul,
            mode="markers+lines",
            marker=dict(size=7, color=A2, line=dict(color=T["border"], width=1)),
            line=dict(color=A2, width=2),
            name="cumulative",
            hovertemplate="Year %{x}: %{y} known<extra></extra>",
        ))
        fig.update_xaxes(title_text="Year of discovery")
        fig.update_yaxes(title_text="Cumulative Mersenne primes")
        chart(fig, "Cumulative Mersenne Primes Discovered",
              "Discovery surged after GIMPS launched distributed computing in 1996.")

    with c2:
        fig = sfig(400)
        fig.add_trace(go.Histogram(
            x=exp_gaps_all, nbinsx=18,
            marker_color=A, opacity=0.8, name="gap frequency",
            hovertemplate="Gap ≈ %{x:,}: count %{y}<extra></extra>",
        ))
        if len(exp_gaps_all) > 3:
            kde_m  = scipy.stats.gaussian_kde(exp_gaps_all)
            xs_m   = np.linspace(exp_gaps_all.min(), exp_gaps_all.max(), 250)
            bin_w  = (exp_gaps_all.max() - exp_gaps_all.min()) / 18
            scale  = len(exp_gaps_all) * bin_w
            fig.add_trace(go.Scatter(
                x=xs_m, y=kde_m(xs_m) * scale,
                mode="lines", line=dict(color=A3, width=2, dash="dash"),
                name="KDE",
            ))
        fig.add_vline(x=mean_exp_gap, line_dash="dot", line_color=AY,
                      annotation_text=f"mean {mean_exp_gap:,.0f}",
                      annotation_font_color=AY, annotation_position="top right")
        fig.update_xaxes(title_text="Gap size between exponents")
        fig.update_yaxes(title_text="Frequency")
        chart(fig, "Exponent Gap Distribution + KDE",
              "Highly right-skewed — most gaps are small, a few are enormous.")

    # ── Row 4: Gap CDF + Box plot ──
    c1, c2 = st.columns(2)

    with c1:
        sorted_eg = np.sort(exp_gaps_all)
        cdf_eg    = np.arange(1, len(sorted_eg) + 1) / len(sorted_eg)
        med_gap   = float(np.median(exp_gaps_all))
        fig = sfig(360)
        fig.add_trace(go.Scatter(
            x=sorted_eg, y=cdf_eg,
            mode="lines+markers",
            marker=dict(size=5, color=A2),
            line=dict(color=A2, width=2),
            name="CDF",
            hovertemplate="Gap ≤ %{x:,}: P = %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=0.5, line_dash="dot", line_color=AY,
                      annotation_text=f"median {med_gap:,.0f}",
                      annotation_font_color=AY)
        fig.update_xaxes(title_text="Exponent gap")
        fig.update_yaxes(title_text="Cumulative probability")
        chart(fig, "Exponent Gap CDF",
              "50% of all gaps fall below the median; the long right tail is clear.")

    with c2:
        fig = sfig(360)
        fig.add_trace(go.Box(
            y=exp_gaps_all,
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(size=6, color=AY, opacity=0.75),
            line=dict(color=A),
            fillcolor=f"rgba(200,255,87,0.07)",
            name="exponent gaps",
            hovertemplate="Gap: %{y:,}<extra></extra>",
        ))
        fig.update_yaxes(title_text="Gap size between exponents")
        chart(fig, "Exponent Gap Box Plot",
              "Box = IQR; every individual gap shown. Outliers are the GIMPS-era leaps.")

    # ── Row 5: Gap trend over rank (linear regression) ──
    st.markdown('<p class="sec-head"><span>↳</span>GAP TREND — LINEAR REGRESSION</p>',
                unsafe_allow_html=True)

    df_tr = df_m.iloc[1:].copy().reset_index(drop=True)
    X_t   = df_tr["rank"].values.reshape(-1, 1)
    y_t   = df_tr["exp_gap"].values
    lm    = LinearRegression().fit(X_t, y_t)
    y_fit = lm.predict(X_t)
    resid_m = y_t - y_fit

    c1, c2 = st.columns(2)

    with c1:
        fig = sfig(360)
        fig.add_trace(go.Scatter(
            x=df_tr["rank"], y=df_tr["exp_gap"],
            mode="markers",
            marker=dict(size=7, color=A, opacity=0.85),
            name="gap",
            hovertemplate="M%{x}: gap = %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_tr["rank"], y=y_fit,
            mode="lines",
            line=dict(color=A3, width=2, dash="dash"),
            name=f"fit  R²={lm.score(X_t, y_t):.3f}",
        ))
        fig.update_xaxes(title_text="Mersenne rank")
        fig.update_yaxes(title_text="Exponent gap")
        chart(fig, "Exponent Gap vs Rank — Regression",
              f"Slope = {lm.coef_[0]:,.1f} per rank · R² = {lm.score(X_t, y_t):.3f}")

    with c2:
        fig = sfig(360)
        fig.add_trace(go.Scatter(
            x=df_tr["rank"], y=resid_m,
            mode="markers",
            marker=dict(size=6, color=A2, opacity=0.75),
            name="residual",
            hovertemplate="M%{x}: resid = %{y:,.0f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=A3)
        fig.update_xaxes(title_text="Mersenne rank")
        fig.update_yaxes(title_text="Residual (actual − predicted)")
        chart(fig, "Gap Regression Residuals",
              "Large positive residuals = unexpectedly long gaps (hard searches).")

    # ── Summary info box ──
    # FIX: was incorrectly labelled "M51 = 2^136,279,841 − 1" — that is M52
    st.markdown(f"""
    <div class="info-box">
      Largest known Mersenne prime: <strong>M52 = 2^136,279,841 − 1</strong>
      &nbsp;|&nbsp; Digits: <strong>41,024,320</strong>
      &nbsp;|&nbsp; Discovered: <strong>2024</strong> via GIMPS
      &nbsp;|&nbsp; Mean exponent gap: <strong>{mean_exp_gap:,.0f}</strong>
      &nbsp;|&nbsp; Median exponent gap: <strong>{med_gap:,.0f}</strong>
      &nbsp;|&nbsp; Largest exponent gap: <strong>{max_exp_gap:,}</strong>
      (between M32 and M33, exponents 756,839 → 859,433)
    </div>""", unsafe_allow_html=True)