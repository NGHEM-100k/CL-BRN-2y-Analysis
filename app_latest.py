import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os

st.set_page_config(page_title="Crude Curve Dashboard", layout="wide", page_icon="🛢️")

# ---------------------------------------------------------------
# Styling
# ---------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0b0d11; }
    h1 { color: #e6e8eb !important; font-size: calc(2.2rem * var(--app-font-scale, 1)) !important; }
    h2, h3 { color: #e6e8eb !important; font-size: calc(1.4rem * var(--app-font-scale, 1)) !important; }
    p, span, label, div { font-size: calc(1rem * var(--app-font-scale, 1)); }
    .stMarkdown, .stCaption, [data-testid="stCaptionContainer"] { font-size: calc(0.95rem * var(--app-font-scale, 1)) !important; }
    .stButton button { font-size: calc(1rem * var(--app-font-scale, 1)) !important; padding: 0.5rem 1rem !important; }
    .stRadio label, .stToggle label { font-size: calc(1rem * var(--app-font-scale, 1)) !important; }
    .metric-note { color: #8b93a1; font-size: calc(0.85rem * var(--app-font-scale, 1)); }
</style>
""", unsafe_allow_html=True)

OUT_TARGET = 14
FLY_TARGET = 12
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data(path, max_legs=16):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    price_cols = [f'price_{i}' for i in range(1, max_legs + 1)]
    label_cols = [f'label_{i}' for i in range(1, max_legs + 1)]
    return df, price_cols, label_cols


def get_row_arrays(row, price_cols, label_cols):
    prices, labels = [], []
    for pc, lc in zip(price_cols, label_cols):
        v = row[pc]
        if pd.isna(v):
            break
        prices.append(float(v))
        labels.append(row[lc])
    return prices, labels


def compute_fly(values):
    return [values[i - 1] - 2 * values[i] + values[i + 1] for i in range(1, len(values) - 1)]


def expiry_color(days):
    if days is None or days > 15 or days < 0:
        return '#8b93a1'
    t = 1 - min(days, 15) / 15
    r = int(139 + t * (255 - 139))
    g = int(147 - t * 147)
    b = int(161 - t * 161)
    return f'rgb({r},{g},{b})'


def nice_ticks(vmin, vmax, target=18):
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    rng = vmax - vmin
    raw_step = rng / target
    mag = 10 ** np.floor(np.log10(raw_step)) if raw_step > 0 else 1
    norm = raw_step / mag
    if norm < 1.5:
        step = 1 * mag
    elif norm < 3:
        step = 2 * mag
    elif norm < 7:
        step = 5 * mag
    else:
        step = 10 * mag
    nmin = np.floor(vmin / step) * step
    nmax = np.ceil(vmax / step) * step
    return np.arange(nmin, nmax + step / 2, step)


def build_panel(contracts, today_vals, prev_vals, today_date, prev_date,
                 y_label, change_label, decimals, front_color, show_bars, is_fly, font_scale=1.0):
    fs = lambda base: max(6, round(base * font_scale))
    change = [t - p for t, p in zip(today_vals, prev_vals)]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if show_bars:
        fig.add_trace(go.Bar(
            x=contracts, y=change, name='Net change vs prev day',
            marker_color='#33384a', opacity=0.9,
        ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=contracts, y=prev_vals, name=f'Prev day ({prev_date})',
        mode='lines+markers+text', line=dict(color='#ffb74d', dash='dot', width=2),
        marker=dict(size=max(3, round(5 * font_scale))),
        text=[f'{v:.{decimals}f}' for v in prev_vals], textposition='bottom center',
        textfont=dict(color='#ffb74d', size=fs(10)),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=contracts, y=today_vals, name=f'Today ({today_date})',
        mode='lines+markers+text', line=dict(color='#ffffff', width=2),
        marker=dict(size=max(3, round(5 * font_scale))),
        text=[f'{v:.{decimals}f}' for v in today_vals], textposition='top center',
        textfont=dict(color='#4dd0e1', size=fs(10)),
    ), secondary_y=False)

    allvals = today_vals + prev_vals
    yticks = nice_ticks(min(allvals), max(allvals), 18)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#12151b', plot_bgcolor='#12151b',
        height=round(520 * min(font_scale, 1.5)), margin=dict(t=30, b=40, l=10, r=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=fs(11))),
        hovermode='x unified',
        font=dict(size=fs(11)),
    )
    fig.update_xaxes(
        tickmode='array', tickvals=list(range(len(contracts))), ticktext=contracts,
        tickfont=dict(color='#8b93a1', size=fs(11)),
    )
    # Highlight the front-month tick in its expiry color, bold
    fig.add_annotation(
        x=0, y=-0.12, xref='x', yref='paper', text=f"<b>{contracts[0]}</b>",
        showarrow=False, font=dict(color=front_color, size=fs(13)), yanchor='top'
    )

    fig.update_yaxes(
        title_text=y_label, tickvals=yticks, tickfont=dict(color='#d8dce3', size=fs(10)),
        title_font=dict(color='#ffffff', size=fs(12)), secondary_y=False, gridcolor='#1d2129',
        tickformat=f'.{decimals}f',
    )
    if show_bars:
        fig.update_yaxes(
            title_text=change_label, tickfont=dict(color='#a9adba', size=fs(10)),
            title_font=dict(color='#a9adba', size=fs(12)), secondary_y=True, gridcolor='#1d2129',
            zeroline=True, zerolinecolor='#3a3f4b',
        )
    else:
        fig.update_yaxes(visible=False, secondary_y=True)

    return fig


# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
co_df, co_pcols, co_lcols = load_data(os.path.join(BASE_DIR, 'brent_curve.csv'))
cl_df, cl_pcols, cl_lcols = load_data(os.path.join(BASE_DIR, 'wti_curve.csv'))

# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
if 'idx' not in st.session_state:
    st.session_state.idx = 1
if 'playing' not in st.session_state:
    st.session_state.playing = False

st.title("🛢️ Crude Curve Dashboard")

top1, top2 = st.columns([3, 1])
with top1:
    st.caption("Daily outright & 1-month fly curve — front month labeled at the true front leg, capped at 14 (outright) / 12 (fly) months")
with top2:
    asset = st.radio("Asset", ["Brent (CO)", "WTI (CL)"], horizontal=True, label_visibility="collapsed")

df, pcols, lcols = (co_df, co_pcols, co_lcols) if asset.startswith("Brent") else (cl_df, cl_pcols, cl_lcols)
n_rows = len(df)

if st.session_state.idx >= n_rows:
    st.session_state.idx = 1

# ---------------------------------------------------------------
# Font size control (applies to headers/captions via CSS var + chart fonts)
# ---------------------------------------------------------------
fcol1, fcol2 = st.columns([1, 5])
with fcol1:
    st.markdown("<div style='padding-top:8px;font-weight:600;'>🔠 Font size</div>", unsafe_allow_html=True)
with fcol2:
    font_scale = st.slider("Font size", 0.7, 2.0, 1.2, 0.1, label_visibility="collapsed")

st.markdown(f"<style>:root {{ --app-font-scale: {font_scale}; }}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Play / slider / date
# ---------------------------------------------------------------
c1, c2, c3 = st.columns([1, 6, 2])
with c1:
    label = "⏸ Pause" if st.session_state.playing else "▶ Play"
    if st.button(label, use_container_width=True):
        st.session_state.playing = not st.session_state.playing
with c2:
    idx = st.slider("Date index", 1, n_rows - 1, st.session_state.idx, label_visibility="collapsed")
    if idx != st.session_state.idx:
        st.session_state.idx = idx
        st.session_state.playing = False
with c3:
    st.markdown(f"<div style='text-align:right;font-weight:700;color:#4dd0e1;padding-top:6px;font-size:calc(1.1rem * {font_scale})'>{df.iloc[st.session_state.idx]['date'].strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Speed control — explicit buttons (always visible, not a squished slider)
# ---------------------------------------------------------------
if 'speed_label' not in st.session_state:
    st.session_state.speed_label = '1x'

SPEED_MAP = {"0.1x": 2.0, "0.25x": 1.0, "0.5x": 0.5, "1x": 0.25, "2x": 0.12, "4x": 0.06, "8x": 0.03}

scol1, *speed_cols, scol_bars = st.columns([1] + [1] * len(SPEED_MAP) + [2])
with scol1:
    st.markdown("<div style='padding-top:8px;font-weight:600;'>⏱ Speed</div>", unsafe_allow_html=True)
for col, sp in zip(speed_cols, SPEED_MAP.keys()):
    with col:
        is_active = st.session_state.speed_label == sp
        if st.button(sp, key=f'speed_{sp}', use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.speed_label = sp
with scol_bars:
    show_bars = st.toggle("Show net-change bars", value=True)

speed_ms = SPEED_MAP[st.session_state.speed_label]

st.markdown(f"<div class='metric-note'>{st.session_state.idx} / {n_rows - 1}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Data for current + previous day
# ---------------------------------------------------------------
row_today = df.iloc[st.session_state.idx]
row_prev = df.iloc[st.session_state.idx - 1]

today_prices, today_labels = get_row_arrays(row_today, pcols, lcols)
prev_prices, prev_labels = get_row_arrays(row_prev, pcols, lcols)

days_to_roll = row_today['days_to_roll']
front_color = expiry_color(days_to_roll)
near_expiry = 0 <= days_to_roll <= 15

if near_expiry:
    st.markdown(
        f"<div style='color:{front_color};font-size:13px;margin-bottom:8px'>"
        f"● Front contract ({today_labels[0]}) expires in {int(days_to_roll)} day(s)</div>",
        unsafe_allow_html=True
    )

# ---- Outright (capped at 14) ----
n_out = min(len(today_labels), len(prev_labels), OUT_TARGET)
out_contracts = today_labels[:n_out]
out_today = today_prices[:n_out]
out_prev = prev_prices[:n_out]
out_short = n_out < OUT_TARGET

st.subheader(f"{asset} — Outright curve ({n_out}{' of ' + str(OUT_TARGET) if out_short else ''} months)")
fig_out = build_panel(out_contracts, out_today, out_prev, row_today['date'].strftime('%Y-%m-%d'),
                       row_prev['date'].strftime('%Y-%m-%d'), "Outright price ($/bbl)",
                       "Δ vs prev day ($/bbl)", 2, front_color, show_bars, is_fly=False)
st.plotly_chart(fig_out, use_container_width=True)
if out_short:
    st.caption(f"Only {n_out} of the target {OUT_TARGET} months are shown for {row_today['date'].strftime('%Y-%m-%d')} "
               f"— this date's file only has {len(today_prices)} forward months quoted.")

# ---- Fly (capped at 12, labeled by FRONT leg) ----
today_fly_full = compute_fly(today_prices)
prev_fly_full = compute_fly(prev_prices)
n_fly = min(len(today_fly_full), len(prev_fly_full), FLY_TARGET)
fly_contracts = today_labels[:n_fly]          # labeled by front leg, e.g. Q24
fly_today = today_fly_full[:n_fly]
fly_prev = prev_fly_full[:n_fly]
fly_short = n_fly < FLY_TARGET

st.subheader(f"{asset} — 1-Month Fly curve ({n_fly}{' of ' + str(FLY_TARGET) if fly_short else ''} points)")
fig_fly = build_panel(fly_contracts, fly_today, fly_prev, row_today['date'].strftime('%Y-%m-%d'),
                       row_prev['date'].strftime('%Y-%m-%d'), "Fly value ($/bbl)",
                       "Δ vs prev day ($/bbl)", 3, front_color, show_bars, is_fly=True)
st.plotly_chart(fig_fly, use_container_width=True)
if fly_short:
    st.caption(f"Only {n_fly} of the target {FLY_TARGET} fly points are shown for {row_today['date'].strftime('%Y-%m-%d')} "
               f"— a 12-point fly needs 14 consecutive forward months, and this date's file only has {len(today_prices)}.")

# ---------------------------------------------------------------
# Autoplay loop
# ---------------------------------------------------------------
if st.session_state.playing:
    time.sleep(speed_ms)
    if st.session_state.idx < n_rows - 1:
        st.session_state.idx += 1
    else:
        st.session_state.playing = False
    st.rerun()
