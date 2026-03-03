import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import (
    THEME_CSS, PITCH_SPIN_AXIS_IDEAL, VELO_BENCHMARKS,
    IVB_BENCHMARKS, HB_BENCHMARKS, GROUP_TO_DISPLAY,
)

# ---- Wrist mechanics data ----
WRIST_COMPAT = {
    'Pronator': {
        'Fastball':  ('Natural',   '#3dcc7c', "Backspin via pronation — your most natural pitch."),
        'Sinker':    ('Natural',   '#3dcc7c', "Arm-side sink via pronation — ideal motion."),
        'Changeup':  ('Natural',   '#3dcc7c', "Same arm path as fastball — natural arm-side fade."),
        'Cutter':    ('Workable',  '#f0c040', "Mild supination required — monitor elbow load."),
        'Slider':    ('Workable',  '#f0c040', "Moderate supination needed — may sacrifice sharpness."),
        'Sweeper':   ('Workable',  '#f0c040', "Less supination than curveball — achievable for most pronators."),
        'Curveball': ('Unnatural', '#f05050', "Significant supination against your natural motion — elevated elbow stress."),
    },
    'Supinator': {
        'Fastball':  ('Workable',  '#f0c040', "Achievable backspin, but arm-side ride may be reduced."),
        'Sinker':    ('Difficult', '#f05050', "Pronation-dependent — supinators typically cannot generate natural sink."),
        'Changeup':  ('Difficult', '#f05050', "Pronation required for arm-side fade — fights your natural wrist action."),
        'Cutter':    ('Natural',   '#3dcc7c', "Glove-side tilt via supination — a natural fit."),
        'Slider':    ('Natural',   '#3dcc7c', "Supination drives glove-side break — elite natural fit."),
        'Sweeper':   ('Natural',   '#3dcc7c', "Horizontal supination — sweeper is ideal for supinators."),
        'Curveball': ('Natural',   '#3dcc7c', "Full supination — curveball is the most natural pitch for supinators."),
    },
    'Neutral / Both': {
        'Fastball':  ('Flexible',  '#3a9dff', "Full wrist range — can shape backspin or add natural cut."),
        'Sinker':    ('Flexible',  '#3a9dff', "Pronation available — arm-side sink fully achievable."),
        'Changeup':  ('Flexible',  '#3a9dff', "Can pronate — natural deception pitch with full arm action."),
        'Cutter':    ('Flexible',  '#3a9dff', "Either direction available — slot determines ideal shape."),
        'Slider':    ('Flexible',  '#3a9dff', "Can supinate — glove-side break fully accessible."),
        'Sweeper':   ('Flexible',  '#3a9dff', "Full range — can generate horizontal break from either direction."),
        'Curveball': ('Flexible',  '#3a9dff', "Supination available — can execute full 12-6 rotation."),
    },
}

_CANONICAL_PAIR = {
    'Fastball':  ['Changeup', 'Slider'],
    'Sinker':    ['Changeup', 'Sweeper'],
    'Changeup':  ['Fastball', 'Slider'],
    'Slider':    ['Fastball', 'Changeup'],
    'Sweeper':   ['Fastball', 'Changeup'],
    'Curveball': ['Fastball', 'Cutter'],
    'Cutter':    ['Fastball', 'Changeup'],
}

_COMPLEMENT_WHY = {
    ('Fastball',  'Changeup'):  "Same arm action deceives hitters — velocity differential is the weapon. Arm-side fade for extra deception.",
    ('Fastball',  'Slider'):    "Glove-side break contrasts arm-side fastball life. Tunnels under your fastball for late separation.",
    ('Sinker',    'Changeup'):  "Same arm path with less velocity — tunnels your sinker while adding arm-side fade.",
    ('Sinker',    'Sweeper'):   "Horizontal glove-side break contrasts arm-side sinker. Two-way movement threat across the plate.",
    ('Changeup',  'Fastball'):  "Velocity contrast anchors your changeup. Hitters' timing collapses when speeds differ 10–15 mph.",
    ('Changeup',  'Slider'):    "Glove-side break alongside arm-side changeup. FB + CH + SL covers all movement quadrants.",
    ('Slider',    'Fastball'):  "Velocity contrast and vertical separation from the glove-side slider. FB sets up low slider exit velocity.",
    ('Slider',    'Changeup'):  "Arm-side fade alongside glove-side slider — two-way horizontal spread across the plate.",
    ('Sweeper',   'Fastball'):  "Velocity and vertical contrast with horizontal sweeper. Arm-side FB + horizontal sweeper is unsolvable timing.",
    ('Sweeper',   'Changeup'):  "Arm-side deception pairs with horizontal sweeper. FB + CH + SW covers all three movement profiles.",
    ('Curveball', 'Fastball'):  "Classic vertical contrast — curveball sets up elevated fastball. High-low tunnel collapses hitter vision.",
    ('Curveball', 'Cutter'):    "Glove-side cut adds a different look off the curveball. Both pitches play off your vertical breaking ball.",
    ('Cutter',    'Fastball'):  "Velocity and arm-side contrast with glove-side cutter. FB/cutter is one of the most effective pairings in MLB.",
    ('Cutter',    'Changeup'):  "Arm-side fade alongside glove-side cutter — covers both sides of the plate with natural velocity separation.",
}

def _get_complements(pitch_group):
    return _CANONICAL_PAIR.get(pitch_group, [])

def _infer_wrist_type(pitch_group, ivb, hb, spin_eff_pct):
    """
    Infer pronator/supinator bias from pitch movement profile.
    Returns (wrist_type, confidence, note).
    """
    h, v = abs(hb), abs(ivb)

    if pitch_group == 'Sweeper':
        if h >= 16: return ('Supinator', 'High confidence',    f"{h:.0f} in horizontal sweep — strong supinator signature")
        if h >= 11: return ('Supinator', 'Moderate confidence', f"{h:.0f} in sweep — consistent with supinator arm action")
        return ('Neutral / Both', 'Low confidence', "Below-average sweep — atypical sweeper mechanics")

    if pitch_group == 'Curveball':
        if ivb <= -10 and spin_eff_pct >= 75:
            return ('Supinator', 'High confidence',    f"{v:.0f} in drop + {spin_eff_pct}% efficiency — full supination")
        if ivb <= -7:
            return ('Supinator', 'Moderate confidence', f"{v:.0f} in drop — supinator curveball signature")
        if spin_eff_pct < 50:
            return ('Neutral / Both', 'Low confidence', f"Low efficiency ({spin_eff_pct}%) — unusual curveball mechanics")
        return ('Supinator', 'Moderate confidence', "Curveball shape — typically driven by supination")

    if pitch_group == 'Fastball':
        if ivb >= 16 and h <= 8 and spin_eff_pct >= 88:
            return ('Pronator', 'High confidence',    f"{ivb:.0f} in ride + {spin_eff_pct}% efficiency — elite pronator backspin")
        if ivb >= 13 and h <= 10:
            return ('Pronator', 'Moderate confidence', f"Vertical-dominant FB ({ivb:.0f} IVB vs {hb:.0f} HB) — pronation likely")
        if h >= 13:
            return ('Supinator', 'Moderate confidence', f"{h:.0f} in run — seam-shifted supinator fastball")
        if ivb >= 11:
            return ('Pronator', 'Low confidence',      f"Above-avg IVB ({ivb:.0f} in) — slight pronation tendency")
        return ('Neutral / Both', 'Low confidence', "Balanced movement — mixed or neutral mechanics")

    if pitch_group == 'Sinker':
        if h >= 14 and ivb <= 5:
            return ('Supinator', 'Moderate confidence', f"{h:.0f} in run + heavy sink — seam-shifted supinator signature")
        if ivb >= 7 and h <= 8:
            return ('Pronator', 'Moderate confidence',  f"Vertical-dominant sinker ({ivb:.0f} IVB) — pronation bias")
        return ('Neutral / Both', 'Low confidence', "Standard sinker movement — mechanics ambiguous")

    if pitch_group == 'Slider':
        if spin_eff_pct <= 35 and h <= 7 and v <= 5:
            return ('Pronator', 'High confidence',    f"Gyro slider ({spin_eff_pct}% eff, {h:.0f}/{v:.0f} in break) — pronator bullet spin")
        if h >= 10 and spin_eff_pct >= 55:
            return ('Supinator', 'High confidence',    f"Active-spin slider with {h:.0f} in sweep — supinator signature")
        if h >= 7:
            return ('Supinator', 'Moderate confidence', f"{h:.0f} in glove-side break — supination indicator")
        return ('Neutral / Both', 'Low confidence', "Mixed slider mechanics")

    if pitch_group == 'Changeup':
        if h >= 10 and spin_eff_pct >= 80:
            return ('Pronator', 'High confidence',    f"{h:.0f} in arm-side fade + {spin_eff_pct}% efficiency — natural pronation")
        if h >= 7 and ivb >= 6:
            return ('Pronator', 'Moderate confidence', f"Arm-side fade changeup ({h:.0f} in HB) — pronation signature")
        if h < 5:
            return ('Supinator', 'Low confidence',     "Low arm-side fade — may be fighting natural pronation")
        return ('Pronator', 'Low confidence', "Changeup arm action typically reflects pronation")

    if pitch_group == 'Cutter':
        if h >= 8:
            return ('Supinator', 'Moderate confidence', f"{h:.0f} in glove-side cut — supination-driven")
        if h <= 4 and spin_eff_pct >= 70:
            return ('Pronator', 'Low confidence',       "Tight cutter — pronation limiting glove-side break")
        return ('Neutral / Both', 'Low confidence', "Balanced cutter — mechanics ambiguous")

    return ('Neutral / Both', 'Low confidence', "Insufficient data to determine wrist bias")

st.set_page_config(page_title="Pitch Shape Advisor · AlpAnalytics", layout="wide", page_icon="🔬", initial_sidebar_state="expanded")
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-logo">⚾ AlpAnalytics</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="info-box" style="margin-top:1rem;">
    All recommendations are benchmarked against MLB averages and elite thresholds only.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <p class="app-title">Pitch Shape Advisor</p>
    <p class="app-subtitle">AlpAnalytics · MLB-Calibrated · Shape & Spin Targets</p>
</div>
""", unsafe_allow_html=True)

# ---- Inputs ----
st.markdown('<div class="section-header">Pitcher Profile</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([2, 1])
with col_a:
    pitch_group = st.selectbox("Pitch Type", list(GROUP_TO_DISPLAY.keys()),
                               format_func=lambda k: GROUP_TO_DISPLAY[k])
with col_b:
    handedness = st.radio("Handedness", ["R", "L"], horizontal=True,
                          format_func=lambda x: "RHP" if x == "R" else "LHP")

st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("**Velocity & Spin**")
    velo      = st.slider("Velocity (mph)",      min_value=60.0, max_value=105.0, value=93.0, step=0.5)
    spin_rate = st.slider("Spin Rate (rpm)",     min_value=1200, max_value=3800,  value=2300, step=10)
    spin_axis = st.slider("Spin Axis (°)",       min_value=0,    max_value=359,   value=190,  step=1,
                          help="Clock face: 0/360=6 o'clock, 90=9 o'clock, 180=12 o'clock, 270=3 o'clock")
    spin_eff  = st.slider("Spin Efficiency (%)", min_value=0,    max_value=110,   value=92,   step=1,
                          help="Percentage of spin that generates movement (transverse spin). 100% = all active spin.")

with c_right:
    st.markdown("**Movement & Release**")
    hb  = st.slider("Horizontal Break (in)",       min_value=-25.0, max_value=25.0, value=8.0,  step=0.5,
                    help="Arm-side adjusted. Positive = arm-side.")
    ivb = st.slider("Induced Vert Break (in)",     min_value=-25.0, max_value=25.0, value=14.0, step=0.5,
                    help="Positive = ride (resists gravity), negative = depth/drop.")
    ext = st.slider("Extension (ft)",              min_value=4.0,   max_value=7.5,  value=6.2,  step=0.05)
    rel_height = st.slider("Release Height (ft)",  min_value=4.0,   max_value=8.0,  value=6.2,  step=0.05)
    rel_side   = st.slider("Release Side (ft)",    min_value=-4.0,  max_value=4.0,  value=1.5,  step=0.05,
                           help="Distance from center of rubber. Positive = arm side.")
    arm_angle = int(np.clip(np.degrees(np.arctan2(max(0.0, rel_height - 5.0), max(0.01, abs(rel_side)))), 0, 90))
    st.markdown(f"""
    <div style="background:#1c2230;border:1px solid #2a3348;border-radius:4px;padding:0.5rem 0.9rem;margin-top:0.4rem;display:flex;align-items:center;gap:0.8rem;">
        <span style="color:#8a94aa;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;">Calculated Arm Angle</span>
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:700;color:#f0c040;">{arm_angle}°</span>
        <span style="color:#505a70;font-size:0.75rem;">{'Sidearm' if arm_angle < 15 else 'Low 3/4' if arm_angle < 30 else 'Three-Quarter' if arm_angle < 50 else 'High 3/4' if arm_angle < 65 else 'Over the Top'}</span>
    </div>
    """, unsafe_allow_html=True)

# Infer wrist mechanics from the entered movement profile
wrist_type, wrist_conf, wrist_note = _infer_wrist_type(pitch_group, ivb, hb, spin_eff)
_wc = '#3dcc7c' if wrist_type == 'Pronator' else '#f0c040' if wrist_type == 'Supinator' else '#3a9dff'
st.markdown(f"""
<div style="background:#1c2230;border:1px solid #2a3348;border-radius:4px;padding:0.5rem 0.9rem;margin-top:0.6rem;display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap;">
    <span style="color:#8a94aa;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;">Inferred Wrist Mechanics</span>
    <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:700;color:{_wc};">{wrist_type}</span>
    <span style="color:{_wc};font-size:0.72rem;font-weight:600;opacity:0.75;">{wrist_conf}</span>
    <span style="color:#505a70;font-size:0.75rem;flex:1;">{wrist_note}</span>
</div>
""", unsafe_allow_html=True)

analyze = st.button("🔬 Analyze Pitch Shape", use_container_width=False)

if analyze:
    spin_eff_frac = spin_eff / 100.0
    active_spin   = spin_rate * spin_eff_frac
    gyro_spin     = spin_rate * (1 - spin_eff_frac)

    bv   = VELO_BENCHMARKS.get(pitch_group, {})
    bi   = IVB_BENCHMARKS.get(pitch_group, {})
    bh   = HB_BENCHMARKS.get(pitch_group, {})
    bsp  = PITCH_SPIN_AXIS_IDEAL.get(pitch_group, {})

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Shape Analysis</div>', unsafe_allow_html=True)

    # ---- Row 1: 4 metric cards ----
    mc1, mc2, mc3, mc4 = st.columns(4)
    def metric_card(col, label, value, sub, color='#f0c040'):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.7rem;color:{color};">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Velo tier
    if bv:
        if velo >= bv['elite']:   v_tier, v_col = "Elite", "#3dcc7c"
        elif velo >= bv['avg']:   v_tier, v_col = "MLB Avg", "#8a94aa"
        else:                     v_tier, v_col = "Below Avg", "#f05050"
        metric_card(mc1, "Velocity", f"{velo:.1f} mph", f"{v_tier} — MLB avg {bv['avg']} mph", v_col)
    else:
        metric_card(mc1, "Velocity", f"{velo:.1f} mph", "No benchmark available")

    # Spin efficiency
    if pitch_group in ['Curveball', 'Slider', 'Sweeper']:
        if spin_eff > 75:   se_tier, se_col = "High Active", "#3dcc7c"
        elif spin_eff < 30: se_tier, se_col = "Gyro-Heavy", "#3a9dff"
        else:               se_tier, se_col = "Mixed", "#8a94aa"
    else:
        eff_min = bsp.get('topspin_eff_min', 0.75) * 100
        if spin_eff >= eff_min: se_tier, se_col = "High", "#3dcc7c"
        else:                   se_tier, se_col = "Low — leaking gyro", "#f05050"
    metric_card(mc2, "Active Spin", f"{spin_eff}%", f"{se_tier} — {active_spin:.0f} rpm active", se_col)

    # Spin axis
    if bsp:
        lo, hi = bsp['ideal_axis_range']
        in_range = lo <= spin_axis <= hi or (lo > hi and (spin_axis >= lo or spin_axis <= hi))
        if in_range:
            sa_tier, sa_col = "Ideal", "#3dcc7c"
            sa_note = bsp['desc']
        else:
            gap = min(abs(spin_axis - lo), abs(spin_axis - hi))
            sa_tier, sa_col = f"{gap:.0f}° off", "#f05050"
            sa_note = f"Target: {bsp['desc']}"
        metric_card(mc3, "Spin Axis", f"{spin_axis}°", f"{sa_tier} — {sa_note}", sa_col)
    else:
        metric_card(mc3, "Spin Axis", f"{spin_axis}°", "No benchmark available")

    # IVB
    if bi:
        if abs(ivb) >= abs(bi['elite']): ivb_tier, ivb_col = "Elite", "#3dcc7c"
        elif abs(ivb) >= abs(bi['avg']): ivb_tier, ivb_col = "MLB Avg", "#8a94aa"
        else:                             ivb_tier, ivb_col = "Below Avg", "#f05050"
        metric_card(mc4, "Vert Break", f"{ivb:.1f} in", f"{ivb_tier} — avg {bi['avg']} · elite {bi['elite']}", ivb_col)
    else:
        metric_card(mc4, "Vert Break", f"{ivb:.1f} in", "No benchmark available")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ---- Row 2: Recommendations + Movement Plot ----
    rec_col, plot_col = st.columns([1.2, 1])

    with rec_col:
        st.markdown('<div class="section-header">MLB-Calibrated Recommendations</div>', unsafe_allow_html=True)

        def rec_row(label, value, note, badge_class):
            dot = '#3dcc7c' if badge_class == 'badge-gain' else '#f05050' if badge_class == 'badge-warn' else '#3a9dff'
            return f"""
            <tr>
                <td style="padding:7px 12px 7px 0;color:#8a94aa;font-size:0.78rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;white-space:nowrap;">{label}</td>
                <td style="padding:7px 16px 7px 0;color:#e8eaf0;font-size:0.92rem;font-weight:700;white-space:nowrap;">{value}</td>
                <td style="padding:7px 0;color:#8a94aa;font-size:0.82rem;">
                    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{dot};margin-right:6px;vertical-align:middle;"></span>{note}
                </td>
            </tr>"""

        rows_html = ""

        # Velocity recommendation
        if bv:
            if velo >= bv['elite']:
                rows_html += rec_row("Velocity", f"{velo:.1f} mph", f"Elite tier (≥{bv['elite']} mph) — primary weapon", "badge-gain")
            elif velo >= bv['avg']:
                rows_html += rec_row("Velocity", f"{velo:.1f} mph", f"MLB average — movement & tunneling become critical", "badge-info")
            else:
                rows_html += rec_row("Velocity", f"{velo:.1f} mph", f"Below MLB avg ({bv['avg']} mph) — prioritize spin and shape quality", "badge-warn")

        # Spin efficiency recommendation
        if pitch_group in ['Curveball', 'Slider', 'Sweeper']:
            if spin_eff > 75:
                rows_html += rec_row("Active Spin", f"{spin_eff}%", "High active spin → genuine sweep/depth — protect with pitch selection", "badge-gain")
            elif spin_eff < 30:
                rows_html += rec_row("Active Spin", f"{spin_eff}%", "Gyro-heavy → slider cut/bullet — limits sweep, adjust grip for more break", "badge-info")
            else:
                rows_html += rec_row("Active Spin", f"{spin_eff}%", "Mixed spin — tighten axis toward 6 o'clock for depth, 9 o'clock for sweep", "badge-info")
        else:
            eff_min_pct = bsp.get('topspin_eff_min', 0.75) * 100
            if spin_eff >= eff_min_pct:
                rows_html += rec_row("Active Spin", f"{spin_eff}%", f"High active spin → generating ride and life for {GROUP_TO_DISPLAY[pitch_group]}", "badge-gain")
            else:
                rows_html += rec_row("Active Spin", f"{spin_eff}%", f"Leaking gyro spin — target axis: {bsp.get('desc', 'N/A')}", "badge-warn")

        # Spin axis recommendation
        if bsp:
            lo, hi = bsp['ideal_axis_range']
            in_range = lo <= spin_axis <= hi or (lo > hi and (spin_axis >= lo or spin_axis <= hi))
            if in_range:
                rows_html += rec_row("Spin Axis", f"{spin_axis}°", f"Ideal for {GROUP_TO_DISPLAY[pitch_group]} ({bsp['desc']}) — full break potential unlocked", "badge-gain")
            else:
                gap = min(abs(spin_axis - lo), abs(spin_axis - hi))
                rows_html += rec_row("Spin Axis", f"{spin_axis}°", f"{gap:.0f}° off ideal — target {bsp['desc']} to maximize movement", "badge-warn")

        # IVB recommendation
        if bi:
            if abs(ivb) >= abs(bi['elite']):
                rows_html += rec_row("Vert Break", f"{ivb:.1f} in", f"Elite depth vs. MLB (elite ≥{bi['elite']} in) — tunnel high-low aggressively", "badge-gain")
            elif abs(ivb) >= abs(bi['avg']):
                rows_html += rec_row("Vert Break", f"{ivb:.1f} in", f"Above MLB avg (≥{bi['avg']} in) — solid movement profile", "badge-gain")
            else:
                rows_html += rec_row("Vert Break", f"{ivb:.1f} in", f"Below MLB avg ({bi['avg']} in) — focus on spin axis and efficiency", "badge-warn")

        # HB recommendation
        if bh:
            h_val = abs(hb)
            h_bench = bh.get('elite', bh.get('avg', 0))
            h_avg   = bh.get('avg', 0)
            if h_val >= abs(h_bench):
                rows_html += rec_row("Horiz Break", f"{hb:.1f} in", f"Elite horizontal movement vs. MLB (elite ≥{h_bench} in)", "badge-gain")
            elif h_val >= abs(h_avg):
                rows_html += rec_row("Horiz Break", f"{hb:.1f} in", f"Above MLB avg horizontal movement (avg {h_avg} in)", "badge-info")
            else:
                rows_html += rec_row("Horiz Break", f"{hb:.1f} in", f"Below MLB avg horizontal movement (avg {h_avg} in)", "badge-warn")

        # Arm angle slot context
        if arm_angle <= 25:
            rows_html += rec_row("Arm Slot", f"{arm_angle}°", "Low/sidearm — horizontal movement is your power. Lean into run and ride.", "badge-info")
        elif arm_angle <= 45:
            rows_html += rec_row("Arm Slot", f"{arm_angle}°", "Low three-quarter — balanced horizontal and vertical break profiles.", "badge-info")
        elif arm_angle <= 65:
            rows_html += rec_row("Arm Slot", f"{arm_angle}°", "Three-quarter to high — vertical movement primary, work top of zone.", "badge-info")
        else:
            rows_html += rec_row("Arm Slot", f"{arm_angle}°", "High/over-top — maximum backspin efficiency needed. Target VAA near -4°.", "badge-info")

        # Extension
        if ext >= 7.0:
            rows_html += rec_row("Extension", f"{ext:.1f} ft", "Above MLB avg — hitter reaction time shortened ≈15ms. Attack up in zone.", "badge-gain")
        elif ext < 6.0:
            rows_html += rec_row("Extension", f"{ext:.1f} ft", "Below MLB avg — tunneling and deception become more critical.", "badge-warn")
        else:
            rows_html += rec_row("Extension", f"{ext:.1f} ft", "MLB average range (6.0–7.0 ft).", "badge-info")

        # Wrist mechanics compatibility
        wc_label, wc_col, wc_note = WRIST_COMPAT[wrist_type][pitch_group]
        wc_badge = 'badge-gain' if wc_col == '#3dcc7c' else 'badge-warn' if wc_col == '#f05050' else 'badge-info'
        rows_html += rec_row("Wrist Fit", wc_label, wc_note, wc_badge)

        st.markdown(f"""
        <div style="background:#141820;border:1px solid #2a3348;border-left:4px solid #3a9dff;border-radius:6px;padding:1.1rem 1.4rem;">
            <table style="width:100%;border-collapse:collapse;">{rows_html}</table>
        </div>
        """, unsafe_allow_html=True)

    with plot_col:
        st.markdown('<div class="section-header">Movement vs. MLB</div>', unsafe_allow_html=True)

        bi_avg   = bi.get('avg', 0)   if bi else 0
        bi_elite = bi.get('elite', 0) if bi else 0
        bh_avg   = bh.get('avg', 0)   if bh else 0
        bh_elite = bh.get('elite', 0) if bh else 0

        fig_mv = go.Figure()

        # MLB avg zone (circle approximation)
        theta = np.linspace(0, 2*np.pi, 60)
        fig_mv.add_trace(go.Scatter(
            x=bh_avg + 3*np.cos(theta), y=bi_avg + 3*np.sin(theta),
            mode='lines', line=dict(color='rgba(138,148,170,0.3)', width=1, dash='dot'),
            name='MLB Avg Zone', showlegend=True,
            hoverinfo='skip'
        ))
        # MLB elite zone
        fig_mv.add_trace(go.Scatter(
            x=bh_elite + 2*np.cos(theta), y=bi_elite + 2*np.sin(theta),
            mode='lines', line=dict(color='rgba(240,192,64,0.35)', width=1, dash='dash'),
            name='MLB Elite Zone', showlegend=True,
            hoverinfo='skip'
        ))

        # Crosshairs
        fig_mv.add_hline(y=0, line_color='#2a3348', line_dash='dot', line_width=1)
        fig_mv.add_vline(x=0, line_color='#2a3348', line_dash='dot', line_width=1)

        # Pitch dot
        fig_mv.add_trace(go.Scatter(
            x=[hb], y=[ivb],
            mode='markers+text',
            marker=dict(size=18, color='#f0c040', line=dict(color='#0c0f14', width=2)),
            text=[GROUP_TO_DISPLAY[pitch_group]],
            textposition='top center',
            textfont=dict(color='#f0c040', size=11, family='Barlow Condensed'),
            name='Your Pitch',
            showlegend=True,
        ))

        lim = max(28, abs(hb)+5, abs(ivb)+5, abs(bh_elite)+5, abs(bi_elite)+5)
        fig_mv.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#141820',
            font=dict(color='#e8eaf0', family='Barlow'),
            xaxis=dict(range=[-lim, lim], gridcolor='#1c2230', zeroline=False,
                       title='Horizontal Break (in)', tickfont=dict(size=10)),
            yaxis=dict(range=[-lim, lim], gridcolor='#1c2230', zeroline=False,
                       title='Induced Vertical Break (in)', tickfont=dict(size=10)),
            legend=dict(bgcolor='#141820', bordercolor='#2a3348', borderwidth=1,
                        font=dict(size=10), x=0.01, y=0.99),
            margin=dict(l=0, r=0, t=10, b=0),
            height=420,
        )
        st.plotly_chart(fig_mv, use_container_width=True)

        # MLB benchmark table below plot
        st.markdown('<div class="section-header" style="margin-top:0.5rem;">MLB Benchmarks</div>', unsafe_allow_html=True)
        if bi and bh and bv:
            bm_data = {
                'Metric': ['Velocity', 'IVB', 'HB', 'Active Spin'],
                'Yours':  [f"{velo:.1f} mph", f"{ivb:.1f} in", f"{hb:.1f} in", f"{spin_eff}%"],
                'MLB Avg': [f"{bv.get('avg','—')} mph", f"{bi.get('avg','—')} in",
                            f"{bh.get('avg','—')} in", "~80%"],
                'MLB Elite': [f"{bv.get('elite','—')} mph", f"{bi.get('elite','—')} in",
                              f"{bh.get('elite','—')} in", "~92%"],
            }
            bm_df = pd.DataFrame(bm_data)
            fig_bm = go.Figure(data=[go.Table(
                header=dict(values=list(bm_df.columns), fill_color='#1c2230',
                            font=dict(color='#f0c040', size=11, family='Barlow Condensed'),
                            line_color='#2a3348', align='left', height=30),
                cells=dict(values=[bm_df[c] for c in bm_df.columns],
                           fill_color=[['#141820' if i%2==0 else '#111418' for i in range(len(bm_df))]],
                           font=dict(color='#e8eaf0', size=11, family='Barlow'),
                           line_color='#2a3348', align='left', height=26)
            )])
            fig_bm.update_layout(paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0),
                                 height=max(120, len(bm_df)*28+50))
            st.plotly_chart(fig_bm, use_container_width=True)

    # ---- Complement Pitch Suggestion ----
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Complement Pitch Suggestion</div>', unsafe_allow_html=True)

    comp_groups = _get_complements(pitch_group)
    comp_cols = st.columns(len(comp_groups)) if comp_groups else []

    for idx, comp_group in enumerate(comp_groups):
        bv_c  = VELO_BENCHMARKS.get(comp_group, {})
        bi_c  = IVB_BENCHMARKS.get(comp_group, {})
        bh_c  = HB_BENCHMARKS.get(comp_group, {})
        bsp_c = PITCH_SPIN_AXIS_IDEAL.get(comp_group, {})
        why   = _COMPLEMENT_WHY.get((pitch_group, comp_group),
                                    f"Creates movement contrast with your {GROUP_TO_DISPLAY[pitch_group]}.")

        wc_label_c, wc_col_c, _ = WRIST_COMPAT[wrist_type][comp_group]
        axis_str = bsp_c.get('desc', '—') if bsp_c else '—'
        ivb_str  = f"{bi_c.get('avg','?')} (avg) · {bi_c.get('elite','?')} in (elite)" if bi_c else '—'
        hb_str   = f"{bh_c.get('avg','?')} (avg) · {bh_c.get('elite','?')} in (elite)" if bh_c else '—'
        velo_str = f"~{bv_c.get('avg','?')} (avg) · ~{bv_c.get('elite','?')} mph (elite)" if bv_c else '—'

        with comp_cols[idx]:
            st.markdown(f"""
            <div style="background:#141820;border:1px solid #2a3348;border-top:3px solid {wc_col_c};border-radius:6px;padding:1.2rem 1.4rem;height:100%;">
                <div style="display:flex;align-items:baseline;gap:0.7rem;margin-bottom:0.4rem;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:800;color:#e8eaf0;letter-spacing:0.04em;">{GROUP_TO_DISPLAY[comp_group]}</div>
                    <span style="color:{wc_col_c};font-size:0.75rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;">{wc_label_c} · {wrist_type}</span>
                </div>
                <div style="color:#8a94aa;font-size:0.82rem;line-height:1.55;margin-bottom:0.9rem;">{why}</div>
                <table style="width:100%;border-collapse:collapse;">
                    <tr>
                        <td style="padding:3px 14px 3px 0;color:#505a70;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;white-space:nowrap;">Spin Axis</td>
                        <td style="padding:3px 0;color:#e8eaf0;font-size:0.8rem;">{axis_str}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px 14px 3px 0;color:#505a70;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;white-space:nowrap;">IVB</td>
                        <td style="padding:3px 0;color:#e8eaf0;font-size:0.8rem;">{ivb_str}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px 14px 3px 0;color:#505a70;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;white-space:nowrap;">HB</td>
                        <td style="padding:3px 0;color:#e8eaf0;font-size:0.8rem;">{hb_str}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px 14px 3px 0;color:#505a70;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;white-space:nowrap;">Velocity</td>
                        <td style="padding:3px 0;color:#e8eaf0;font-size:0.8rem;">{velo_str}</td>
                    </tr>
                </table>
            </div>""", unsafe_allow_html=True)
