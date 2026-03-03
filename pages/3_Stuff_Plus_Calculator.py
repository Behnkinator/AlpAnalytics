import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import (
    THEME_CSS, load_stuff_model, add_engineered_features,
    run_stuff_plus, VELO_BENCHMARKS, IVB_BENCHMARKS, HB_BENCHMARKS,
    PITCH_SPIN_AXIS_IDEAL, PITCH_TYPE_TO_GROUP, GROUP_TO_DISPLAY,
)

st.set_page_config(page_title="Stuff+ Calculator · AlpAnalytics", layout="wide", page_icon="🎯", initial_sidebar_state="expanded")
st.markdown(THEME_CSS, unsafe_allow_html=True)

model_dict = load_stuff_model()

st.sidebar.markdown('<div class="sidebar-logo">⚾ AlpAnalytics</div>', unsafe_allow_html=True)
if model_dict:
    st.sidebar.success(f"✅ Model ready · {len(model_dict['features'])} features")
else:
    st.sidebar.error("⚠️ model.pkl not found — cannot score")

st.markdown("""
<div class="app-header">
    <p class="app-title">Stuff+ Calculator</p>
    <p class="app-subtitle">AlpAnalytics · Single-Pitch Grader · MLB Benchmarks</p>
</div>
""", unsafe_allow_html=True)

# ---- Inputs ----
st.markdown('<div class="section-header">Pitch Parameters</div>', unsafe_allow_html=True)

col_pitch, col_hand = st.columns([2, 1])
with col_pitch:
    pitch_group = st.selectbox("Pitch Type", list(GROUP_TO_DISPLAY.keys()),
                               format_func=lambda k: GROUP_TO_DISPLAY[k])
with col_hand:
    handedness = st.radio("Handedness", ["R", "L"], horizontal=True,
                          format_func=lambda x: "RHP" if x == "R" else "LHP")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# Two-column input layout
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("**Release & Velocity**")
    velo      = st.slider("Velocity (mph)",        min_value=60.0,  max_value=105.0, value=93.0, step=0.5)
    spin_rate = st.slider("Spin Rate (rpm)",        min_value=1200,  max_value=3800,  value=2300, step=10)
    spin_axis = st.slider("Spin Axis (°)",          min_value=0,     max_value=359,   value=190,  step=1)
    spin_eff  = st.slider("Spin Efficiency (%)",    min_value=0,     max_value=110,   value=92,   step=1,
                           help="Percentage of spin that generates movement (transverse spin). 100% = all active spin.")

    st.markdown("**Release Point**")
    rel_height = st.slider("Release Height (ft)",   min_value=4.0,   max_value=8.0,   value=6.2,  step=0.05)
    rel_side   = st.slider("Release Side (ft)",     min_value=-4.0,  max_value=4.0,   value=1.5,  step=0.05,
                            help="Distance from center of rubber. Positive = arm side.")
    arm_angle  = int(np.clip(np.degrees(np.arctan2(max(0.0, rel_height - 5.0), max(0.01, abs(rel_side)))), 0, 90))
    st.markdown(f"""
    <div style="background:#1c2230;border:1px solid #2a3348;border-radius:4px;padding:0.5rem 0.9rem;margin-top:0.4rem;display:flex;align-items:center;gap:0.8rem;">
        <span style="color:#8a94aa;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;">Calculated Arm Angle</span>
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:700;color:#f0c040;">{arm_angle}°</span>
        <span style="color:#505a70;font-size:0.75rem;">{'Sidearm' if arm_angle < 15 else 'Low 3/4' if arm_angle < 30 else 'Three-Quarter' if arm_angle < 50 else 'High 3/4' if arm_angle < 65 else 'Over the Top'}</span>
    </div>
    """, unsafe_allow_html=True)

with c_right:
    st.markdown("**Pitch Movement**")
    hb    = st.slider("Horizontal Break (in)",      min_value=-25.0, max_value=25.0,  value=8.0,  step=0.5,
                       help="Arm-side adjusted. Positive = arm-side run, negative = glove-side.")
    ivb   = st.slider("Induced Vertical Break (in)", min_value=-25.0, max_value=25.0, value=14.0, step=0.5,
                       help="Positive = ride (resists gravity), negative = drop.")
    ext   = st.slider("Extension (ft)",             min_value=4.0,   max_value=7.5,   value=6.2,  step=0.05,
                       help="How far in front of the rubber the pitcher releases. MLB avg ≈ 6.2 ft.")

    # Auto-calculate VAA from physics
    # Convention matches add_engineered_features: positive = falling (lower = flatter = elite)
    _v_fts    = velo * 1.46667
    _v_avg    = _v_fts * 0.91
    _t        = 55.0 / _v_avg
    _ivb_ft   = ivb / 12.0
    _az_mag   = 2.0 * _ivb_ft / (_t * _t)
    _az_total = -32.174 + _az_mag
    _vz0      = (2.5 - rel_height - 0.5 * _az_total * _t * _t) / _t
    _vz_plate = _vz0 + _az_total * _t
    # Model convention: VAA is stored as positive degrees (higher = steeper = worse)
    # Display convention (user-facing): negative degrees (e.g. -4.5°)
    vaa_display = float(np.clip(np.degrees(np.arctan2(_vz_plate, _v_fts)), -10, -1))
    vaa_model   = -vaa_display   # always positive for model input
    # Horizontal approach angle (estimated from HB physics, model convention)
    _hb_raw_ft  = (hb / 12.0) if handedness == 'R' else -(hb / 12.0)
    _ax_est     = 2.0 * _hb_raw_ft / (_t * _t)
    _vx_plate   = _ax_est * _t
    haa_model   = float(-np.degrees(np.arctan(_vx_plate / _v_fts)))
    st.markdown(f"""
    <div style="background:#1c2230;border:1px solid #2a3348;border-radius:4px;padding:0.5rem 0.9rem;margin-top:0.4rem;display:flex;align-items:center;gap:0.8rem;">
        <span style="color:#8a94aa;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;">Calculated VAA</span>
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:700;color:#3a9dff;">{vaa_display:.1f}°</span>
        <span style="color:#505a70;font-size:0.75rem;">{'Flat (elite)' if vaa_display >= -4.0 else 'Average' if vaa_display >= -5.5 else 'Steep'}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    calculate = st.button("⚡ Calculate Stuff+", use_container_width=True)

# ---- Compute ----
if calculate:
    if model_dict is None:
        st.error("Model not loaded. Please ensure model.pkl is present.")
        st.stop()

    spin_eff_frac = spin_eff / 100.0
    active_spin   = spin_rate * spin_eff_frac
    gyro_spin     = spin_rate * (1 - spin_eff_frac)
    spin_axis_sin = np.sin(np.deg2rad(spin_axis))
    spin_axis_cos = np.cos(np.deg2rad(spin_axis))

    # Reverse the arm-side sign for LHP to match model convention
    pfx_x_adj      = hb   * (-1 if handedness == 'L' else 1)
    release_x_adj  = rel_side * (-1 if handedness == 'L' else 1)

    row = {
        'pitch_type':               'FF',   # placeholder raw type
        'pitch_type_group':         pitch_group,
        'p_throws':                 handedness,
        'release_speed':            velo,
        'release_spin_rate':        spin_rate,
        'spin_axis':                spin_axis,
        'spin_axis_sin':            spin_axis_sin,
        'spin_axis_cos':            spin_axis_cos,
        'spin_efficiency':          spin_eff_frac,
        'active_spin':              active_spin,
        'gyro_spin':                gyro_spin,
        'pfx_x_adj':                pfx_x_adj,
        'pfx_z_adj':                ivb,
        'pfx_x':                    hb,
        'pfx_z':                    ivb,
        # VAA in model convention: positive = falling (matches add_engineered_features output)
        'vaa':                      vaa_model,
        'horizontal_approach_angle': haa_model,
        'adjusted_hhaa':            0.0,   # residual from regression; 0 = average
        'release_extension':        ext,
        'release_pos_z':            rel_height,
        'release_pos_x_adj':        release_x_adj,
        'release_pos_x':            rel_side,
        'arm_angle':                arm_angle,
        'player_name':              'Input',
    }
    df_input = pd.DataFrame([row])
    df_scored = run_stuff_plus(df_input, model_dict)
    score = float(df_scored['stuff_plus'].iloc[0])

    # ---- Score display ----
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)

    grade_label = "Elite"      if score >= 130 else \
                  "Above Avg"  if score >= 110 else \
                  "Average"    if score >= 90  else "Below Avg"
    score_color = '#f0c040'    if score >= 130 else \
                  '#3dcc7c'    if score >= 110 else \
                  '#8a94aa'    if score >= 90  else '#f05050'

    r1, r2, r3 = st.columns([1, 1.5, 1])
    with r1:
        st.markdown(f"""
        <div style="background:#141820;border:1px solid #2a3348;border-top:4px solid {score_color};
                    border-radius:8px;padding:2rem 1.5rem;text-align:center;">
            <div style="color:#8a94aa;font-size:0.72rem;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">Stuff+</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:4rem;font-weight:800;color:{score_color};line-height:1;">{score:.0f}</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:700;color:{score_color};letter-spacing:0.1em;text-transform:uppercase;margin-top:0.5rem;">{grade_label}</div>
            <div style="color:#505a70;font-size:0.78rem;margin-top:0.4rem;">{GROUP_TO_DISPLAY[pitch_group]} · {'RHP' if handedness=='R' else 'LHP'}</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(color=score_color, family='Barlow Condensed', size=42)),
            gauge=dict(
                axis=dict(range=[50, 160], tickwidth=1, tickcolor='#2a3348',
                          tickfont=dict(color='#8a94aa', size=10)),
                bar=dict(color=score_color, thickness=0.25),
                bgcolor='#141820',
                borderwidth=0,
                steps=[
                    dict(range=[50, 90],   color='rgba(240,80,80,0.15)'),
                    dict(range=[90, 110],  color='rgba(138,148,170,0.12)'),
                    dict(range=[110, 130], color='rgba(61,204,124,0.15)'),
                    dict(range=[130, 160], color='rgba(240,192,64,0.18)'),
                ],
                threshold=dict(line=dict(color='#f0c040', width=2), thickness=0.75, value=100)
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', height=240,
            margin=dict(l=20, r=20, t=30, b=10),
            font=dict(color='#e8eaf0', family='Barlow')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with r3:
        # Context vs benchmarks
        bv  = VELO_BENCHMARKS.get(pitch_group, {})
        bi  = IVB_BENCHMARKS.get(pitch_group, {})
        bh  = HB_BENCHMARKS.get(pitch_group, {})
        bsp = PITCH_SPIN_AXIS_IDEAL.get(pitch_group, {})

        def tier(val, elite, avg):
            if val >= elite: return "Elite", "#3dcc7c"
            if val >= avg:   return "MLB Avg", "#8a94aa"
            return "Below Avg", "#f05050"

        rows = []
        if bv:
            t, c = tier(velo, bv['elite'], bv['avg'])
            rows.append(("Velocity", f"{velo:.1f} mph", t, c, f"Avg {bv['avg']} · Elite {bv['elite']}"))
        if bsp:
            lo, hi = bsp['ideal_axis_range']
            in_r = lo <= spin_axis <= hi or (lo > hi and (spin_axis >= lo or spin_axis <= hi))
            rows.append(("Spin Axis", f"{spin_axis}°",
                          "Ideal" if in_r else "Off Ideal",
                          "#3dcc7c" if in_r else "#f05050",
                          bsp['desc']))
        rows.append(("Spin Efficiency", f"{spin_eff}%",
                      "High" if spin_eff >= 88 else "Avg" if spin_eff >= 75 else "Low",
                      "#3dcc7c" if spin_eff >= 88 else "#8a94aa" if spin_eff >= 75 else "#f05050",
                      "MLB range 75–95%"))
        if bi:
            t, c = tier(abs(ivb), abs(bi['elite']), abs(bi['avg']))
            rows.append(("IVB", f"{ivb:.1f} in", t, c, f"Avg {bi['avg']} · Elite {bi['elite']}"))

        table_rows = "".join(f"""
        <tr>
            <td style="padding:5px 10px 5px 0;color:#8a94aa;font-size:0.76rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">{r[0]}</td>
            <td style="padding:5px 10px 5px 0;color:#e8eaf0;font-size:0.85rem;font-weight:700;white-space:nowrap;">{r[1]}</td>
            <td style="padding:5px 0;white-space:nowrap;">
                <span style="color:{r[3]};font-size:0.78rem;font-weight:700;">{r[2]}</span>
                <span style="color:#505a70;font-size:0.72rem;margin-left:6px;">{r[4]}</span>
            </td>
        </tr>""" for r in rows)

        st.markdown(f"""
        <div style="background:#141820;border:1px solid #2a3348;border-radius:6px;padding:1.2rem 1.4rem;height:100%;">
            <div style="font-family:'Barlow Condensed',sans-serif;color:#f0c040;font-size:0.85rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">vs. MLB Benchmarks</div>
            <table style="width:100%;border-collapse:collapse;">{table_rows}</table>
        </div>""", unsafe_allow_html=True)
