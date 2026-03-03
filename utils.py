"""Shared utilities for AlpAnalytics — model, feature engineering, constants, CSS."""
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ====================== THEME CSS ======================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:    #0c0f14;
    --bg-card:       #141820;
    --bg-surface:    #1c2230;
    --border:        #2a3348;
    --accent-gold:   #f0c040;
    --accent-blue:   #3a9dff;
    --accent-green:  #3dcc7c;
    --accent-red:    #f05050;
    --text-primary:  #e8eaf0;
    --text-secondary:#8a94aa;
    --text-muted:    #505a70;
}

.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Barlow', sans-serif;
    color: var(--text-primary);
}
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
.app-header {
    background: linear-gradient(135deg, #0c0f14 0%, #141820 60%, #1a1f2e 100%);
    border-bottom: 2px solid var(--accent-gold);
    padding: 2rem 0 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(240,192,64,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.app-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent-gold);
    line-height: 1;
    margin: 0;
    text-shadow: 0 0 40px rgba(240,192,64,0.3);
}
.app-subtitle {
    font-family: 'Barlow', sans-serif;
    font-weight: 300;
    font-size: 0.95rem;
    color: var(--text-secondary);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent-gold);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    transition: transform 0.15s, border-color 0.15s;
}
.metric-card:hover { transform: translateY(-2px); border-top-color: var(--accent-blue); }
.metric-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem; font-weight: 700; color: var(--text-primary); line-height: 1;
}
.metric-sub { font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.2rem; }
.section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.3rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--accent-gold);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}
.rec-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-left: 4px solid var(--accent-blue); border-radius: 6px;
    padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.rec-card.high-priority { border-left-color: var(--accent-gold); }
.rec-pitch-name {
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.1rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--accent-gold); margin-bottom: 0.5rem;
}
.badge-gain { background: rgba(61,204,124,0.15); color: var(--accent-green); }
.badge-warn { background: rgba(240,80,80,0.15);  color: var(--accent-red);   }
.badge-info { background: rgba(58,157,255,0.15); color: var(--accent-blue);  }
.stuff-elite { color: #f0c040; font-weight: 700; }
.stuff-great { color: #3dcc7c; font-weight: 700; }
.stuff-avg   { color: #8a94aa; font-weight: 600; }
.stuff-below { color: #f05050; font-weight: 600; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 6px; }
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border); gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-secondary) !important; background: transparent !important;
    border-bottom: 3px solid transparent !important;
    padding: 0.7rem 1.4rem !important; font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-gold) !important;
    border-bottom: 3px solid var(--accent-gold) !important;
}
.sidebar-logo {
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.4rem;
    font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent-gold); padding: 0.5rem 0 1rem;
    border-bottom: 1px solid var(--border); margin-bottom: 1rem;
}
.sidebar-section {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--text-muted); margin: 1.2rem 0 0.5rem;
}
.stButton > button {
    font-family: 'Barlow Condensed', sans-serif !important; font-weight: 700 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    background: var(--accent-gold) !important; color: #0c0f14 !important;
    border: none !important; border-radius: 4px !important;
    padding: 0.55rem 1.8rem !important; transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stSelectbox > div > div,
.stDateInput > div > div > input {
    background: var(--bg-surface) !important;
    border-color: var(--border) !important; color: var(--text-primary) !important;
}
.js-plotly-plot .plotly { background: transparent !important; }
.info-box {
    background: rgba(58,157,255,0.07); border: 1px solid rgba(58,157,255,0.25);
    border-radius: 6px; padding: 1rem 1.2rem; font-size: 0.88rem;
    color: var(--text-secondary); margin-bottom: 1rem;
}
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

/* Hide Player Lookup + Pitch Shape Advisor from sidebar nav */
[data-testid="stSidebarNav"] li:has(a[href*="4_Player"])  { display: none !important; }
[data-testid="stSidebarNav"] a[href*="4_Player"]           { display: none !important; }
[data-testid="stSidebarNav"] li:has(a[href*="2_Pitch_Shape"]) { display: none !important; }
[data-testid="stSidebarNav"] a[href*="2_Pitch_Shape"]         { display: none !important; }
</style>
"""

# ====================== CONSTANTS ======================
PITCH_SPIN_AXIS_IDEAL = {
    'Fastball':  {'topspin_eff_min': 0.85, 'ideal_axis_range': (175, 200), 'desc': "12-1 o'clock (backspin)"},
    'Sinker':    {'topspin_eff_min': 0.75, 'ideal_axis_range': (210, 240), 'desc': "2-3 o'clock"},
    'Cutter':    {'topspin_eff_min': 0.55, 'ideal_axis_range': (300, 330), 'desc': "10-11 o'clock"},
    'Slider':    {'topspin_eff_min': 0.35, 'ideal_axis_range': (30, 70),   'desc': "1-2 o'clock gyro"},
    'Sweeper':   {'topspin_eff_min': 0.40, 'ideal_axis_range': (315, 345), 'desc': "10-11 o'clock"},
    'Curveball': {'topspin_eff_min': 0.80, 'ideal_axis_range': (0, 20),    'desc': "6 o'clock (topspin)"},
    'Changeup':  {'topspin_eff_min': 0.75, 'ideal_axis_range': (180, 220), 'desc': "12-2 o'clock pronated"},
}

VELO_BENCHMARKS = {
    'Fastball':  {'elite': 97.0, 'avg': 93.5, 'below': 91.0},
    'Sinker':    {'elite': 96.0, 'avg': 92.5, 'below': 90.0},
    'Cutter':    {'elite': 92.0, 'avg': 88.5, 'below': 86.0},
    'Slider':    {'elite': 88.0, 'avg': 84.0, 'below': 81.0},
    'Sweeper':   {'elite': 85.0, 'avg': 81.5, 'below': 79.0},
    'Curveball': {'elite': 84.0, 'avg': 79.0, 'below': 76.0},
    'Changeup':  {'elite': 89.0, 'avg': 85.0, 'below': 82.0},
}

IVB_BENCHMARKS = {
    'Fastball':  {'elite': 18, 'avg': 13,  'below': 9},
    'Sinker':    {'elite': 10, 'avg': 6,   'below': 3},
    'Cutter':    {'elite': 5,  'avg': 1,   'below': -3},
    'Slider':    {'elite': -3, 'avg': -7,  'below': -12},
    'Sweeper':   {'elite': 2,  'avg': -1,  'below': -4},
    'Curveball': {'elite': -12,'avg': -7,  'below': -3},
    'Changeup':  {'elite': 12, 'avg': 8,   'below': 4},
}

HB_BENCHMARKS = {
    'Fastball':  {'elite': 10, 'avg': 7,   'below': 4},
    'Sinker':    {'elite': 16, 'avg': 11,  'below': 7},
    'Cutter':    {'elite': -8, 'avg': -5,  'below': -2},
    'Slider':    {'elite': -12,'avg': -8,  'below': -4},
    'Sweeper':   {'elite': -18,'avg': -13, 'below': -8},
    'Curveball': {'elite': -5, 'avg': -2,  'below': 2},
    'Changeup':  {'elite': 12, 'avg': 8,   'below': 4},
}

PITCH_TYPE_TO_GROUP = {
    'FF': 'Fastball', 'FT': 'Fastball', 'FA': 'Fastball',
    'SI': 'Sinker',   'CH': 'Changeup', 'FS': 'Changeup',
    'CU': 'Curveball','KC': 'Curveball','FC': 'Cutter',
    'SL': 'Slider',   'ST': 'Sweeper',  'SV': 'Sweeper',
}

GROUP_TO_DISPLAY = {
    'Fastball': '4-Seam Fastball', 'Sinker': 'Sinker / 2-Seam',
    'Cutter': 'Cutter', 'Slider': 'Slider', 'Sweeper': 'Sweeper',
    'Curveball': 'Curveball', 'Changeup': 'Changeup / Splitter',
}

# ====================== MODEL ======================
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_stuff_model():
    try:
        loaded = joblib.load(MODEL_PATH)
        if not isinstance(loaded, dict):
            return None
        return {
            "nonbip_models": loaded["nonbip_models"],
            "bip_models":    loaded["bip_models"],
            "features":      loaded["features"],
            "bip_rates":     loaded["bip_rates"],
            "norm_stats":    loaded["norm_stats"],
        }
    except Exception as e:
        st.sidebar.warning(f"Model not found: {e}")
        return None

# ====================== FEATURE ENGINEERING ======================
def add_engineered_features(df):
    df = df.copy()
    df['pitch_type_group'] = df['pitch_type'].map(PITCH_TYPE_TO_GROUP).fillna('Other')

    is_left = df['p_throws'] == 'L'
    df['pfx_x_adj'] = df['pfx_x'] * np.where(is_left, -1, 1)
    df['release_pos_x_adj'] = df['release_pos_x'] * np.where(is_left, -1, 1)
    df['pfx_z_adj'] = df['pfx_z']

    for col, default in [('vy0', -130), ('ay', 25), ('vz0', -10), ('az', -32), ('vx0', 0), ('ax', 0)]:
        df[col] = df[col].fillna(default) if col in df.columns else default

    y0, yf = 50.0, 17 / 12.0
    discriminant = df['vy0'] ** 2 - 2 * df['ay'] * (y0 - yf)
    valid = discriminant > 0
    t = np.full(len(df), np.nan)
    t[valid] = (-df.loc[valid, 'vy0'] - np.sqrt(discriminant[valid])) / df.loc[valid, 'ay']

    vz_f = df['vz0'] + df['az'] * t
    df['vaa'] = np.where(valid, -np.degrees(np.arctan(vz_f / -df['vy0'])), np.nan)
    vx_f = df['vx0'] + df['ax'] * t
    df['horizontal_approach_angle'] = np.where(valid, -np.degrees(np.arctan(vx_f / -df['vy0'])), np.nan)

    df['spin_axis_sin'] = np.sin(np.deg2rad(df['spin_axis'].fillna(0)))
    df['spin_axis_cos'] = np.cos(np.deg2rad(df['spin_axis'].fillna(0)))

    A, B, r, K, PI = 0.336, 6.041, 0.12, 0.005, np.pi
    total_break_ft = np.sqrt((df['pfx_x'] / 12) ** 2 + (df['pfx_z'] / 12) ** 2)
    a_M = np.where(t > 0, 2 * total_break_ft / t ** 2, 0)
    v_mean = -df['vy0'] - 0.5 * df['ay'] * t
    v_mean_sq = v_mean ** 2
    C_L = np.where(v_mean_sq > 0, a_M / (K * v_mean_sq), 0)
    C_L = np.clip(C_L, 0, A - 1e-6)
    S = np.where(C_L > 0, -(1 / B) * np.log(1 - C_L / A), 0)
    omega_T_rpm = (S * v_mean / r) * (60 / (2 * PI))

    valid_spin = pd.notna(df['release_spin_rate']) & (df['release_spin_rate'] > 0)
    df['spin_efficiency'] = np.where(valid_spin, omega_T_rpm / df['release_spin_rate'], np.nan)
    df['spin_efficiency'] = df['spin_efficiency'].fillna(df['spin_efficiency'].median())

    df['active_spin'] = df['release_spin_rate'] * df['spin_efficiency']
    df['gyro_spin']   = df['release_spin_rate'] * (1 - df['spin_efficiency'])
    for col in ['active_spin', 'gyro_spin']:
        df[col] = df[col].fillna(df[col].median())

    df['adjusted_hhaa'] = np.nan
    if 'plate_x' in df.columns:
        for (ptype, hand), g in df.groupby(['pitch_type_group', 'p_throws']):
            mask = (df['pitch_type_group'] == ptype) & (df['p_throws'] == hand)
            if len(g) < 5:
                df.loc[mask, 'adjusted_hhaa'] = g['horizontal_approach_angle']
                continue
            X_reg = g[['release_pos_x_adj', 'plate_x']].fillna(0)
            y_reg = g['horizontal_approach_angle']
            reg = LinearRegression().fit(X_reg, y_reg)
            df.loc[mask, 'adjusted_hhaa'] = y_reg - reg.predict(X_reg)

    return df


def standardize_to_model_input(df, source="statcast"):
    df = df.copy()
    if source == "trackman":
        mapping = {
            "RelSpeed": "release_speed", "SpinRate": "release_spin_rate",
            "InducedVertBreak": "pfx_z", "HorzBreak": "pfx_x",
            "Extension": "release_extension", "RelHeight": "release_pos_z",
            "ReleaseHeight": "release_pos_z", "RelSide": "release_pos_x",
            "ReleaseSide": "release_pos_x", "SpinAxis": "spin_axis",
            "TaggedPitchType": "pitch_type", "PitchType": "pitch_type",
            "PitcherName": "player_name", "PitcherThrows": "p_throws", "Throws": "p_throws",
        }
        df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)

    for col, default in [("player_name", "Unknown"), ("pitch_type", "UNK"), ("p_throws", "R")]:
        if col not in df.columns:
            df[col] = default

    if source == "statcast":
        df["pfx_x"] = df["pfx_x"] * 12
        df["pfx_z"] = df["pfx_z"] * 12

    return df


def run_stuff_plus(df, model_dict):
    if model_dict is None:
        df = df.copy()
        df["stuff_plus"] = 100.0
        return df
    df = df.copy()
    feature_cols  = model_dict["features"]
    nonbip_models = model_dict["nonbip_models"]
    bip_models    = model_dict["bip_models"]
    bip_rates     = model_dict["bip_rates"]
    norm_stats    = model_dict["norm_stats"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df["stuff_plus"] = 100.0
        return df

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).reindex(columns=feature_cols, fill_value=0)
    df["pred_rv"] = np.nan

    for (ptype, hand) in set(nonbip_models.keys()) | set(bip_models.keys()):
        mask = (df["pitch_type_group"] == ptype) & (df["p_throws"] == hand)
        if not mask.any():
            continue
        bip_rate    = bip_rates.get((ptype, hand), 0.18)
        nonbip_pred = np.zeros(mask.sum())
        bip_pred    = np.zeros(mask.sum())
        if (ptype, hand) in nonbip_models:
            try:
                nonbip_pred = nonbip_models[(ptype, hand)].predict(X[mask])
            except Exception:
                pass
        if (ptype, hand) in bip_models:
            try:
                bip_pred = bip_models[(ptype, hand)].predict(X[mask])
            except Exception:
                pass
        df.loc[mask, "pred_rv"] = (1 - bip_rate) * nonbip_pred + bip_rate * bip_pred

    df["stuff_plus"] = np.nan
    for (ptype, hand), stats in norm_stats.items():
        mask = (df["pitch_type_group"] == ptype) & (df["p_throws"] == hand)
        if not mask.any():
            continue
        rv_mean = stats["rv_mean"]
        rv_std  = stats["rv_std"] if stats["rv_std"] > 0 else 0.001
        df.loc[mask, "stuff_plus"] = 100 - 15 * ((df.loc[mask, "pred_rv"] - rv_mean) / rv_std)

    df["stuff_plus"] = df["stuff_plus"].clip(50, 160).fillna(100.0)
    return df

# ====================== RECOMMENDATION ENGINE ======================
def get_stuff_color_class(val):
    if val >= 130:   return "stuff-elite"
    elif val >= 110: return "stuff-great"
    elif val >= 90:  return "stuff-avg"
    else:            return "stuff-below"


def generate_arsenal_recommendations(pitcher_df):
    hand      = pitcher_df['p_throws'].iloc[0] if not pitcher_df.empty else 'R'
    arm_angle = pitcher_df['arm_angle'].median() if 'arm_angle' in pitcher_df.columns else 45.0
    if pd.isna(arm_angle):
        arm_angle = 45.0

    recs = []
    for pt, grp in pitcher_df.groupby('pitch_type'):
        ptype_group = grp['pitch_type_group'].iloc[0] if 'pitch_type_group' in grp.columns else 'Other'
        if ptype_group == 'Other' or len(grp) < 3:
            continue

        rec = {
            'pitch_type': pt, 'pitch_group': ptype_group,
            'count': len(grp), 'suggestions': [], 'priority': 'normal',
        }

        avg_velo      = grp['release_speed'].mean()
        avg_spin_axis = grp['spin_axis'].mean() if 'spin_axis' in grp.columns else np.nan
        avg_spin_eff  = grp['spin_efficiency'].mean() if 'spin_efficiency' in grp.columns else np.nan
        avg_ivb       = grp['pfx_z_adj'].mean() if 'pfx_z_adj' in grp.columns else grp['ivb'].mean() if 'ivb' in grp.columns else np.nan
        avg_stuff     = grp['stuff_plus'].mean() if 'stuff_plus' in grp.columns else 100.0
        avg_ext       = grp['release_extension'].mean() if 'release_extension' in grp.columns else np.nan
        avg_spin      = grp['release_spin_rate'].mean() if 'release_spin_rate' in grp.columns else np.nan

        rec['stuff_plus'] = avg_stuff
        rec['avg_velo']   = avg_velo
        rec['avg_spin']   = avg_spin

        bmarks_v   = VELO_BENCHMARKS.get(ptype_group, {})
        bmarks_ivb = IVB_BENCHMARKS.get(ptype_group, {})

        # 1. Velocity
        if bmarks_v:
            if avg_velo >= bmarks_v.get('elite', 999):
                rec['suggestions'].append({'type': 'velo', 'label': 'Velocity', 'value': f"{avg_velo:.1f} mph", 'note': f"Elite tier (≥{bmarks_v['elite']} mph)", 'badge_class': 'badge-gain'})
            elif avg_velo >= bmarks_v.get('avg', 999):
                rec['suggestions'].append({'type': 'velo', 'label': 'Velocity', 'value': f"{avg_velo:.1f} mph", 'note': f"MLB average (≥{bmarks_v['avg']} mph)", 'badge_class': 'badge-info'})
            else:
                rec['priority'] = 'warning'
                rec['suggestions'].append({'type': 'velo', 'label': 'Velocity', 'value': f"{avg_velo:.1f} mph", 'note': f"Below MLB avg — target {bmarks_v['avg']} mph", 'badge_class': 'badge-warn'})

        # 2. Spin Efficiency
        if not pd.isna(avg_spin_eff):
            ideal_info = PITCH_SPIN_AXIS_IDEAL.get(ptype_group, {})
            eff_min = ideal_info.get('topspin_eff_min', 0.6)
            if ptype_group in ['Curveball', 'Slider', 'Sweeper']:
                if avg_spin_eff < 0.3:
                    rec['suggestions'].append({'type': 'spin', 'label': 'Spin Efficiency', 'value': f"{avg_spin_eff:.0%}", 'note': 'Gyro-heavy — limits lateral break', 'badge_class': 'badge-info'})
                elif avg_spin_eff > 0.75:
                    rec['priority'] = 'high'
                    rec['suggestions'].append({'type': 'spin', 'label': 'Spin Efficiency', 'value': f"{avg_spin_eff:.0%}", 'note': 'High — active spin driving break', 'badge_class': 'badge-gain'})
                else:
                    rec['suggestions'].append({'type': 'spin', 'label': 'Spin Efficiency', 'value': f"{avg_spin_eff:.0%}", 'note': 'Mixed — adjust axis for depth vs. sweep', 'badge_class': 'badge-info'})
            else:
                if avg_spin_eff >= eff_min:
                    rec['suggestions'].append({'type': 'spin', 'label': 'Spin Efficiency', 'value': f"{avg_spin_eff:.0%}", 'note': 'High — generating ride/life', 'badge_class': 'badge-gain'})
                else:
                    rec['suggestions'].append({'type': 'spin', 'label': 'Spin Efficiency', 'value': f"{avg_spin_eff:.0%}", 'note': f"Leaking gyro — ideal axis: {ideal_info.get('desc', 'N/A')}", 'badge_class': 'badge-warn'})

        # 3. Spin Axis
        if not pd.isna(avg_spin_axis) and ptype_group in PITCH_SPIN_AXIS_IDEAL:
            ideal_info = PITCH_SPIN_AXIS_IDEAL[ptype_group]
            lo, hi = ideal_info['ideal_axis_range']
            in_range = lo <= avg_spin_axis <= hi or (lo > hi and (avg_spin_axis >= lo or avg_spin_axis <= hi))
            if in_range:
                rec['suggestions'].append({'type': 'axis', 'label': 'Spin Axis', 'value': f"{avg_spin_axis:.0f}°", 'note': f"Ideal — {ideal_info['desc']}", 'badge_class': 'badge-gain'})
            else:
                gap = min(abs(avg_spin_axis - lo), abs(avg_spin_axis - hi))
                rec['suggestions'].append({'type': 'axis', 'label': 'Spin Axis', 'value': f"{avg_spin_axis:.0f}°", 'note': f"{gap:.0f}° off ideal ({ideal_info['desc']})", 'badge_class': 'badge-warn'})

        # 4. IVB
        if not pd.isna(avg_ivb) and bmarks_ivb:
            if ptype_group == 'Fastball':
                if avg_ivb >= bmarks_ivb.get('elite', 999):
                    rec['priority'] = 'high'
                    rec['suggestions'].append({'type': 'shape', 'label': 'IVB', 'value': f"{avg_ivb:.1f} in", 'note': f"Elite ride (MLB elite ≥{bmarks_ivb['elite']} in)", 'badge_class': 'badge-gain'})
                elif avg_ivb >= bmarks_ivb.get('avg', 999):
                    rec['suggestions'].append({'type': 'shape', 'label': 'IVB', 'value': f"{avg_ivb:.1f} in", 'note': f"Above MLB avg (≥{bmarks_ivb['avg']} in)", 'badge_class': 'badge-gain'})
                else:
                    rec['suggestions'].append({'type': 'shape', 'label': 'IVB', 'value': f"{avg_ivb:.1f} in", 'note': f"Below MLB avg — target {bmarks_ivb['avg']} in", 'badge_class': 'badge-warn'})
            elif ptype_group in ['Curveball', 'Slider', 'Sweeper']:
                if abs(avg_ivb) >= abs(bmarks_ivb.get('elite', 0)):
                    rec['suggestions'].append({'type': 'shape', 'label': 'Vert Break', 'value': f"{avg_ivb:.1f} in", 'note': 'Elite depth vs. MLB', 'badge_class': 'badge-gain'})

        # 5. Arm Angle tips (Fastball only)
        if ptype_group == 'Fastball':
            if arm_angle <= 25:
                rec['suggestions'].append({'type': 'arm_angle', 'label': 'Arm Slot', 'value': f"{arm_angle:.0f}°", 'note': 'Low slot — leverage horizontal movement', 'badge_class': 'badge-info'})
            elif arm_angle >= 65:
                rec['suggestions'].append({'type': 'arm_angle', 'label': 'Arm Slot', 'value': f"{arm_angle:.0f}°", 'note': 'High slot — maximize spin efficiency for ride', 'badge_class': 'badge-info'})

        # 6. Extension
        if not pd.isna(avg_ext):
            if avg_ext >= 7.0:
                rec['suggestions'].append({'type': 'extension', 'label': 'Extension', 'value': f"{avg_ext:.1f} ft", 'note': 'Above MLB avg — shortens hitter reaction time', 'badge_class': 'badge-gain'})
            elif avg_ext < 6.0:
                rec['suggestions'].append({'type': 'extension', 'label': 'Extension', 'value': f"{avg_ext:.1f} ft", 'note': 'Below MLB avg — tunneling becomes more critical', 'badge_class': 'badge-warn'})

        if rec['suggestions']:
            recs.append(rec)

    priority_order = {'warning': 0, 'high': 1, 'normal': 2}
    recs.sort(key=lambda x: (priority_order.get(x.get('priority', 'normal'), 2), -x.get('stuff_plus', 100)))
    return recs, hand, arm_angle


def render_recommendations(recs, hand, arm_angle):
    if not recs:
        st.info("Not enough pitch data to generate recommendations.")
        return

    st.markdown(f"""
    <div class="info-box">
        Arm angle: <strong>{arm_angle:.0f}°</strong> &nbsp;·&nbsp;
        Handedness: <strong>{'LHP' if hand == 'L' else 'RHP'}</strong> &nbsp;·&nbsp;
        All benchmarks are MLB-calibrated
    </div>
    """, unsafe_allow_html=True)

    for rec in recs:
        stuff_val   = rec.get('stuff_plus', 100)
        stuff_color = '#f0c040' if stuff_val >= 130 else '#3dcc7c' if stuff_val >= 110 else '#8a94aa' if stuff_val >= 90 else '#f05050'
        border_color = '#f05050' if rec['priority'] == 'warning' else '#f0c040' if rec['priority'] == 'high' else '#3a9dff'

        rows_html = ""
        for s in rec['suggestions']:
            dot = '#3dcc7c' if s['badge_class'] == 'badge-gain' else '#f05050' if s['badge_class'] == 'badge-warn' else '#3a9dff'
            rows_html += f"""
            <tr>
                <td style="padding:6px 12px 6px 0;color:#8a94aa;font-size:0.78rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;white-space:nowrap;">{s['label']}</td>
                <td style="padding:6px 16px 6px 0;color:#e8eaf0;font-size:0.92rem;font-weight:700;white-space:nowrap;">{s['value']}</td>
                <td style="padding:6px 0;color:#8a94aa;font-size:0.82rem;">
                    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{dot};margin-right:6px;vertical-align:middle;"></span>{s['note']}
                </td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#141820;border:1px solid #2a3348;border-left:4px solid {border_color};border-radius:6px;padding:1rem 1.3rem;margin-bottom:0.85rem;">
            <div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:0.75rem;flex-wrap:wrap;">
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.15rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#e8eaf0;">{rec['pitch_type']}</span>
                <span style="font-size:0.8rem;color:#505a70;">{rec['pitch_group']}</span>
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;font-weight:700;color:{stuff_color};">Stuff+ {stuff_val:.0f}</span>
                <span style="font-size:0.8rem;color:#505a70;margin-left:auto;">{rec['avg_velo']:.1f} mph &nbsp;·&nbsp; {rec.get('count',0)} pitches</span>
            </div>
            <table style="width:100%;border-collapse:collapse;">{rows_html}</table>
        </div>
        """, unsafe_allow_html=True)

# ====================== MISC HELPERS ======================
def create_summary(p_data):
    agg_dict = {}
    if 'release_speed' in p_data.columns:
        agg_dict['Avg Velo'] = ('release_speed', 'mean')
        agg_dict['Max Velo'] = ('release_speed', 'max')
    agg_dict['Count'] = ('pitch_type', 'count')
    if 'description' in p_data.columns:
        agg_dict['Strike %'] = ('description', lambda x: round(x.str.contains("strike", case=False).sum() / len(x) * 100, 1))
    if 'hb' in p_data.columns:
        agg_dict['Avg HB'] = ('hb', 'mean')
    if 'ivb' in p_data.columns:
        agg_dict['Avg IVB'] = ('ivb', 'mean')
    if 'release_pos_z' in p_data.columns:
        agg_dict['Rel Ht'] = ('release_pos_z', 'mean')
    if 'stuff_plus' in p_data.columns:
        agg_dict['Stuff+'] = ('stuff_plus', 'mean')

    summary = p_data.groupby("pitch_type").agg(**{k: v for k, v in agg_dict.items() if v[0] in p_data.columns}).reset_index()
    num_cols = [c for c in summary.columns if c != 'pitch_type']
    summary[num_cols] = summary[num_cols].round(1)
    if 'Stuff+' in summary.columns:
        summary['Stuff+'] = summary['Stuff+'].round(0).astype('Int64')
    return summary


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = plt.matplotlib.transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_games_for_date(game_date_str):
    import requests
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={game_date_str}"
        data = requests.get(url, timeout=10).json()
        games = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                games.append({
                    "game_pk": g["gamePk"],
                    "away": g["teams"]["away"]["team"]["name"],
                    "home": g["teams"]["home"]["team"]["name"],
                })
        return pd.DataFrame(games)
    except Exception:
        return pd.DataFrame()
