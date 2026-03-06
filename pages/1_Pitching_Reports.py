import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from datetime import date
from pybaseball import statcast
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (
    THEME_CSS, load_stuff_model, add_engineered_features,
    standardize_to_model_input, run_stuff_plus, create_summary,
    confidence_ellipse, get_games_for_date,
)

st.set_page_config(page_title="Pitching Reports · AlpAnalytics", layout="wide", page_icon="📊", initial_sidebar_state="expanded")
st.markdown(THEME_CSS, unsafe_allow_html=True)

model_dict = load_stuff_model()

# ---- Sidebar ----
st.sidebar.markdown('<div class="sidebar-logo">⚾ AlpAnalytics</div>', unsafe_allow_html=True)
if model_dict:
    st.sidebar.success(f"✅ Model loaded · {len(model_dict['features'])} features")
else:
    st.sidebar.warning("⚠️ model.pkl not found — Stuff+ will be placeholder")

st.sidebar.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("", ["MLB Game", "Upload CSV"], label_visibility="collapsed")

if "report_data" not in st.session_state:
    st.session_state.report_data = pd.DataFrame()

# ---- Header ----
st.markdown("""
<div class="app-header">
    <p class="app-title">Pitching Reports</p>
    <p class="app-subtitle">AlpAnalytics · Game Analysis · Stuff+ Grading</p>
</div>
""", unsafe_allow_html=True)

# ---- Data Loading ----
if mode == "MLB Game":
    st.sidebar.markdown('<div class="sidebar-section">Select Game</div>', unsafe_allow_html=True)
    game_date = st.sidebar.date_input("Date", value=date(2025, 7, 15))
    date_str  = game_date.strftime("%Y-%m-%d")
    games_df  = get_games_for_date(date_str)

    if not games_df.empty:
        game_choice = st.sidebar.selectbox(
            "Game", options=games_df.index,
            format_func=lambda i: f"{games_df.loc[i,'away']} @ {games_df.loc[i,'home']}"
        )
        selected_game_pk = games_df.loc[game_choice, "game_pk"]

        if st.sidebar.button("🚀 Load Game Data"):
            with st.spinner("Pulling Statcast data..."):
                raw_df = statcast(start_dt=date_str, end_dt=date_str)
                if raw_df is None or raw_df.empty:
                    st.error("No Statcast data available.")
                    st.stop()
                game_df = raw_df[raw_df["game_pk"] == selected_game_pk].copy()
                if game_df.empty:
                    st.error("No pitch data for this game.")
                    st.stop()
                raw_needed = [
                    "pitcher", "player_name", "pitch_type", "release_speed", "release_spin_rate",
                    "pfx_x", "pfx_z", "release_extension", "release_pos_z", "release_pos_x",
                    "spin_axis", "vx0", "vy0", "vz0", "ax", "ay", "az", "p_throws",
                    "plate_x", "plate_z", "description", "arm_angle",
                    "events", "at_bat_number", "bat_score", "post_bat_score",
                ]
                needed_cols = list(set((model_dict["features"] if model_dict else []) + raw_needed))
                game_df = game_df[[c for c in needed_cols if c in game_df.columns]]
                processed = standardize_to_model_input(game_df, "statcast")
                processed = add_engineered_features(processed)
                processed = run_stuff_plus(processed, model_dict)
                processed["hb"]  = processed["pfx_x"]
                processed["ivb"] = processed["pfx_z"]
                st.session_state.report_data = processed
            st.success(f"✅ Loaded {len(processed):,} pitches")
    else:
        st.sidebar.info("No games found for this date.")

elif mode == "Upload CSV":
    st.sidebar.markdown('<div class="sidebar-section">Upload File</div>', unsafe_allow_html=True)
    uploaded = st.sidebar.file_uploader("Trackman or Statcast CSV", type="csv")
    if uploaded:
        raw_df = pd.read_csv(uploaded)
        if "PitcherName" in raw_df.columns:
            raw_df.rename(columns={"PitcherName": "player_name"}, inplace=True)
        if st.sidebar.button("🚀 Run Analysis"):
            with st.spinner("Processing..."):
                processed = standardize_to_model_input(raw_df, "trackman")
                processed = add_engineered_features(processed)
                processed = run_stuff_plus(processed, model_dict)
                processed["hb"]  = processed["pfx_x"]
                processed["ivb"] = processed["pfx_z"]
                st.session_state.report_data = processed
            st.success(f"✅ Stuff+ calculated on {len(processed):,} pitches")

# ---- Game line helper ----
def _game_line(df):
    """Compute IP, PA, R, H, K, BB, HBP, HR, Strike%, Whiffs from Statcast pitch-level data."""
    _STRIKE_DESCS = {
        'called_strike', 'swinging_strike', 'swinging_strike_blocked',
        'foul', 'foul_tip', 'foul_bunt', 'foul_pitchout', 'missed_bunt', 'bunt_foul_tip',
    }
    _WHIFF_DESCS = {'swinging_strike', 'swinging_strike_blocked'}
    _SINGLE_OUT = {
        'strikeout', 'field_out', 'force_out', 'fielders_choice_out', 'fielders_choice',
        'sac_fly', 'sac_bunt', 'caught_stealing_2b', 'caught_stealing_3b',
        'caught_stealing_home', 'batter_interference', 'fan_interference', 'other_out',
    }
    _DOUBLE_OUT = {
        'grounded_into_double_play', 'double_play', 'strikeout_double_play', 'sac_fly_double_play',
    }
    _TRIPLE_OUT = {'triple_play'}

    if 'events' in df.columns:
        ev = df['events']
        terminal = df[ev.notna() & ev.astype(str).str.strip().isin(
            set(ev.dropna().astype(str).unique()) - {'', 'nan', 'None'}
        )]
        # Simpler: terminal = rows where events is a real string
        terminal = df[ev.apply(lambda x: pd.notna(x) and str(x).strip() not in ('', 'nan', 'None'))]
        evs = terminal['events'].astype(str)

        pa  = len(terminal)
        h   = evs.isin(['single', 'double', 'triple', 'home_run']).sum()
        hr  = (evs == 'home_run').sum()
        k   = evs.isin(['strikeout', 'strikeout_double_play']).sum()
        bb  = evs.isin(['walk', 'intent_walk']).sum()
        hbp = (evs == 'hit_by_pitch').sum()

        s_outs = evs.isin(_SINGLE_OUT).sum()
        d_outs = evs.isin(_DOUBLE_OUT).sum()
        t_outs = evs.isin(_TRIPLE_OUT).sum()
        total_outs = int(s_outs + d_outs * 2 + t_outs * 3)
        ip = f"{total_outs // 3}.{total_outs % 3}"

        r = '—'
        if 'bat_score' in df.columns and 'post_bat_score' in df.columns:
            try:
                runs = (pd.to_numeric(terminal['post_bat_score'], errors='coerce') -
                        pd.to_numeric(terminal['bat_score'],      errors='coerce')).clip(lower=0).sum()
                r = int(runs)
            except Exception:
                pass
    else:
        ip = '—'; pa = '—'; r = '—'; h = '—'; k = '—'; bb = '—'; hbp = '—'; hr = '—'

    if 'description' in df.columns:
        descs = df['description'].fillna('')
        n = len(descs)
        str_pct = f"{descs.isin(_STRIKE_DESCS).sum() / n * 100:.1f}" if n else '—'
        whiffs  = int(descs.isin(_WHIFF_DESCS).sum())
    else:
        str_pct = '—'; whiffs = '—'

    return {'IP': ip, 'PA': pa, 'R': r, 'H': h, 'K': k,
            'BB': bb, 'HBP': hbp, 'HR': hr, 'Strike%': str_pct, 'Whiffs': whiffs}


# ---- Main Display ----
data = st.session_state.report_data

if not data.empty:
    tab1, tab2 = st.tabs(["🎯 Pitcher Report", "💾 Export"])

    # ---- TAB 1: PITCHER REPORT ----
    with tab1:
        pitchers = sorted(data["player_name"].dropna().unique())
        selected_pitcher = st.selectbox("Select Pitcher", pitchers, key='pitcher_report')
        pitcher_df = data[data["player_name"] == selected_pitcher].copy()

        if not pitcher_df.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            metrics = [
                ("PITCHES",     len(pitcher_df), ""),
                ("AVG STUFF+",  f"{pitcher_df['stuff_plus'].mean():.0f}", ""),
                ("AVG VELO",    f"{pitcher_df['release_speed'].mean():.1f}", "mph"),
                ("MAX VELO",    f"{pitcher_df['release_speed'].max():.1f}", "mph"),
                ("PITCH TYPES", pitcher_df['pitch_type'].nunique(), ""),
            ]
            for col, (label, value, unit) in zip([col1,col2,col3,col4,col5], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-sub">{unit}</div>
                    </div>""", unsafe_allow_html=True)

            # ── Game line stat table ──────────────────────────────────────────
            gl = _game_line(pitcher_df)
            _cols = ['IP', 'PA', 'R', 'H', 'K', 'BB', 'HBP', 'HR', 'Strike%', 'Whiffs']
            _header = "".join(
                f"<th style='padding:7px 14px;text-align:center;font-family:\"Barlow Condensed\",sans-serif;"
                f"font-size:0.78rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;"
                f"color:#f0c040;border-bottom:2px solid #f0c040;white-space:nowrap;'>{c}</th>"
                for c in _cols
            )
            _row = "".join(
                f"<td style='padding:8px 14px;text-align:center;font-family:\"Barlow Condensed\",sans-serif;"
                f"font-size:1.15rem;font-weight:700;color:#e8eaf0;'>{gl[c]}</td>"
                for c in _cols
            )
            st.markdown(f"""
            <div style="overflow-x:auto;margin:1rem 0 1.5rem;">
              <table style="border-collapse:collapse;background:#141820;border:1px solid #2a3348;
                            border-radius:8px;overflow:hidden;width:auto;">
                <thead><tr style="background:#1c2230;">{_header}</tr></thead>
                <tbody><tr>{_row}</tr></tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:1.5rem;">Pitch Arsenal Breakdown</div>', unsafe_allow_html=True)
            summary_table = create_summary(pitcher_df)
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(summary_table.columns), fill_color='#1c2230',
                            font=dict(color='#f0c040', size=11, family='Barlow Condensed'),
                            line_color='#2a3348', align='left', height=34),
                cells=dict(values=[summary_table[c] for c in summary_table.columns],
                           fill_color=[['#141820' if i%2==0 else '#111418' for i in range(len(summary_table))]],
                           font=dict(color='#e8eaf0', size=11, family='Barlow'),
                           line_color='#2a3348', align='left', height=30)
            )])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0),
                              height=max(150, len(summary_table)*32+60))
            st.plotly_chart(fig, use_container_width=True)

            hand = pitcher_df['p_throws'].iloc[0]
            arm_angle_val = round(pitcher_df['arm_angle'].median()) if 'arm_angle' in pitcher_df.columns and not pitcher_df['arm_angle'].isna().all() else 90

            fig_charts, axes = plt.subplots(1, 2, figsize=(14, 7))
            fig_charts.patch.set_facecolor('#0c0f14')
            for ax in axes:
                ax.set_facecolor('#141820')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2a3348')
                ax.tick_params(colors='#8a94aa')
                ax.xaxis.label.set_color('#8a94aa')
                ax.yaxis.label.set_color('#8a94aa')
                ax.title.set_color('#f0c040')
                ax.grid(True, linestyle='--', alpha=0.2, color='#2a3348')

            if all(c in pitcher_df.columns for c in ['plate_x', 'plate_z']):
                loc_data = pitcher_df.copy()
                if hand == 'L':
                    loc_data['plate_x'] = -loc_data['plate_x']
                palette = sns.color_palette("husl", n_colors=len(pitcher_df['pitch_type'].unique()))
                for i, (pt, grp) in enumerate(loc_data.groupby('pitch_type')):
                    axes[0].scatter(grp['plate_x'], grp['plate_z'], alpha=0.7, s=55,
                                    color=palette[i], label=pt, edgecolors='none')
                axes[0].set_title("Pitch Locations — Pitcher's View", fontsize=12, fontweight='bold')
                axes[0].set_xlim(-2.5, 2.5); axes[0].set_ylim(0.5, 4.5)
                axes[0].add_patch(Rectangle((-0.83, 1.5), 1.66, 2.0, linewidth=2, edgecolor='#f0c040', facecolor='none', zorder=5))
                axes[0].legend(fontsize=8, facecolor='#1c2230', edgecolor='#2a3348', labelcolor='#e8eaf0')

            if 'hb' in pitcher_df.columns and 'ivb' in pitcher_df.columns:
                hb_plot = -pitcher_df['hb']  # pitcher's-own-view: right=3B, left=1B
                lim = max(abs(hb_plot.max()), abs(hb_plot.min()), abs(pitcher_df['ivb'].max()), abs(pitcher_df['ivb'].min())) * 1.2
                palette = sns.color_palette("husl", n_colors=len(pitcher_df['pitch_type'].unique()))
                color_map = dict(zip(pitcher_df['pitch_type'].unique(), palette))
                for pt, grp in pitcher_df.groupby('pitch_type'):
                    hb_g = -grp['hb']  # same sign for all pitchers
                    axes[1].scatter(hb_g, grp['ivb'], alpha=0.75, s=55, color=color_map[pt], label=pt, edgecolors='none')
                    if len(grp) > 2:
                        confidence_ellipse(hb_g.values, grp['ivb'].values, axes[1], facecolor=color_map[pt], alpha=0.15, edgecolor='none')
                axes[1].axhline(0, color='#2a3348', ls='--', alpha=0.7)
                axes[1].axvline(0, color='#2a3348', ls='--', alpha=0.7)
                axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
                axes[1].set_xlabel("← 1B Side · 3B Side →"); axes[1].set_ylabel("Induced Vertical Break (inches)")
                axes[1].set_title(f"Movement Map — Arm Angle: {arm_angle_val}°", fontsize=12, fontweight='bold')
                axes[1].legend(fontsize=8, facecolor='#1c2230', edgecolor='#2a3348', labelcolor='#e8eaf0')
                arm_x = 0.97 if hand == 'R' else 0.03
                axes[1].text(arm_x, 0.96, f"Arm: {arm_angle_val}°", transform=axes[1].transAxes,
                             ha='right' if hand == 'R' else 'left', va='top', fontsize=10, fontweight='bold',
                             color='#0c0f14', bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0c040', alpha=0.95))

            plt.tight_layout()
            st.pyplot(fig_charts)
            plt.close(fig_charts)

    # ---- TAB 2: EXPORT ----
    with tab2:
        st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            Dataset contains <strong>{len(data):,} pitches</strong> with Stuff+ scores and engineered features.
        </div>""", unsafe_allow_html=True)
        st.dataframe(data, use_container_width=True)
        st.download_button("📥 Download Full CSV", data.to_csv(index=False),
                           f"alp_report_{date.today().strftime('%Y%m%d')}.csv", mime="text/csv")

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#505a70;">
        <div style="font-size:4rem;margin-bottom:1rem;">⚾</div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:700;color:#8a94aa;text-transform:uppercase;letter-spacing:0.1em;">
            Ready to Analyze
        </div>
        <div style="margin-top:0.5rem;font-size:0.95rem;">
            Select a data source in the sidebar to load pitch data.
        </div>
    </div>
    """, unsafe_allow_html=True)
