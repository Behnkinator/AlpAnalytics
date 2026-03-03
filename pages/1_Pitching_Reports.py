import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    standardize_to_model_input, run_stuff_plus, generate_arsenal_recommendations,
    render_recommendations, get_stuff_color_class, create_summary,
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
                    "plate_x", "plate_z", "description", "arm_angle"
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

# ---- Main Display ----
data = st.session_state.report_data

if not data.empty:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Game Summary", "🎯 Pitcher Report", "🔬 Arsenal Optimizer", "💾 Export"
    ])

    # ---- TAB 1: GAME SUMMARY ----
    with tab1:
        st.markdown('<div class="section-header">Game Summary</div>', unsafe_allow_html=True)
        summary = (
            data.groupby(["player_name"])
            .agg(Pitches=("release_speed","count"), Avg_Stuff=("stuff_plus","mean"),
                 Avg_Velo=("release_speed","mean"), Max_Velo=("release_speed","max"))
            .round(1).reset_index()
            .rename(columns={"player_name":"Pitcher","Avg_Stuff":"Avg Stuff+","Avg_Velo":"Avg Velo","Max_Velo":"Max Velo"})
            .sort_values("Avg Stuff+", ascending=False)
        )
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary.columns), fill_color='#1c2230',
                        font=dict(color='#f0c040', size=12, family='Barlow Condensed'),
                        line_color='#2a3348', align='left', height=36),
            cells=dict(values=[summary[c] for c in summary.columns],
                       fill_color=[['#141820' if i%2==0 else '#111418' for i in range(len(summary))]],
                       font=dict(color='#e8eaf0', size=12, family='Barlow'),
                       line_color='#2a3348', align='left', height=32)
        )])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0,r=0,t=0,b=0), height=max(200, len(summary)*34+70))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Pitch Type Distribution</div>', unsafe_allow_html=True)
        if 'pitch_type' in data.columns:
            pt_counts = data['pitch_type'].value_counts().reset_index()
            pt_counts.columns = ['Pitch Type', 'Count']
            fig_bar = px.bar(pt_counts, x='Pitch Type', y='Count', color='Count',
                             color_continuous_scale=[[0,'#2a3348'],[1,'#f0c040']])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#e8eaf0', family='Barlow'), coloraxis_showscale=False,
                                  xaxis=dict(gridcolor='#2a3348'), yaxis=dict(gridcolor='#2a3348'),
                                  margin=dict(l=0,r=0,t=20,b=0), height=280)
            fig_bar.update_traces(marker_line_width=0)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ---- TAB 2: PITCHER REPORT ----
    with tab2:
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

    # ---- TAB 3: ARSENAL OPTIMIZER ----
    with tab3:
        st.markdown('<div class="section-header">Arsenal Optimizer & Shape Recommendations</div>', unsafe_allow_html=True)
        pitchers_opt = sorted(data["player_name"].dropna().unique())
        selected_opt = st.selectbox("Select Pitcher", pitchers_opt, key='pitcher_opt')
        pitcher_opt_df = data[data["player_name"] == selected_opt].copy()

        if not pitcher_opt_df.empty:
            recs, hand_opt, arm_opt = generate_arsenal_recommendations(pitcher_opt_df)

            c1, c2, c3, c4 = st.columns(4)
            for col, (label, value, sub) in zip([c1,c2,c3,c4], [
                ("ARM ANGLE",   f"{arm_opt:.0f}°", "degrees from horizontal"),
                ("HANDEDNESS",  "LHP" if hand_opt == 'L' else "RHP", "throwing arm"),
                ("PITCH TYPES", pitcher_opt_df['pitch_type'].nunique(), "in arsenal"),
                ("AVG STUFF+",  f"{pitcher_opt_df['stuff_plus'].mean():.0f}", "across all pitches"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            if 'spin_efficiency' in pitcher_opt_df.columns and 'release_spin_rate' in pitcher_opt_df.columns:
                st.markdown('<div class="section-header">Spin Profile by Pitch Type</div>', unsafe_allow_html=True)
                spin_summary = pitcher_opt_df.groupby('pitch_type').agg(
                    Spin_Rate=('release_spin_rate','mean'),
                    Spin_Efficiency=('spin_efficiency','mean'),
                ).reset_index().dropna()

                palette_hex = ['#f0c040','#3a9dff','#3dcc7c','#f05050','#c084fc','#fb923c']
                col_r, col_e = st.columns(2)
                with col_r:
                    fig_spin = go.Figure()
                    for i, (_, row) in enumerate(spin_summary.iterrows()):
                        fig_spin.add_trace(go.Bar(name=row['pitch_type'], x=[row['pitch_type']], y=[row['Spin_Rate']],
                                                  marker_color=palette_hex[i % len(palette_hex)],
                                                  text=f"{row['Spin_Rate']:.0f}", textposition='outside',
                                                  textfont=dict(color='#e8eaf0', size=11)))
                    fig_spin.update_layout(title=dict(text="Spin Rate by Pitch", font=dict(color='#f0c040', size=13)),
                                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#141820',
                                           font=dict(color='#e8eaf0', family='Barlow'), showlegend=False, height=280,
                                           yaxis=dict(gridcolor='#2a3348', title='RPM'),
                                           xaxis=dict(gridcolor='#2a3348'), margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_spin, use_container_width=True)

                with col_e:
                    fig_eff = go.Figure()
                    for i, (_, row) in enumerate(spin_summary.iterrows()):
                        fig_eff.add_trace(go.Bar(name=row['pitch_type'], x=[row['pitch_type']], y=[row['Spin_Efficiency']*100],
                                                 marker_color=palette_hex[i % len(palette_hex)],
                                                 text=f"{row['Spin_Efficiency']:.0%}", textposition='outside',
                                                 textfont=dict(color='#e8eaf0', size=11)))
                    fig_eff.update_layout(title=dict(text="Spin Efficiency %", font=dict(color='#f0c040', size=13)),
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#141820',
                                          font=dict(color='#e8eaf0', family='Barlow'), showlegend=False, height=280,
                                          yaxis=dict(gridcolor='#2a3348', title='%', range=[0,115]),
                                          xaxis=dict(gridcolor='#2a3348'), margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_eff, use_container_width=True)

            st.markdown('<div class="section-header">Shape Recommendations</div>', unsafe_allow_html=True)
            render_recommendations(recs, hand_opt, arm_opt)

            from utils import IVB_BENCHMARKS
            st.markdown('<div class="section-header">Movement vs. MLB Benchmarks</div>', unsafe_allow_html=True)
            pitch_avgs = pitcher_opt_df.groupby('pitch_type').agg(
                IVB=('ivb','mean'), HB=('hb','mean'), pitch_group=('pitch_type_group','first')
            ).reset_index().dropna(subset=['IVB'])

            if not pitch_avgs.empty:
                bm_rows = []
                for _, row in pitch_avgs.iterrows():
                    b = IVB_BENCHMARKS.get(row['pitch_group'], {})
                    bm_rows.append({'Pitch': row['pitch_type'], 'Your IVB': round(row['IVB'],1),
                                    'MLB Avg IVB': b.get('avg','—'), 'MLB Elite IVB': b.get('elite','—')})
                bm_df = pd.DataFrame(bm_rows)
                fig_bm = go.Figure(data=[go.Table(
                    header=dict(values=list(bm_df.columns), fill_color='#1c2230',
                                font=dict(color='#f0c040', size=11, family='Barlow Condensed'),
                                line_color='#2a3348', align='left', height=32),
                    cells=dict(values=[bm_df[c] for c in bm_df.columns],
                               fill_color=[['#141820' if i%2==0 else '#111418' for i in range(len(bm_df))]],
                               font=dict(color='#e8eaf0', size=11, family='Barlow'),
                               line_color='#2a3348', align='left', height=28)
                )])
                fig_bm.update_layout(paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0),
                                     height=max(120, len(bm_df)*30+55))
                st.plotly_chart(fig_bm, use_container_width=True)

    # ---- TAB 4: EXPORT ----
    with tab4:
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
