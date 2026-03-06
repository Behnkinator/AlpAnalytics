"""Player Lookup — AlpAnalytics.
Universal search for MLB pitchers and hitters with Statcast data,
career stats, movement plots, heat maps, spray charts, and game logs.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pybaseball as _pybb
try:
    _pybb.cache.enable()   # disk cache — survives app restarts
except Exception:
    pass

from pybaseball import playerid_lookup, statcast_pitcher, statcast_batter
from pybaseball import pitching_stats as fg_pitch_stats
from pybaseball import batting_stats as fg_bat_stats
try:
    from pybaseball import percentile_rankings as _pct_rankings
    _HAS_PCT = True
except ImportError:
    _HAS_PCT = False
try:
    from pybaseball import statcast_pitcher_expected_stats as _sc_pitcher_exp
    from pybaseball import statcast_pitcher_exitvelo_barrels as _sc_pitcher_ev
    from pybaseball import statcast_batter_expected_stats as _sc_batter_exp
    from pybaseball import statcast_batter_exitvelo_barrels as _sc_batter_ev
    _HAS_AGG = True
except ImportError:
    _HAS_AGG = False
from utils import THEME_CSS, load_stuff_model, add_engineered_features, run_stuff_plus

CURRENT_YEAR = 2025

@st.cache_resource(show_spinner=False)
def _get_stuff_model():
    return load_stuff_model()

# ── Pitch type colors / display names ──────────────────────────────────────────
PT_COLOR = {
    'FF': '#f05050', 'SI': '#f0a050', 'FC': '#f0c040',
    'SL': '#3dcc7c', 'ST': '#20aa60', 'CU': '#3a9dff',
    'CH': '#b47fff', 'FS': '#ff7fbf', 'KC': '#40c0f0',
    'KN': '#a0a0a0', 'EP': '#c0c0c0', 'CS': '#6080ff',
    'SV': '#80d080', 'FO': '#d080ff',
}
PT_NAME = {
    'FF': '4-Seam FB', 'SI': 'Sinker',    'FC': 'Cutter',
    'SL': 'Slider',    'ST': 'Sweeper',   'CU': 'Curveball',
    'CH': 'Changeup',  'FS': 'Splitter',  'KC': 'Knuckle-Curve',
    'KN': 'Knuckleball','EP': 'Eephus',   'CS': 'Slow Curve',
    'SV': 'Slurve',    'FO': 'Forkball',
}

# ── Batted ball event colors ───────────────────────────────────────────────────
EV_COLOR = {
    'single': '#3a9dff', 'double': '#3dcc7c',
    'triple': '#f0c040', 'home_run': '#f05050',
}
OUT_EVENTS = {
    'field_out', 'grounded_into_double_play', 'double_play',
    'force_out', 'fielders_choice', 'fielders_choice_out',
    'sac_fly', 'sac_fly_double_play', 'sac_bunt',
    'batter_interference', 'fan_interference', 'caught_stealing_2b',
    'other_out',
}

# ── Dark layout template for Plotly ───────────────────────────────────────────
_DL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e8eaf0', family='Barlow', size=12),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ── Utility functions ──────────────────────────────────────────────────────────
def pt_color(pt):
    return PT_COLOR.get(str(pt).upper(), '#8a94aa')


def pt_name(pt):
    return PT_NAME.get(str(pt).upper(), str(pt))


def fmt_val(v, decimals=2, suffix=''):
    if pd.isna(v):
        return '—'
    if isinstance(v, float):
        return f"{v:.{decimals}f}{suffix}"
    return f"{v}{suffix}"


def savant_url(*id_vals):
    """Return Baseball Savant video URL. Tries each id in order (play_id UUID first, sv_id fallback)."""
    for v in id_vals:
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        if v is None:
            continue
        s = str(v).strip()
        if s and s not in ('', 'nan', 'None', 'NaT', 'nat'):
            return f"https://baseballsavant.mlb.com/sporty-videos?playId={s}"
    return ''


# ── Query parser ──────────────────────────────────────────────────────────────
def parse_query(q):
    """Parse 'Last', 'First Last', or 'Last, First' into (last, first)."""
    q = q.strip()
    if ',' in q:
        parts = [p.strip() for p in q.split(',', 1)]
        return parts[0], parts[1]
    parts = q.split()
    if len(parts) == 1:
        return parts[0], ''
    return parts[-1], ' '.join(parts[:-1])  # last word = last name


# ── Per-year download helpers (cached by year, shared across ALL players) ─────
# Caching at the year level means once any player triggers a year download,
# every subsequent lookup for that year is instant — far faster than caching
# per-player per-range which re-downloads all years for each new player.
_FG_TTL = 86400 * 7   # 7 days — historical seasons don't change

@st.cache_data(ttl=_FG_TTL, show_spinner=False)
def _fg_pitch_year(year):
    try:
        return fg_pitch_stats(year, year, qual=0)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=_FG_TTL, show_spinner=False)
def _fg_bat_year(year):
    try:
        return fg_bat_stats(year, year, qual=0)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=_FG_TTL, show_spinner=False)
def _sc_pitch_exp_year(year):
    if not _HAS_AGG:
        return pd.DataFrame()
    try:
        return _sc_pitcher_exp(year, minPA=1)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=_FG_TTL, show_spinner=False)
def _sc_pitch_ev_year(year):
    if not _HAS_AGG:
        return pd.DataFrame()
    try:
        return _sc_pitcher_ev(year, minBBE=1)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=_FG_TTL, show_spinner=False)
def _sc_bat_exp_year(year):
    if not _HAS_AGG:
        return pd.DataFrame()
    try:
        return _sc_batter_exp(year, minPA=1)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=_FG_TTL, show_spinner=False)
def _sc_bat_ev_year(year):
    if not _HAS_AGG:
        return pd.DataFrame()
    try:
        return _sc_batter_ev(year, minBBE=1)
    except Exception:
        return pd.DataFrame()


# ── Cached API functions ───────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="Searching players…")
def search_player(last, first):
    try:
        return playerid_lookup(last.strip(), first.strip(), fuzzy=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def get_player_info(mlbam_id):
    """Fetch position and team from the free MLB Stats API (no field filter)."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            p = r.json()['people'][0]
            pos  = p.get('primaryPosition', {})
            abbr = pos.get('abbreviation', 'N/A')
            ptype = pos.get('type', '')
            pname = pos.get('name', '').lower()
            # Detect two-way players by any of several MLB API signals
            twoway = (
                'two-way' in ptype.lower()
                or 'two-way' in pname
                or abbr in ('TWP', 'TW')
            )
            return {
                'pos_abbr':  abbr,
                'pos_type':  ptype,
                'is_twoway': twoway,
                'team':      p.get('currentTeam', {}).get('name', 'N/A'),
                'active':    p.get('active', False),
                'full_name': p.get('fullName', ''),
            }
    except Exception:
        pass
    return {'pos_abbr': 'N/A', 'pos_type': '', 'is_twoway': False,
            'team': 'N/A', 'active': False, 'full_name': ''}


@st.cache_data(ttl=3600, show_spinner="Loading Statcast pitching data…")
def get_sc_pitcher(player_id, season):
    try:
        df = statcast_pitcher(f"{season}-03-01", f"{season}-11-30", player_id)
        return df if df is not None and len(df) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading Statcast batting data…")
def get_sc_batter(player_id, season):
    try:
        df = statcast_batter(f"{season}-03-01", f"{season}-11-30", player_id)
        return df if df is not None and len(df) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_career_pitching(fg_id, start_yr, end_yr):
    """Assemble per-season FanGraphs pitching rows by calling the per-year cache."""
    dfs = []
    for yr in range(end_yr, start_yr - 1, -1):
        yr_df = _fg_pitch_year(yr)
        if yr_df is None or len(yr_df) == 0:
            continue
        for id_col in ['IDfg', 'playerid', 'id']:
            if id_col in yr_df.columns:
                row = yr_df[yr_df[id_col] == fg_id]
                if len(row) > 0:
                    dfs.append(row)
                break
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    if 'Season' in result.columns:
        result = result.sort_values('Season', ascending=False)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_career_batting(fg_id, start_yr, end_yr):
    """Assemble per-season FanGraphs batting rows by calling the per-year cache."""
    dfs = []
    for yr in range(end_yr, start_yr - 1, -1):
        yr_df = _fg_bat_year(yr)
        if yr_df is None or len(yr_df) == 0:
            continue
        for id_col in ['IDfg', 'playerid', 'id']:
            if id_col in yr_df.columns:
                row = yr_df[yr_df[id_col] == fg_id]
                if len(row) > 0:
                    dfs.append(row)
                break
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    if 'Season' in result.columns:
        result = result.sort_values('Season', ascending=False)
    return result


# ── Percentile rankings ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_percentile_data(mlbam_id, season, is_pitcher):
    """Return a dict of percentile ranks for the player from Baseball Savant."""
    if not _HAS_PCT:
        return {}
    try:
        ptype = 'pitcher' if is_pitcher else 'batter'
        df = _pct_rankings(season, type=ptype)
        if df is None or len(df) == 0:
            return {}
        pid_col = next((c for c in df.columns
                        if c.lower() in ('player_id', 'mlbam', 'pid', 'id')), None)
        if pid_col is None:
            return {}
        row = df[df[pid_col] == mlbam_id]
        return row.iloc[0].to_dict() if len(row) > 0 else {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_career_sc_pitcher(mlbam_id, start_yr, end_yr):
    """Return per-season Statcast pitcher aggregate stats, using per-year cache."""
    rows = []
    for yr in range(start_yr, end_yr + 1):
        try:
            exp = _sc_pitch_exp_year(yr)
            ev  = _sc_pitch_ev_year(yr)
            pid_col_e = next((c for c in exp.columns if 'player_id' in c.lower()), None) if len(exp) > 0 else None
            pid_col_v = next((c for c in ev.columns  if 'player_id' in c.lower()), None) if len(ev) > 0 else None
            r_exp = exp[exp[pid_col_e] == mlbam_id].iloc[0].to_dict() if pid_col_e and len(exp[exp[pid_col_e] == mlbam_id]) > 0 else {}
            r_ev  = ev[ev[pid_col_v]   == mlbam_id].iloc[0].to_dict() if pid_col_v  and len(ev[ev[pid_col_v]   == mlbam_id]) > 0 else {}
            if r_exp or r_ev:
                rows.append({**r_exp, **r_ev, 'season': yr})
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_career_sc_batter(mlbam_id, start_yr, end_yr):
    """Return per-season Statcast batter aggregate stats, using per-year cache."""
    rows = []
    for yr in range(start_yr, end_yr + 1):
        try:
            exp = _sc_bat_exp_year(yr)
            ev  = _sc_bat_ev_year(yr)
            pid_col_e = next((c for c in exp.columns if 'player_id' in c.lower()), None) if len(exp) > 0 else None
            pid_col_v = next((c for c in ev.columns  if 'player_id' in c.lower()), None) if len(ev) > 0 else None
            r_exp = exp[exp[pid_col_e] == mlbam_id].iloc[0].to_dict() if pid_col_e and len(exp[exp[pid_col_e] == mlbam_id]) > 0 else {}
            r_ev  = ev[ev[pid_col_v]   == mlbam_id].iloc[0].to_dict() if pid_col_v  and len(ev[ev[pid_col_v]   == mlbam_id]) > 0 else {}
            if r_exp or r_ev:
                rows.append({**r_exp, **r_ev, 'season': yr})
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Percentile display ─────────────────────────────────────────────────────────
def _fmt_sc(row, col, fmt='.1f', pct=False):
    """Format a value from an aggregate Statcast row dict."""
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    try:
        return f"{float(v):{fmt}}{'%' if pct else ''}"
    except Exception:
        return '—'


def _pct_color(p):
    """Map a percentile (0-100) to a color. Higher = better (Savant convention)."""
    try:
        p = int(p)
    except Exception:
        return '#505a70'
    if p >= 80: return '#3dcc7c'
    if p >= 60: return '#88cc88'
    if p >= 40: return '#8a94aa'
    if p >= 20: return '#f0a050'
    return '#f05050'


def _render_percentile_row(pct_dict, metrics):
    """Render a row of percentile circles. metrics = list of (key, label)."""
    if not pct_dict:
        return
    cards = ''
    for key, label in metrics:
        val = pct_dict.get(key)
        if val is None:
            continue
        try:
            p = int(round(float(val)))
        except Exception:
            continue
        color = _pct_color(p)
        cards += (
            f"<div style='text-align:center;flex:0 0 auto;'>"
            f"<div style='width:52px;height:52px;border-radius:50%;border:3px solid {color};"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-family:\"Barlow Condensed\",sans-serif;font-size:1.15rem;font-weight:800;color:{color};'>"
            f"{p}</div>"
            f"<div style='font-size:0.62rem;color:#8a94aa;margin-top:4px;font-weight:600;"
            f"letter-spacing:0.04em;text-transform:uppercase;width:58px;word-wrap:break-word;"
            f"text-align:center;'>{label}</div>"
            f"</div>"
        )
    if cards:
        st.markdown(
            f"<div style='display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;"
            f"margin:0.8rem 0 1.2rem;'>{cards}</div>",
            unsafe_allow_html=True,
        )


# ── Plotly helpers ─────────────────────────────────────────────────────────────
def dark_stats_table(df, cols, labels, pct_cols=None, dec2_cols=None, int_cols=None):
    """Render a dark-themed Plotly table for career stats."""
    pct_cols  = pct_cols  or []
    dec2_cols = dec2_cols or []
    int_cols  = int_cols  or []

    cell_vals = []
    for c in cols:
        if c not in df.columns:
            cell_vals.append(['—'] * len(df))
            continue
        col_data = df[c]
        if c in pct_cols:
            cell_vals.append([f"{v*100:.1f}%" if not pd.isna(v) else '—' for v in col_data])
        elif c in dec2_cols:
            cell_vals.append([f"{v:.2f}" if not pd.isna(v) else '—' for v in col_data])
        elif c in int_cols:
            cell_vals.append([str(int(v)) if not pd.isna(v) else '—' for v in col_data])
        else:
            cell_vals.append([str(v) if not pd.isna(v) else '—' for v in col_data])

    # Alternate row colors
    n = len(df)
    fill_colors = [['#141820' if i % 2 == 0 else '#1c2230' for i in range(n)]] * len(cols)

    fig = go.Figure(go.Table(
        header=dict(
            values=labels,
            fill_color='#1c2230',
            line_color='#2a3348',
            font=dict(color='#f0c040', family='Barlow Condensed', size=13),
            align='center', height=32,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=fill_colors,
            line_color='#2a3348',
            font=dict(color='#e8eaf0', family='Barlow', size=12),
            align='center', height=28,
        ),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=max(120, 32 + 28 * n + 10),
    )
    return fig


# ── Statcast season stats from raw pitch data ──────────────────────────────────
def _show_statcast_section(df_raw, season):
    """Compute and render a Statcast stats row from raw pitch-level Statcast data."""
    if df_raw is None or len(df_raw) == 0:
        return

    df = df_raw.copy()
    n_pitches = len(df)
    bip = df[df['launch_speed'].notna()].copy() if 'launch_speed' in df.columns else pd.DataFrame()
    n_bip = len(bip)
    bf = int(df['at_bat_number'].nunique()) if 'at_bat_number' in df.columns else 0

    barrels = int(bip['barrel'].fillna(0).sum()) if ('barrel' in bip.columns and n_bip > 0) else 0
    brl_pct = barrels / n_bip * 100 if n_bip > 0 else None

    avg_ev = float(bip['launch_speed'].mean()) if n_bip > 0 else None
    max_ev = float(bip['launch_speed'].max())  if n_bip > 0 else None

    if 'launch_angle' in bip.columns and n_bip > 0:
        avg_la = float(bip['launch_angle'].mean())
        sweet  = int(((bip['launch_angle'] >= 8) & (bip['launch_angle'] <= 32)).sum())
        sweet_pct = sweet / n_bip * 100
    else:
        avg_la = sweet_pct = None

    hh_pct = ((bip['launch_speed'] >= 95).sum() / n_bip * 100) if n_bip > 0 else None

    def _cm(col):
        return float(df[col].mean()) if col in df.columns and df[col].notna().any() else None

    xba   = _cm('estimated_ba_using_speedangle')
    xslg  = _cm('estimated_slg_using_speedangle')
    xwoba = _cm('estimated_woba_using_speedangle')
    woba  = _cm('woba_value')

    k_pct = bb_pct = None
    if bf > 0 and 'events' in df.columns and 'at_bat_number' in df.columns:
        k_pct  = df[df['events'].isin(['strikeout','strikeout_double_play'])]['at_bat_number'].nunique() / bf * 100
        bb_pct = df[df['events'].isin(['walk','intent_walk'])]['at_bat_number'].nunique() / bf * 100

    def _f(v, fmt='.1f', pct_suffix=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '—'
        try:
            return f"{v:{fmt}}{'%' if pct_suffix else ''}"
        except Exception:
            return '—'

    la_str = (f"{avg_la:.1f}°" if avg_la is not None else '—')

    cells = [
        ('Pitches',      str(n_pitches)),
        ('BIP',          str(n_bip)),
        ('Barrels',      str(barrels)),
        ('Brl%',         _f(brl_pct,  '.1f', True)),
        ('Avg EV',       _f(avg_ev,   '.1f')),
        ('Max EV',       _f(max_ev,   '.1f')),
        ('Avg LA',       la_str),
        ('Sweet Spot%',  _f(sweet_pct, '.1f', True)),
        ('xBA',          _f(xba,  '.3f')),
        ('xSLG',         _f(xslg, '.3f')),
        ('wOBA',         _f(woba, '.3f')),
        ('xwOBA',        _f(xwoba,'.3f')),
        ('Hard Hit%',    _f(hh_pct, '.1f', True)),
        ('K%',           _f(k_pct,  '.1f', True)),
        ('BB%',          _f(bb_pct, '.1f', True)),
    ]

    th_style = "padding:6px 12px;white-space:nowrap;text-align:center;"
    td_style = "padding:6px 12px;color:#e8eaf0;font-weight:600;text-align:center;white-space:nowrap;"
    header = ''.join(f"<th style='{th_style}'>{h}</th>" for h, _ in cells)
    row    = ''.join(f"<td style='{td_style}'>{v}</td>" for _, v in cells)

    st.markdown(f"""
<div style="margin-top:1.4rem;">
  <div style="font-family:'Barlow Condensed',sans-serif;color:#3a9dff;font-size:0.85rem;
              font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">
    Statcast Stats — {season}
  </div>
  <div style="overflow-x:auto;">
  <table style="border-collapse:collapse;font-size:0.8rem;font-family:'Barlow',sans-serif;
                background:#141820;border:1px solid #2a3348;border-radius:6px;min-width:100%;">
    <thead>
      <tr style="background:#1c2230;color:#3a9dff;font-family:'Barlow Condensed',sans-serif;
                 font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">
        {header}
      </tr>
    </thead>
    <tbody>
      <tr style="background:#141820;">{row}</tr>
    </tbody>
  </table>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Baseball field drawing (Statcast hc_x/hc_y space, centered) ───────────────
def field_traces():
    """Return Plotly scatter traces for a baseball field outline."""
    traces = []

    # Outfield wall — quadratic Bezier through LF, CF, RF
    lf  = np.array([-110.0, 118.0])
    cf_ = np.array([   0.0, 172.0])
    rf  = np.array([ 110.0, 118.0])
    ctrl = 2 * cf_ - 0.5 * lf - 0.5 * rf
    u = np.linspace(0, 1, 120)
    wx = (1-u)**2 * lf[0] + 2*u*(1-u) * ctrl[0] + u**2 * rf[0]
    wy = (1-u)**2 * lf[1] + 2*u*(1-u) * ctrl[1] + u**2 * rf[1]
    traces.append(go.Scatter(x=wx, y=wy, mode='lines',
                             line=dict(color='#3a4558', width=2),
                             showlegend=False, hoverinfo='skip'))

    # Foul lines from home to the wall
    ang = np.radians(45)
    flen = 155.0
    for sign in [1, -1]:
        traces.append(go.Scatter(
            x=[0, sign * flen * np.cos(ang)],
            y=[0,       flen * np.sin(ang)],
            mode='lines',
            line=dict(color='#3a4558', width=1.5, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))

    # Base paths: home → 1B → 2B → 3B → home
    bx = [0,  71,   0, -71, 0]
    by = [0,  70, 140,  70, 0]
    traces.append(go.Scatter(x=bx, y=by, mode='lines',
                             line=dict(color='#505a70', width=1.5),
                             showlegend=False, hoverinfo='skip'))

    # Base markers
    traces.append(go.Scatter(
        x=[71, 0, -71], y=[70, 140, 70],
        mode='markers',
        marker=dict(color='#8a94aa', size=7, symbol='square'),
        showlegend=False, hoverinfo='skip',
    ))

    # Pitching mound
    traces.append(go.Scatter(
        x=[0], y=[55],
        mode='markers',
        marker=dict(color='#505a70', size=10, symbol='circle'),
        showlegend=False, hoverinfo='skip',
    ))

    return traces


# ── Zone heat map (5×5 grid for plate_x / plate_z) ────────────────────────────
def build_zone_heatmap(df_zone, metric_col, title, colorscale='RdYlGn', zrange=None):
    """5×5 zone grid heat map. plate_x / plate_z in feet (Statcast units)."""
    x_bins = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
    z_bins = [1.0,   1.6,  2.2,  2.8,  3.4,  4.0]
    x_ctrs = [-1.2, -0.6,  0.0,  0.6,  1.2]
    z_ctrs = [ 3.7,  3.1,  2.5,  1.9,  1.3]   # inverted (high z = top of chart)

    df_z = df_zone.dropna(subset=['plate_x', 'plate_z', metric_col]).copy()
    df_z['xb'] = pd.cut(df_z['plate_x'], x_bins, labels=False)
    df_z['zb'] = pd.cut(df_z['plate_z'], z_bins, labels=False)

    grid   = np.full((5, 5), np.nan)
    counts = np.zeros((5, 5), dtype=int)

    for (xi, zi), grp in df_z.dropna(subset=['xb', 'zb']).groupby(['xb', 'zb']):
        xi, zi = int(xi), int(zi)
        row_idx = 4 - zi        # invert so high zone is at top of chart
        grid[row_idx, xi]   = grp[metric_col].mean()
        counts[row_idx, xi] = len(grp)

    # Suppress cells with fewer than 3 pitches
    sparse = counts < 3
    display_grid = np.where(sparse, np.nan, grid)

    text = []
    for ri in range(5):
        row_t = []
        for ci in range(5):
            v = display_grid[ri, ci]
            n = counts[ri, ci]
            if np.isnan(v):
                row_t.append('')
            elif '%' in title or 'Rate' in title or 'Whiff' in title:
                row_t.append(f"{v*100:.0f}%<br><sub>n={n}</sub>")
            else:
                row_t.append(f"{v:.3f}<br><sub>n={n}</sub>")
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=display_grid, x=x_ctrs, y=z_ctrs,
        text=text, texttemplate='%{text}',
        colorscale=colorscale,
        zmin=zrange[0] if zrange else None,
        zmax=zrange[1] if zrange else None,
        showscale=True,
        hoverongaps=False,
        colorbar=dict(
            tickfont=dict(color='#8a94aa', size=10),
            outlinewidth=0, len=0.8,
        ),
    ))
    # Strike zone rectangle
    fig.add_shape(type='rect', x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                  line=dict(color='#f0c040', width=2),
                  fillcolor='rgba(0,0,0,0)')
    # Home plate marker
    fig.add_shape(type='rect', x0=-0.25, x1=0.25, y0=0.9, y1=1.1,
                  fillcolor='#8a94aa', line=dict(width=0))

    fig.update_layout(
        title=dict(text=title, font=dict(color='#f0c040', size=13)),
        **_DL,
        height=360,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#8a94aa'),
                   title=dict(text='← Glove Side · Arm Side →', font=dict(color='#505a70', size=10))),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#8a94aa'),
                   title=dict(text='Height (ft)', font=dict(color='#505a70', size=10))),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Player Lookup · AlpAnalytics",
    layout="wide", page_icon="🔍",
    initial_sidebar_state="expanded",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-logo">⚾ AlpAnalytics</div>', unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <p class="app-title">Player Lookup</p>
    <p class="app-subtitle">AlpAnalytics · Pitcher &amp; Hitter Profiles · Baseball Savant Data</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if 'player_state' not in st.session_state:
    st.session_state.player_state = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Auto-trigger search if navigated here from the Home page search bar
_auto_q = st.session_state.get('_query', '')
if _auto_q:
    _last, _first = parse_query(_auto_q)
    st.session_state.search_results = search_player(_last, _first)
    st.session_state.player_state   = None
    del st.session_state['_query']

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH SECTION
# ══════════════════════════════════════════════════════════════════════════════
def _do_live_search():
    q = st.session_state.get('live_search_q', '').strip()
    if len(q) >= 3:
        last, first = parse_query(q)
        st.session_state.search_results = search_player(last, first)
        st.session_state.player_state   = None
    elif len(q) == 0:
        st.session_state.search_results = None
        st.session_state.player_state   = None

st.markdown('<div class="section-header">Player Search</div>', unsafe_allow_html=True)
st.text_input(
    "Player",
    placeholder="Start typing a name — e.g. 'Ohtani'  or  'Gerrit Cole'  or  'Cole, Gerrit'",
    label_visibility="collapsed",
    key="live_search_q",
    on_change=_do_live_search,
)
st.markdown(
    "<div style='color:#505a70;font-size:0.75rem;margin-top:-0.3rem;margin-bottom:0.8rem;'>"
    "Results appear automatically after 3+ characters</div>",
    unsafe_allow_html=True,
)

# Display search results
if st.session_state.search_results is not None:
    res = st.session_state.search_results
    if res is None or len(res) == 0:
        st.warning("No players found. Try a different spelling or leave First Name blank for broader search.")
    else:
        res_clean = res.copy()
        res_clean['_Name'] = (res_clean['name_first'].fillna('').str.title()
                              + ' ' + res_clean['name_last'].fillna('').str.title())
        res_clean['_Years'] = (
            res_clean['mlb_played_first'].fillna('').astype(str).str.replace('.0', '', regex=False)
            + '–'
            + res_clean['mlb_played_last'].fillna('').astype(str).str.replace('.0', '', regex=False)
        )
        st.markdown(f"**{len(res_clean)} result(s) found** — click Select to view profile:")
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        for i, (_, row) in enumerate(res_clean.iterrows()):
            mlbam = row.get('key_mlbam', None)
            fg    = row.get('key_fangraphs', None)
            debut = row.get('mlb_played_first', CURRENT_YEAR)
            last  = row.get('mlb_played_last',  CURRENT_YEAR)
            try:
                debut = int(debut) if not pd.isna(debut) else CURRENT_YEAR
            except Exception:
                debut = CURRENT_YEAR
            try:
                last = int(last) if not pd.isna(last) else CURRENT_YEAR
            except Exception:
                last = CURRENT_YEAR
            try:
                fg_id = int(fg) if fg and not pd.isna(fg) else None
            except Exception:
                fg_id = None
            try:
                mlbam_id = int(mlbam) if mlbam and not pd.isna(mlbam) else None
            except Exception:
                mlbam_id = None

            rc1, rc2 = st.columns([5, 1])
            with rc1:
                _id_hint = f"MLBAM: {mlbam_id}" if mlbam_id else (f"FG: {fg_id}" if fg_id else "")
                st.markdown(
                    f"<div style='padding:0.55rem 1rem;background:#141820;border:1px solid #2a3348;"
                    f"border-radius:5px;display:flex;align-items:center;gap:1.2rem;margin-bottom:0.3rem;'>"
                    f"<span style='color:#e8eaf0;font-weight:700;font-size:1rem;'>{row['_Name']}</span>"
                    f"<span style='color:#505a70;font-size:0.8rem;'>{row['_Years']}</span>"
                    f"<span style='color:#3a3d4a;font-size:0.72rem;'>{_id_hint}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with rc2:
                if st.button("Select →", key=f"sel_{i}_{mlbam_id}_{fg_id}"):
                    st.session_state.player_state = {
                        'mlbam_id':   mlbam_id,
                        'fg_id':      fg_id,
                        'name':       row['_Name'],
                        'debut_year': debut,
                        'last_year':  last,
                    }
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PLAYER PROFILE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.player_state:
    st.stop()

ps        = st.session_state.player_state
mlbam_id  = ps['mlbam_id']
fg_id     = ps['fg_id']
debut_yr  = ps['debut_year']
last_yr   = ps['last_year']
player_name = ps['name']

# MLB Stats API — position + team
info      = get_player_info(mlbam_id) if mlbam_id else {}
pos_abbr  = info.get('pos_abbr', 'N/A')
pos_type  = info.get('pos_type', '')
team      = info.get('team', 'N/A')

# Pitcher = any primary position 'P', or position-type containing 'Pitcher'
is_pitcher = (pos_abbr == 'P') or ('pitcher' in pos_type.lower()) or ('pitcher' in pos_abbr.lower())
is_twoway  = info.get('is_twoway', False)
# Two-way players (e.g. Ohtani): show both pitcher + hitter views
is_hitter  = (not is_pitcher) or is_twoway

# FanGraphs two-way fallback: the MLB API may show a two-way player as DH
# when they're injured/not pitching (e.g. Ohtani 2024). Check FanGraphs
# career stats: if they appear in BOTH pitching and batting records with
# meaningful PA, force two-way mode.
if not is_twoway and fg_id is not None:
    try:
        # Check only the last 6 years — enough to catch Ohtani-style players
        # who may be injured but still have recent pitching history.
        _fg_start = max(debut_yr, CURRENT_YEAR - 6, 2015)
        _p = get_career_pitching(fg_id, _fg_start, CURRENT_YEAR)
        _b = get_career_batting(fg_id, _fg_start, CURRENT_YEAR)
        _has_p = _p is not None and len(_p) > 0
        _has_b = _b is not None and len(_b) > 0
        if _has_b:
            _pa_col = next((c for c in ['PA', 'G'] if c in _b.columns), None)
            _total_pa = pd.to_numeric(_b[_pa_col], errors='coerce').sum() if _pa_col else 0
            _has_b = int(_total_pa) >= 50  # ignore pitchers with token at-bats
        if _has_p and _has_b:
            is_twoway  = True
            is_pitcher = True
            is_hitter  = True
    except Exception:
        pass

# ── Player header ──────────────────────────────────────────────────────────────
photo_url = (
    f"https://img.mlbstatic.com/mlb-photos/image/upload/"
    f"d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
    f"v1/people/{mlbam_id}/headshot/67/current"
) if mlbam_id else ""

ph1, ph2, ph3 = st.columns([1, 4, 2])
with ph1:
    if photo_url:
        st.image(photo_url, width=120)

with ph2:
    pos_disp = "TWP" if is_twoway else pos_abbr
    years_str = f"{debut_yr}–{'Present' if last_yr >= CURRENT_YEAR - 1 else last_yr}"
    st.markdown(f"""
    <div style='padding:1rem 0;'>
      <div style='font-family:"Barlow Condensed",sans-serif;font-size:2.8rem;font-weight:800;
                  color:#e8eaf0;line-height:1;letter-spacing:0.02em;'>{player_name.upper()}</div>
      <div style='color:#8a94aa;font-size:0.92rem;margin-top:0.5rem;'>
        <span style='color:#f0c040;font-weight:700;'>{team}</span>
        &nbsp;·&nbsp; {pos_disp} &nbsp;·&nbsp; {years_str}
      </div>
    </div>
    """, unsafe_allow_html=True)

with ph3:
    # Statcast only goes back to 2015
    sc_start = max(debut_yr, 2015)
    sc_end   = min(last_yr, CURRENT_YEAR)
    avail_seasons = list(range(max(sc_start, 2015), sc_end + 1))[::-1]
    if not avail_seasons:
        avail_seasons = [CURRENT_YEAR]
    season = st.selectbox("Season", avail_seasons, key="season_sel")
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    savant_game_url = f"https://baseballsavant.mlb.com/savant-player/{mlbam_id}" if mlbam_id else ""
    if savant_game_url:
        st.markdown(
            f"<a href='{savant_game_url}' target='_blank' style='color:#3a9dff;font-size:0.8rem;'>"
            f"📊 View on Baseball Savant ↗</a>",
            unsafe_allow_html=True,
        )

st.markdown("<hr style='border-color:#2a3348;margin:0.5rem 0 0.8rem;'>", unsafe_allow_html=True)

# ── Percentile Rankings ──────────────────────────────────────────────────────
_pct_data = get_percentile_data(mlbam_id, season, is_pitcher) if mlbam_id else {}
if _pct_data:
    st.markdown(
        "<div style='font-family:\"Barlow Condensed\",sans-serif;font-size:0.78rem;font-weight:700;"
        "color:#8a94aa;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem;'>"
        "Percentile Rankings (Baseball Savant)</div>",
        unsafe_allow_html=True,
    )
    if is_pitcher:
        _render_percentile_row(_pct_data, [
            ('xera',            'xERA'),
            ('xba',             'xBA'),
            ('fastball_avg_speed', 'FB Velo'),
            ('ev',              'Avg EV'),
            ('whiff_percent',   'Whiff%'),
            ('k_percent',       'K%'),
            ('bb_percent',      'BB%'),
            ('barrels_per_bbe', 'Barrel%'),
            ('hard_hit_percent','Hard Hit%'),
            ('extension',       'Extension'),
        ])
    else:
        _render_percentile_row(_pct_data, [
            ('xba',             'xBA'),
            ('xslg',            'xSLG'),
            ('xwoba',           'xwOBA'),
            ('ev',              'Avg EV'),
            ('speed',           'Sprint Spd'),
            ('whiff_percent',   'Whiff%'),
            ('k_percent',       'K%'),
            ('bb_percent',      'BB%'),
            ('barrels_per_bbe', 'Barrel%'),
            ('hard_hit_percent','Hard Hit%'),
        ])
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#2a3348;margin:0 0 1.2rem;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PITCHER VIEW
# ══════════════════════════════════════════════════════════════════════════════
if is_pitcher:
    st.markdown(
        f"<div style='font-family:\"Barlow Condensed\",sans-serif;font-size:1.3rem;font-weight:700;"
        f"color:#f0c040;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;'>"
        f"⚾ Pitcher Profile</div>",
        unsafe_allow_html=True,
    )
    ptab1, ptab2, ptab3 = st.tabs(["Career Stats", "Arsenal & Movement", "Game Log"])

    # ── Pitcher Tab 1: Career Stats ───────────────────────────────────────────
    with ptab1:
        if fg_id is None:
            st.info("No FanGraphs ID found for this player. Career stats unavailable.")
        else:
            with st.spinner("Loading career pitching stats…"):
                career_df = get_career_pitching(fg_id, max(debut_yr, 2008), CURRENT_YEAR)

            if career_df is None or len(career_df) == 0:
                st.info("No career pitching stats found.")
            else:
                pitch_cols   = ['Season', 'Team', 'IP', 'ERA', 'FIP', 'xFIP', 'K%', 'BB%', 'WHIP', 'WAR']
                pitch_labels = ['Season', 'Team', 'IP', 'ERA', 'FIP', 'xFIP', 'K%', 'BB%', 'WHIP', 'WAR']
                available    = [c for c in pitch_cols if c in career_df.columns]
                av_labels    = [pitch_labels[pitch_cols.index(c)] for c in available]

                # Determine which columns have which format
                pct_cols_p  = [c for c in ['K%', 'BB%'] if c in available]
                dec2_cols_p = [c for c in ['IP', 'ERA', 'FIP', 'xFIP', 'WHIP', 'WAR'] if c in available]

                fig_tbl = dark_stats_table(
                    career_df, available, av_labels,
                    pct_cols=pct_cols_p, dec2_cols=dec2_cols_p,
                )
                st.plotly_chart(fig_tbl, use_container_width=True)

            # ── Career Statcast Stats (aggregate, multi-season) ──────────────
            if mlbam_id:
                sc_start = max(debut_yr, 2015)
                sc_end   = min(last_yr, CURRENT_YEAR)
                with st.spinner("Loading career Statcast stats…"):
                    career_sc = get_career_sc_pitcher(mlbam_id, sc_start, sc_end)

                if career_sc is not None and len(career_sc) > 0:
                    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-family:\"Barlow Condensed\",sans-serif;color:#3a9dff;"
                        "font-size:0.85rem;font-weight:700;letter-spacing:0.1em;"
                        "text-transform:uppercase;margin-bottom:0.5rem;'>Statcast Stats (Career)</div>",
                        unsafe_allow_html=True,
                    )

                    # Map common column names from pybaseball aggregate functions
                    col_map = {
                        'est_ba':          'xBA',
                        'est_slg':         'xSLG',
                        'est_woba':        'xwOBA',
                        'woba':            'wOBA',
                        'avg_hit_speed':   'Avg EV',
                        'max_hit_speed':   'Max EV',
                        'barrels':         'Barrels',
                        'brl_percent':     'Brl%',
                        'ev95percent':     'HardHit%',
                        'anglesweetspotpercent': 'Sweet Spot%',
                        'pa':              'PA',
                        'bip':             'BIP',
                    }

                    # Deduplicate columns and sort descending by season
                    career_sc_sorted = career_sc.copy()
                    if 'season' in career_sc_sorted.columns:
                        career_sc_sorted = career_sc_sorted.sort_values('season', ascending=False)

                    # Build header & rows
                    present_cols = ['season'] + [c for c in col_map if c in career_sc_sorted.columns]
                    headers = ['Season'] + [col_map[c] for c in present_cols[1:]]
                    th_s = "padding:5px 10px;white-space:nowrap;text-align:center;"
                    td_s = "padding:5px 10px;color:#e8eaf0;font-weight:600;text-align:center;white-space:nowrap;"
                    header_html = ''.join(f"<th style='{th_s}'>{h}</th>" for h in headers)
                    rows_html = ''
                    for _, rw in career_sc_sorted.iterrows():
                        cells = [f"<td style='{td_s}'>{int(rw['season']) if 'season' in rw else '—'}</td>"]
                        for c in present_cols[1:]:
                            pct_flag = c in ('brl_percent', 'ev95percent', 'anglesweetspotpercent')
                            dec_fmt = '.3f' if c in ('est_ba','est_slg','est_woba','woba') else '.1f'
                            cells.append(f"<td style='{td_s}'>{_fmt_sc(rw, c, dec_fmt, pct_flag)}</td>")
                        rows_html += f"<tr>{''.join(cells)}</tr>"

                    st.markdown(f"""
<div style="overflow-x:auto;">
<table style="border-collapse:collapse;font-size:0.78rem;font-family:'Barlow',sans-serif;
              background:#141820;border:1px solid #2a3348;border-radius:6px;min-width:100%;">
  <thead>
    <tr style="background:#1c2230;color:#3a9dff;font-family:'Barlow Condensed',sans-serif;
               font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">{header_html}</tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)
                else:
                    # Fallback: single-season from raw pitch data
                    with st.spinner(f"Loading Statcast data for {season}…"):
                        _sc_p = get_sc_pitcher(mlbam_id, season)
                    _show_statcast_section(_sc_p, season)

    # ── Pitcher Tab 2: Arsenal & Movement ─────────────────────────────────────
    with ptab2:
        if not mlbam_id:
            st.info("No MLBAM ID — cannot load Statcast data.")
        else:
            with st.spinner(f"Loading {season} Statcast data…"):
                df_sc = get_sc_pitcher(mlbam_id, season)

            if df_sc is None or len(df_sc) == 0:
                st.warning(f"No Statcast pitching data found for {season}.")
            else:
                # Compute HB / IVB in inches (pfx_x and pfx_z are in feet in Statcast)
                # Pitcher's-own-view: negate pfx_x for ALL pitchers so the chart shows
                # the field as the pitcher sees it — positive x = 3B side (pitcher's right),
                # negative x = 1B side (pitcher's left).
                # RHP arm-side (3B) → RIGHT; LHP arm-side (1B) → LEFT.
                df_sc = df_sc.copy()
                _hand = str(df_sc['p_throws'].iloc[0]) if 'p_throws' in df_sc.columns and len(df_sc) > 0 else 'R'
                _x_sign = -1  # same for both handedness (pitcher's perspective)
                df_sc['hb_in']  = df_sc['pfx_x'].fillna(0) * 12 * _x_sign
                df_sc['ivb_in'] = df_sc['pfx_z'].fillna(0) * 12
                df_sc['pt_label'] = df_sc['pitch_type'].apply(pt_name)
                df_sc['pt_color'] = df_sc['pitch_type'].apply(pt_color)

                # Remove unknown/null pitch types
                df_sc = df_sc[df_sc['pitch_type'].notna() & (df_sc['pitch_type'] != '')]

                if len(df_sc) == 0:
                    st.warning("No valid pitch data in Statcast records.")
                else:
                    usage_series = df_sc['pitch_type'].value_counts()
                    total_pitches = len(df_sc)

                    # ── Row 1: Arsenal usage + Velocity by type ──
                    col_use, col_velo = st.columns(2)

                    with col_use:
                        usage_pct = (usage_series / total_pitches * 100).reset_index()
                        usage_pct.columns = ['pitch_type', 'pct']
                        usage_pct['label'] = usage_pct['pitch_type'].apply(pt_name)
                        usage_pct['color'] = usage_pct['pitch_type'].apply(pt_color)
                        usage_pct = usage_pct.sort_values('pct', ascending=True)

                        fig_use = go.Figure(go.Bar(
                            x=usage_pct['pct'],
                            y=usage_pct['label'],
                            orientation='h',
                            marker=dict(
                                color=usage_pct['color'].tolist(),
                                line=dict(color='#0c0f14', width=0.5),
                            ),
                            text=[f"{v:.1f}%" for v in usage_pct['pct']],
                            textposition='outside',
                            textfont=dict(color='#e8eaf0', size=11),
                        ))
                        fig_use.update_layout(
                            title=dict(text='Arsenal Usage', font=dict(color='#f0c040', size=13)),
                            xaxis=dict(showgrid=False, visible=False),
                            yaxis=dict(showgrid=False, tickfont=dict(color='#8a94aa', size=12)),
                            **_DL, height=320,
                        )
                        st.plotly_chart(fig_use, use_container_width=True)

                    with col_velo:
                        velo_data = []
                        for pt in usage_series.index:
                            subset = df_sc[df_sc['pitch_type'] == pt]['release_speed'].dropna()
                            if len(subset) >= 5:
                                velo_data.append(go.Box(
                                    y=subset, name=pt_name(pt),
                                    marker_color=pt_color(pt),
                                    line=dict(color=pt_color(pt)),
                                    boxmean=True,
                                    hovertemplate="%{y:.1f} mph<extra></extra>",
                                ))
                        if velo_data:
                            fig_velo = go.Figure(velo_data)
                            fig_velo.update_layout(
                                title=dict(text='Velocity by Pitch Type', font=dict(color='#f0c040', size=13)),
                                yaxis=dict(title='mph', gridcolor='#2a3348',
                                           tickfont=dict(color='#8a94aa')),
                                xaxis=dict(tickfont=dict(color='#8a94aa')),
                                showlegend=False,
                                **_DL, height=320,
                            )
                            st.plotly_chart(fig_velo, use_container_width=True)

                    # ── Row 2: Movement plot + Release point ──
                    col_mvt, col_rel = st.columns(2)

                    with col_mvt:
                        fig_mvt = go.Figure()
                        fig_mvt.add_shape(type='line', x0=-25, x1=25, y0=0, y1=0,
                                          line=dict(color='#2a3348', width=1))
                        fig_mvt.add_shape(type='line', x0=0, x1=0, y0=-25, y1=25,
                                          line=dict(color='#2a3348', width=1))

                        for pt in usage_series.index:
                            grp = df_sc[df_sc['pitch_type'] == pt].dropna(subset=['hb_in', 'ivb_in'])
                            if len(grp) < 3:
                                continue
                            fig_mvt.add_trace(go.Scatter(
                                x=grp['hb_in'], y=grp['ivb_in'],
                                mode='markers',
                                name=pt_name(pt),
                                marker=dict(
                                    color=pt_color(pt), size=5, opacity=0.55,
                                    line=dict(width=0),
                                ),
                                hovertemplate=(
                                    f"<b>{pt_name(pt)}</b><br>"
                                    "HB: %{x:.1f} in<br>IVB: %{y:.1f} in<extra></extra>"
                                ),
                            ))
                        fig_mvt.update_layout(
                            title=dict(text='Movement Profile (HB vs IVB)', font=dict(color='#f0c040', size=13)),
                            xaxis=dict(title='← 1B Side · 3B Side →', range=[-25, 25],
                                       gridcolor='#1c2230', zerolinecolor='#2a3348',
                                       tickfont=dict(color='#8a94aa')),
                            yaxis=dict(title='Induced Vertical Break (in)', range=[-25, 25],
                                       gridcolor='#1c2230', zerolinecolor='#2a3348',
                                       tickfont=dict(color='#8a94aa'), scaleanchor='x'),
                            legend=dict(font=dict(color='#8a94aa', size=10),
                                        bgcolor='rgba(0,0,0,0)', borderwidth=0),
                            **_DL, height=360,
                        )
                        st.plotly_chart(fig_mvt, use_container_width=True)

                    with col_rel:
                        if 'release_pos_x' in df_sc.columns and 'release_pos_z' in df_sc.columns:
                            # Same sign convention as movement: flip for RHP so arm-side is right
                            df_sc['rel_x_plot'] = df_sc['release_pos_x'] * _x_sign
                            fig_rel = go.Figure()
                            for pt in usage_series.index:
                                grp = df_sc[df_sc['pitch_type'] == pt].dropna(
                                    subset=['rel_x_plot', 'release_pos_z'])
                                if len(grp) < 3:
                                    continue
                                fig_rel.add_trace(go.Scatter(
                                    x=grp['rel_x_plot'], y=grp['release_pos_z'],
                                    mode='markers',
                                    name=pt_name(pt),
                                    marker=dict(color=pt_color(pt), size=5, opacity=0.55,
                                                line=dict(width=0)),
                                    hovertemplate=(
                                        f"<b>{pt_name(pt)}</b><br>"
                                        "Side: %{x:.2f} ft<br>Height: %{y:.2f} ft<extra></extra>"
                                    ),
                                ))
                            # Rubber line
                            fig_rel.add_shape(type='line', x0=-2.5, x1=2.5, y0=0, y1=0,
                                              line=dict(color='#505a70', width=2))
                            fig_rel.update_layout(
                                title=dict(text='Release Point (Pitcher\'s Perspective)', font=dict(color='#f0c040', size=13)),
                                xaxis=dict(title='← Glove Side · Arm Side →', gridcolor='#1c2230',
                                           tickfont=dict(color='#8a94aa'), zeroline=True,
                                           zerolinecolor='#2a3348'),
                                yaxis=dict(title='Release Height (ft)', gridcolor='#1c2230',
                                           tickfont=dict(color='#8a94aa')),
                                legend=dict(font=dict(color='#8a94aa', size=10),
                                            bgcolor='rgba(0,0,0,0)', borderwidth=0),
                                **_DL, height=360,
                            )
                            st.plotly_chart(fig_rel, use_container_width=True)

                    # ── Row 3: Pitch Arsenal Metrics table with Stuff+ ──────────
                    st.markdown(
                        "<div style='font-family:\"Barlow Condensed\",sans-serif;color:#f0c040;"
                        "font-size:1rem;font-weight:700;letter-spacing:0.1em;"
                        "text-transform:uppercase;margin:1.2rem 0 0.5rem;'>"
                        "Pitch Arsenal Metrics + Stuff+</div>",
                        unsafe_allow_html=True,
                    )
                    try:
                        _mdl = _get_stuff_model()
                        _sc_sp = add_engineered_features(df_sc.copy())
                        _sc_sp = run_stuff_plus(_sc_sp, _mdl)

                        _arsenal_rows = []
                        for _pt in usage_series.index:
                            _g = _sc_sp[_sc_sp['pitch_type'] == _pt]
                            if len(_g) == 0:
                                continue
                            _usage_pct = len(_g) / total_pitches * 100
                            _avg_velo  = _g['release_speed'].mean()
                            _avg_spin  = _g['release_spin_rate'].mean() if 'release_spin_rate' in _g.columns else np.nan
                            _avg_ivb   = _g['pfx_z'].mean() * 12 if 'pfx_z' in _g.columns else np.nan
                            _avg_hb    = _g['pfx_x'].mean() * 12 * _x_sign if 'pfx_x' in _g.columns else np.nan
                            _avg_sp    = _g['stuff_plus'].mean() if 'stuff_plus' in _g.columns else np.nan
                            _sp_color  = ('#3dcc7c' if not np.isnan(_avg_sp) and _avg_sp >= 110
                                          else '#f0c040' if not np.isnan(_avg_sp) and _avg_sp >= 95
                                          else '#f05050' if not np.isnan(_avg_sp) and _avg_sp < 85
                                          else '#8a94aa')
                            _arsenal_rows.append({
                                'pt':       _pt,
                                'name':     pt_name(_pt),
                                'color':    pt_color(_pt),
                                'usage':    _usage_pct,
                                'velo':     _avg_velo,
                                'spin':     _avg_spin,
                                'ivb':      _avg_ivb,
                                'hb':       _avg_hb,
                                'sp':       _avg_sp,
                                'sp_color': _sp_color,
                            })

                        def _fc(v, fmt='+.1f'):
                            """Format float cell; return — for NaN/None."""
                            try:
                                fv = float(v)
                                return '—' if np.isnan(fv) else f"{fv:{fmt}}"
                            except Exception:
                                return '—'

                        if _arsenal_rows:
                            _tbl_rows = ""
                            _td = "padding:6px 10px;text-align:center;"
                            for _r in sorted(_arsenal_rows, key=lambda x: -x['usage']):
                                _dot = (f"<span style='display:inline-block;width:9px;height:9px;"
                                        f"border-radius:50%;background:{_r['color']};"
                                        f"margin-right:5px;vertical-align:middle;'></span>")
                                # Pre-compute conditional cells — avoids Python ternary
                                # consuming adjacent string literals and dropping columns
                                _spin_td = (f"<td style='{_td}'>{_r['spin']:.0f}</td>"
                                            if not np.isnan(_r['spin'])
                                            else f"<td style='{_td}'>—</td>")
                                _sp_span = (f"<span style='color:{_r['sp_color']};font-weight:700;'>"
                                            f"{_r['sp']:.0f}</span>"
                                            if not np.isnan(_r['sp']) else '—')
                                _tbl_rows += (
                                    f"<tr>"
                                    f"<td style='padding:6px 10px;'>{_dot}{_r['name']}</td>"
                                    f"<td style='{_td}'>{_r['usage']:.1f}%</td>"
                                    f"<td style='{_td}'>{_r['velo']:.1f}</td>"
                                    + _spin_td +
                                    f"<td style='{_td}'>{_fc(_r['ivb'])}</td>"
                                    f"<td style='{_td}'>{_fc(_r['hb'])}</td>"
                                    f"<td style='{_td}'>{_sp_span}</td>"
                                    f"</tr>"
                                )
                            st.markdown(f"""
                            <div style="overflow-x:auto;">
                            <table style="width:100%;border-collapse:collapse;font-size:0.82rem;font-family:'Barlow',sans-serif;">
                              <thead>
                                <tr style="background:#1c2230;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">
                                  <th style="padding:7px 10px;text-align:left;">Pitch</th>
                                  <th style="padding:7px 10px;text-align:center;">Usage%</th>
                                  <th style="padding:7px 10px;text-align:center;">Avg Velo</th>
                                  <th style="padding:7px 10px;text-align:center;">Avg Spin</th>
                                  <th style="padding:7px 10px;text-align:center;">Avg IVB</th>
                                  <th style="padding:7px 10px;text-align:center;">Avg HB</th>
                                  <th style="padding:7px 10px;text-align:center;">Stuff+</th>
                                </tr>
                              </thead>
                              <tbody style="color:#e8eaf0;">
                                {_tbl_rows}
                              </tbody>
                            </table>
                            </div>
                            <div style="color:#505a70;font-size:0.7rem;margin-top:0.3rem;">
                              Stuff+ scale: 100 = MLB average · ≥110 elite · &lt;85 below average
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception:
                        pass

    # ── Pitcher Tab 3: Game Log ───────────────────────────────────────────────
    with ptab3:
        if not mlbam_id:
            st.info("No MLBAM ID — cannot load game log.")
        else:
            if 'df_sc' not in dir() or df_sc is None or len(df_sc) == 0:
                with st.spinner(f"Loading {season} Statcast data…"):
                    df_sc = get_sc_pitcher(mlbam_id, season)

            if df_sc is None or len(df_sc) == 0:
                st.warning(f"No Statcast data found for {season}.")
            else:
                df_log = df_sc.copy()
                df_log['game_date'] = pd.to_datetime(df_log['game_date'])

                # Determine opponent from inning context
                def get_opp(gdf):
                    row0 = gdf.iloc[0]
                    is_home = row0.get('inning_topbot', 'Top') == 'Bot'
                    opp  = row0.get('away_team', '?') if is_home else row0.get('home_team', '?')
                    loc  = 'vs' if is_home else '@'
                    return f"{loc} {opp}"

                def is_k(desc):
                    if pd.isna(desc):
                        return False
                    return str(desc) in ('strikeout', 'strikeout_double_play')

                def is_bb(desc):
                    if pd.isna(desc):
                        return False
                    return str(desc) in ('walk', 'intent_walk')

                # Group by game
                games = []
                for (gdate, gpk), gdf in df_log.groupby(['game_date', 'game_pk'], sort=False):
                    opp        = get_opp(gdf)
                    pitch_ct   = len(gdf)
                    avg_velo   = gdf['release_speed'].mean()
                    k_ct       = gdf['events'].apply(is_k).sum() if 'events' in gdf.columns else 0
                    bb_ct      = gdf['events'].apply(is_bb).sum() if 'events' in gdf.columns else 0
                    games.append({
                        'date':      gdate.strftime('%b %d, %Y'),
                        'opp':       opp,
                        'pitches':   pitch_ct,
                        'avg_velo':  avg_velo,
                        'k':         int(k_ct),
                        'bb':        int(bb_ct),
                        '_gdate':    gdate,
                        '_gpk':      gpk,
                        '_gdf':      gdf,
                    })

                games.sort(key=lambda g: g['_gdate'], reverse=True)

                # Display game log
                st.markdown(f"**{len(games)} game(s) in {season}**")

                for gi, game in enumerate(games):
                    label = (
                        f"📅 **{game['date']}**  {game['opp']}  ·  "
                        f"**{game['pitches']}** pitches  ·  "
                        f"Avg Velo: **{game['avg_velo']:.1f}** mph  ·  "
                        f"K: **{game['k']}**  BB: **{game['bb']}**"
                    )
                    with st.expander(label):
                        gdf = game['_gdf'].copy()
                        gdf = gdf.sort_values(['at_bat_number', 'pitch_number']) if \
                              ('at_bat_number' in gdf.columns and 'pitch_number' in gdf.columns) \
                              else gdf
                        gpk_val = game['_gpk']
                        _game_url = f"https://baseballsavant.mlb.com/game?game_pk={gpk_val}"

                        # Build pitch-by-pitch rows
                        pitch_rows = []
                        for _, pr in gdf.iterrows():
                            balls    = pr.get('balls',   '')
                            strikes  = pr.get('strikes',  '')
                            count_str = f"{balls}-{strikes}" if balls != '' else '—'
                            _ev_val   = pr.get('events', None)
                            _has_event = pd.notna(_ev_val) and str(_ev_val).strip() not in ('', 'nan', 'None')
                            _sv  = pr.get('sv_id',   None) if _has_event else None
                            _pid = pr.get('play_id', None) if _has_event else None
                            _vid_url = savant_url(_pid, _sv) if _has_event else ''
                            pitch_rows.append({
                                '#':      pr.get('pitch_number', ''),
                                'Type':   pt_name(pr.get('pitch_type', '')),
                                'Velo':   fmt_val(pr.get('release_speed'), 1, ' mph'),
                                'Spin':   fmt_val(pr.get('release_spin_rate'), 0, ' rpm'),
                                'IVB':    fmt_val(pr.get('pfx_z', np.nan) * 12 if pd.notna(pr.get('pfx_z')) else np.nan, 1, '"'),
                                'HB':     fmt_val(pr.get('pfx_x', np.nan) * 12 if pd.notna(pr.get('pfx_x')) else np.nan, 1, '"'),
                                'Count':  count_str,
                                'Result': str(pr.get('description', '—')).replace('_', ' ').title(),
                                'Event':  str(_ev_val).replace('_', ' ').title() if _has_event else '—',
                                '_vid':   _vid_url,
                            })

                        if pitch_rows:
                            pdf = pd.DataFrame(pitch_rows)
                            rows_html = ""
                            for _, pr_row in pdf.iterrows():
                                _vid = pr_row.get('_vid', '')
                                if _vid:
                                    vid_cell = (f"<a href='{_vid}' target='_blank' style='color:#3a9dff;"
                                                f"font-size:0.75rem;text-decoration:none;'>▶ Watch</a>")
                                else:
                                    vid_cell = "—"
                                rows_html += (
                                    f"<tr>"
                                    f"<td>{pr_row['#']}</td><td>{pr_row['Type']}</td>"
                                    f"<td>{pr_row['Velo']}</td><td>{pr_row['Spin']}</td>"
                                    f"<td>{pr_row['IVB']}</td><td>{pr_row['HB']}</td>"
                                    f"<td>{pr_row['Count']}</td><td>{pr_row['Result']}</td>"
                                    f"<td>{pr_row['Event']}</td>"
                                    f"<td>{vid_cell}</td>"
                                    f"</tr>"
                                )
                            st.markdown(f"""
                            <div style="overflow-x:auto;">
                            <table style="width:100%;border-collapse:collapse;font-size:0.78rem;font-family:'Barlow',sans-serif;">
                              <thead>
                                <tr style="background:#1c2230;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">
                                  <th style="padding:5px 8px;">#</th><th>Type</th><th>Velo</th>
                                  <th>Spin</th><th>IVB</th><th>HB</th><th>Count</th>
                                  <th>Result</th><th>Event</th><th>Video</th>
                                </tr>
                              </thead>
                              <tbody style="color:#e8eaf0;">
                                {rows_html}
                              </tbody>
                            </table>
                            </div>
                            """, unsafe_allow_html=True)

                        # Savant game link
                        st.markdown(
                            f"<a href='{_game_url}' target='_blank' "
                            f"style='color:#505a70;font-size:0.72rem;'>📊 Full Game on Baseball Savant ↗</a>",
                            unsafe_allow_html=True,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# HITTER VIEW
# ══════════════════════════════════════════════════════════════════════════════
if is_hitter:
    if is_twoway:
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-family:\"Barlow Condensed\",sans-serif;font-size:1.3rem;font-weight:700;"
            "color:#3a9dff;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;'>"
            "🏏 Hitter Profile</div>",
            unsafe_allow_html=True,
        )

    htab1, htab2, htab3, htab4 = st.tabs(["Career Stats", "Heat Maps", "Spray Chart", "Game Log"])

    # ── Hitter Tab 1: Career Stats ────────────────────────────────────────────
    with htab1:
        if fg_id is None:
            st.info("No FanGraphs ID found for this player. Career stats unavailable.")
        else:
            with st.spinner("Loading career batting stats…"):
                career_bat = get_career_batting(fg_id, max(debut_yr, 2008), CURRENT_YEAR)

            if career_bat is None or len(career_bat) == 0:
                st.info("No career batting stats found.")
            else:
                bat_cols   = ['Season', 'Team', 'PA', 'AVG', 'OBP', 'SLG', 'wRC+', 'HR', 'BB%', 'K%', 'WAR']
                bat_labels = ['Season', 'Team',  'PA', 'AVG', 'OBP', 'SLG', 'wRC+', 'HR', 'BB%', 'K%', 'WAR']
                available  = [c for c in bat_cols if c in career_bat.columns]
                av_labels  = [bat_labels[bat_cols.index(c)] for c in available]

                pct_cols_b  = [c for c in ['BB%', 'K%'] if c in available]
                dec3_bat    = [c for c in ['AVG', 'OBP', 'SLG'] if c in available]
                dec2_bat    = [c for c in ['WAR'] if c in available]
                int_bat     = [c for c in ['PA', 'HR', 'wRC+', 'Season'] if c in available]

                def _bat_cell(df_b, col):
                    col_data = df_b[col]
                    if col in pct_cols_b:
                        return [f"{v*100:.1f}%" if not pd.isna(v) else '—' for v in col_data]
                    elif col in dec3_bat:
                        return [f".{int(v*1000):03d}" if not pd.isna(v) else '—' for v in col_data]
                    elif col in dec2_bat:
                        return [f"{v:.1f}" if not pd.isna(v) else '—' for v in col_data]
                    elif col in int_bat:
                        return [str(int(v)) if not pd.isna(v) else '—' for v in col_data]
                    else:
                        return [str(v) if not pd.isna(v) else '—' for v in col_data]

                cell_vals = [_bat_cell(career_bat, c) if c in career_bat.columns
                             else ['—'] * len(career_bat) for c in available]
                n = len(career_bat)
                fill_colors = [['#141820' if i % 2 == 0 else '#1c2230' for i in range(n)]] * len(available)

                fig_tbl = go.Figure(go.Table(
                    header=dict(
                        values=av_labels,
                        fill_color='#1c2230', line_color='#2a3348',
                        font=dict(color='#f0c040', family='Barlow Condensed', size=13),
                        align='center', height=32,
                    ),
                    cells=dict(
                        values=cell_vals,
                        fill_color=fill_colors, line_color='#2a3348',
                        font=dict(color='#e8eaf0', family='Barlow', size=12),
                        align='center', height=28,
                    ),
                ))
                fig_tbl.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=max(120, 32 + 28 * n + 10),
                )
                st.plotly_chart(fig_tbl, use_container_width=True)

            # ── Career Statcast Stats (aggregate, multi-season) ──────────────
            if mlbam_id:
                sc_start = max(debut_yr, 2015)
                sc_end   = min(last_yr, CURRENT_YEAR)
                with st.spinner("Loading career Statcast stats…"):
                    career_sc_b = get_career_sc_batter(mlbam_id, sc_start, sc_end)

                if career_sc_b is not None and len(career_sc_b) > 0:
                    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-family:\"Barlow Condensed\",sans-serif;color:#3a9dff;"
                        "font-size:0.85rem;font-weight:700;letter-spacing:0.1em;"
                        "text-transform:uppercase;margin-bottom:0.5rem;'>Statcast Stats (Career)</div>",
                        unsafe_allow_html=True,
                    )
                    col_map_b = {
                        'est_ba':          'xBA',
                        'est_slg':         'xSLG',
                        'est_woba':        'xwOBA',
                        'woba':            'wOBA',
                        'avg_hit_speed':   'Avg EV',
                        'max_hit_speed':   'Max EV',
                        'barrels':         'Barrels',
                        'brl_percent':     'Brl%',
                        'ev95percent':     'HardHit%',
                        'anglesweetspotpercent': 'Sweet Spot%',
                        'pa':              'PA',
                        'bip':             'BIP',
                    }
                    career_sc_b_sorted = career_sc_b.copy()
                    if 'season' in career_sc_b_sorted.columns:
                        career_sc_b_sorted = career_sc_b_sorted.sort_values('season', ascending=False)
                    present_cols_b = ['season'] + [c for c in col_map_b if c in career_sc_b_sorted.columns]
                    headers_b = ['Season'] + [col_map_b[c] for c in present_cols_b[1:]]
                    th_s = "padding:5px 10px;white-space:nowrap;text-align:center;"
                    td_s = "padding:5px 10px;color:#e8eaf0;font-weight:600;text-align:center;white-space:nowrap;"
                    header_html_b = ''.join(f"<th style='{th_s}'>{h}</th>" for h in headers_b)
                    rows_html_b = ''
                    for _, rw in career_sc_b_sorted.iterrows():
                        cells = [f"<td style='{td_s}'>{int(rw['season']) if 'season' in rw else '—'}</td>"]
                        for c in present_cols_b[1:]:
                            pct_flag = c in ('brl_percent', 'ev95percent', 'anglesweetspotpercent')
                            dec_fmt = '.3f' if c in ('est_ba','est_slg','est_woba','woba') else '.1f'
                            cells.append(f"<td style='{td_s}'>{_fmt_sc(rw, c, dec_fmt, pct_flag)}</td>")
                        rows_html_b += f"<tr>{''.join(cells)}</tr>"
                    st.markdown(f"""
<div style="overflow-x:auto;">
<table style="border-collapse:collapse;font-size:0.78rem;font-family:'Barlow',sans-serif;
              background:#141820;border:1px solid #2a3348;border-radius:6px;min-width:100%;">
  <thead>
    <tr style="background:#1c2230;color:#3a9dff;font-family:'Barlow Condensed',sans-serif;
               font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">{header_html_b}</tr>
  </thead>
  <tbody>{rows_html_b}</tbody>
</table></div>""", unsafe_allow_html=True)
                else:
                    with st.spinner(f"Loading Statcast data for {season}…"):
                        _sc_b = get_sc_batter(mlbam_id, season)
                    _show_statcast_section(_sc_b, season)

    # ── Hitter Tab 2: Heat Maps ───────────────────────────────────────────────
    with htab2:
        if not mlbam_id:
            st.info("No MLBAM ID — cannot load Statcast data.")
        else:
            with st.spinner(f"Loading {season} Statcast data…"):
                df_bat = get_sc_batter(mlbam_id, season)

            if df_bat is None or len(df_bat) == 0:
                st.warning(f"No Statcast batting data found for {season}.")
            else:
                df_bat = df_bat.copy()

                hm_col1, hm_col2 = st.columns([1, 3])
                with hm_col1:
                    pitch_types_bat = ['All'] + sorted(df_bat['pitch_type'].dropna().unique().tolist())
                    pt_filter = st.selectbox("Pitch Type", pitch_types_bat, key="hm_pt")
                    metric_sel = st.radio(
                        "Metric",
                        ["xBA", "Whiff %", "Hard Hit %"],
                        key="hm_metric",
                    )
                    count_filter = st.selectbox(
                        "Count",
                        ["All", "Ahead (0-1, 0-2, 1-2)", "Behind (1-0, 2-0, 3-0, 2-1, 3-1, 3-2)",
                         "Even (0-0, 1-1, 2-2)"],
                        key="hm_count",
                    )

                df_hm = df_bat.copy()
                if pt_filter != 'All':
                    df_hm = df_hm[df_hm['pitch_type'] == pt_filter]

                # Count filter
                if count_filter.startswith("Ahead"):
                    ahead_counts = [(0,1),(0,2),(1,2)]
                    df_hm = df_hm[df_hm.apply(
                        lambda r: (r.get('balls',None), r.get('strikes',None)) in ahead_counts, axis=1)]
                elif count_filter.startswith("Behind"):
                    behind_counts = [(1,0),(2,0),(3,0),(2,1),(3,1),(3,2)]
                    df_hm = df_hm[df_hm.apply(
                        lambda r: (r.get('balls',None), r.get('strikes',None)) in behind_counts, axis=1)]
                elif count_filter.startswith("Even"):
                    even_counts = [(0,0),(1,1),(2,2)]
                    df_hm = df_hm[df_hm.apply(
                        lambda r: (r.get('balls',None), r.get('strikes',None)) in even_counts, axis=1)]

                with hm_col2:
                    if metric_sel == "xBA":
                        xba_col = 'estimated_ba_using_speedangle'
                        if xba_col in df_hm.columns:
                            fig_hm = build_zone_heatmap(df_hm, xba_col, 'xBA by Zone',
                                                        colorscale='RdYlGn', zrange=[0.0, 0.6])
                            st.plotly_chart(fig_hm, use_container_width=True)
                        else:
                            st.info("xBA column not available in this dataset.")

                    elif metric_sel == "Whiff %":
                        df_hm = df_hm.copy()
                        swing_descs = {'swinging_strike', 'swinging_strike_blocked',
                                       'foul', 'foul_tip', 'hit_into_play',
                                       'hit_into_play_no_out', 'hit_into_play_score',
                                       'foul_bunt', 'missed_bunt', 'bunt_foul_tip'}
                        whiff_descs = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
                        df_hm['_is_swing'] = df_hm['description'].isin(swing_descs)
                        df_hm['_is_whiff'] = df_hm['description'].isin(whiff_descs)
                        df_swings = df_hm[df_hm['_is_swing']].copy()
                        if len(df_swings) > 0:
                            df_swings['_whiff_rate'] = df_swings['_is_whiff'].astype(float)
                            fig_hm = build_zone_heatmap(df_swings, '_whiff_rate',
                                                        'Whiff % by Zone',
                                                        colorscale='RdYlGn_r', zrange=[0.0, 0.7])
                            st.plotly_chart(fig_hm, use_container_width=True)
                        else:
                            st.info("Insufficient swing data for whiff map.")

                    else:  # Hard Hit %
                        df_hm = df_hm.copy()
                        bip_mask = df_hm['launch_speed'].notna()
                        df_bip   = df_hm[bip_mask].copy()
                        if len(df_bip) > 0:
                            df_bip['_hard_hit'] = (df_bip['launch_speed'] >= 95).astype(float)
                            fig_hm = build_zone_heatmap(df_bip, '_hard_hit',
                                                        'Hard Hit % by Zone (EV ≥ 95 mph)',
                                                        colorscale='RdYlGn', zrange=[0.0, 0.8])
                            st.plotly_chart(fig_hm, use_container_width=True)
                        else:
                            st.info("No balls in play data found for hard hit map.")

    # ── Hitter Tab 3: Spray Chart ─────────────────────────────────────────────
    with htab3:
        if not mlbam_id:
            st.info("No MLBAM ID — cannot load Statcast data.")
        else:
            if 'df_bat' not in dir() or df_bat is None or len(df_bat) == 0:
                with st.spinner(f"Loading {season} Statcast data…"):
                    df_bat = get_sc_batter(mlbam_id, season)

            if df_bat is None or len(df_bat) == 0:
                st.warning(f"No Statcast batting data found for {season}.")
            else:
                spray_df = df_bat.dropna(subset=['hc_x', 'hc_y']).copy()
                spray_df = spray_df[spray_df['events'].notna()].copy()

                sc_f1, sc_f2, sc_f3 = st.columns(3)
                with sc_f1:
                    ev_filter = st.multiselect(
                        "Event type",
                        ['All Hits', 'Single', 'Double', 'Triple', 'Home Run', 'Outs'],
                        default=['All Hits', 'Outs'],
                        key="sc_ev",
                    )
                with sc_f2:
                    ph_filter = st.selectbox("Pitcher hand", ['All', 'RHP', 'LHP'], key="sc_ph")
                with sc_f3:
                    pt_filter_sc = st.selectbox(
                        "Pitch type",
                        ['All'] + sorted(spray_df['pitch_type'].dropna().unique().tolist()),
                        key="sc_pt",
                    )

                # Apply filters
                plot_df = spray_df.copy()
                if ph_filter == 'RHP':
                    plot_df = plot_df[plot_df.get('p_throws', pd.Series(['R']*len(plot_df))) == 'R']
                elif ph_filter == 'LHP':
                    plot_df = plot_df[plot_df.get('p_throws', pd.Series(['L']*len(plot_df))) == 'L']
                if pt_filter_sc != 'All':
                    plot_df = plot_df[plot_df['pitch_type'] == pt_filter_sc]

                # Filter by event type
                if 'All Hits' not in ev_filter and 'Outs' not in ev_filter:
                    # specific hit types
                    sel_events = set()
                    if 'Single'   in ev_filter: sel_events.add('single')
                    if 'Double'   in ev_filter: sel_events.add('double')
                    if 'Triple'   in ev_filter: sel_events.add('triple')
                    if 'Home Run' in ev_filter: sel_events.add('home_run')
                    plot_df = plot_df[plot_df['events'].isin(sel_events)]
                elif 'All Hits' in ev_filter and 'Outs' not in ev_filter:
                    hit_events = {'single', 'double', 'triple', 'home_run'}
                    plot_df = plot_df[plot_df['events'].isin(hit_events)]
                elif 'Outs' in ev_filter and 'All Hits' not in ev_filter:
                    out_incl = set()
                    if 'Single'   in ev_filter: out_incl.add('single')
                    if 'Double'   in ev_filter: out_incl.add('double')
                    if 'Triple'   in ev_filter: out_incl.add('triple')
                    if 'Home Run' in ev_filter: out_incl.add('home_run')
                    out_incl.update(OUT_EVENTS)
                    plot_df = plot_df[plot_df['events'].isin(out_incl)]
                # else: All Hits + Outs → show everything (no filter)

                if len(plot_df) == 0:
                    st.info("No balls in play match the selected filters.")
                else:
                    # Convert coordinates
                    plot_df['x_plot'] = plot_df['hc_x'] - 125.42
                    plot_df['y_plot'] = 198.27 - plot_df['hc_y']

                    # Color by event
                    def ev_col(ev):
                        for key, col in EV_COLOR.items():
                            if ev == key:
                                return col
                        return '#505a70'  # outs / other

                    def ev_disp(ev):
                        return str(ev).replace('_', ' ').title()

                    plot_df['_color']  = plot_df['events'].apply(ev_col)
                    plot_df['_label']  = plot_df['events'].apply(ev_disp)
                    plot_df['_hover']  = (
                        plot_df['_label'] + '<br>' +
                        plot_df['pitch_type'].fillna('').apply(pt_name)
                    )

                    fig_spray = go.Figure()

                    # Draw field
                    for tr in field_traces():
                        fig_spray.add_trace(tr)

                    # Plot batted balls grouped by event type for legend
                    for ev_key in ['home_run', 'triple', 'double', 'single']:
                        ev_sub = plot_df[plot_df['events'] == ev_key]
                        if len(ev_sub) == 0:
                            continue
                        fig_spray.add_trace(go.Scatter(
                            x=ev_sub['x_plot'], y=ev_sub['y_plot'],
                            mode='markers',
                            name=ev_disp(ev_key),
                            marker=dict(
                                color=EV_COLOR[ev_key], size=7, opacity=0.75,
                                line=dict(color='#0c0f14', width=0.5),
                            ),
                            hovertemplate='%{customdata}<extra></extra>',
                            customdata=ev_sub['_hover'],
                        ))

                    # Outs
                    outs_sub = plot_df[plot_df['events'].isin(OUT_EVENTS)]
                    if len(outs_sub) > 0:
                        fig_spray.add_trace(go.Scatter(
                            x=outs_sub['x_plot'], y=outs_sub['y_plot'],
                            mode='markers',
                            name='Out',
                            marker=dict(
                                color='#505a70', size=5, opacity=0.45,
                                line=dict(color='#0c0f14', width=0.3),
                            ),
                            hovertemplate='%{customdata}<extra></extra>',
                            customdata=outs_sub['_hover'],
                        ))

                    fig_spray.update_layout(
                        title=dict(
                            text=f'Spray Chart — {player_name} ({season})',
                            font=dict(color='#f0c040', size=13),
                        ),
                        xaxis=dict(
                            range=[-140, 140], showgrid=False, zeroline=False,
                            showticklabels=False, scaleanchor='y',
                        ),
                        yaxis=dict(
                            range=[-10, 195], showgrid=False, zeroline=False,
                            showticklabels=False,
                        ),
                        legend=dict(
                            font=dict(color='#8a94aa', size=11),
                            bgcolor='rgba(20,24,32,0.8)',
                            bordercolor='#2a3348', borderwidth=1,
                        ),
                        **_DL,
                        height=480,
                    )
                    st.plotly_chart(fig_spray, use_container_width=True)

                    # Summary counts
                    hit_events = {'single', 'double', 'triple', 'home_run'}
                    all_bip = spray_df
                    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
                    for col_sm, ev_key, ev_label in [
                        (sm1, 'single',   '1B'),
                        (sm2, 'double',   '2B'),
                        (sm3, 'triple',   '3B'),
                        (sm4, 'home_run', 'HR'),
                        (sm5, None,       'Total BIP'),
                    ]:
                        ct = len(all_bip[all_bip['events'] == ev_key]) if ev_key else len(all_bip)
                        color = EV_COLOR.get(ev_key, '#8a94aa') if ev_key else '#8a94aa'
                        col_sm.markdown(
                            f"<div style='text-align:center;background:#141820;border:1px solid #2a3348;"
                            f"border-radius:5px;padding:0.6rem;'>"
                            f"<div style='color:{color};font-family:\"Barlow Condensed\",sans-serif;"
                            f"font-size:1.8rem;font-weight:800;'>{ct}</div>"
                            f"<div style='color:#8a94aa;font-size:0.72rem;font-weight:600;"
                            f"text-transform:uppercase;letter-spacing:0.08em;'>{ev_label}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # ── Hitter Tab 4: Game Log ────────────────────────────────────────────────
    with htab4:
        if not mlbam_id:
            st.info("No MLBAM ID — cannot load game log.")
        else:
            if 'df_bat' not in dir() or df_bat is None or len(df_bat) == 0:
                with st.spinner(f"Loading {season} Statcast data…"):
                    df_bat = get_sc_batter(mlbam_id, season)

            if df_bat is None or len(df_bat) == 0:
                st.warning(f"No Statcast batting data found for {season}.")
            else:
                df_bat_log = df_bat.copy()
                df_bat_log['game_date'] = pd.to_datetime(df_bat_log['game_date'])

                def get_opp_bat(gdf):
                    row0 = gdf.iloc[0]
                    # Batter's team bats; if inning_topbot == 'Top', batter is away team
                    is_away = row0.get('inning_topbot', 'Top') == 'Top'
                    opp = row0.get('home_team', '?') if is_away else row0.get('away_team', '?')
                    loc = '@' if is_away else 'vs'
                    return f"{loc} {opp}"

                def count_pa_results(gdf):
                    """Count hits, HR, BB, K from events column."""
                    evs = gdf.dropna(subset=['events'])['events']
                    hits = evs.isin(['single', 'double', 'triple', 'home_run']).sum()
                    hrs  = (evs == 'home_run').sum()
                    bbs  = evs.isin(['walk', 'intent_walk']).sum()
                    ks   = evs.isin(['strikeout', 'strikeout_double_play']).sum()
                    # Count PAs as unique at_bat_number values
                    pa_ct = gdf['at_bat_number'].nunique() if 'at_bat_number' in gdf.columns else len(evs)
                    return pa_ct, int(hits), int(hrs), int(bbs), int(ks)

                bat_games = []
                for (gdate, gpk), gdf in df_bat_log.groupby(['game_date', 'game_pk'], sort=False):
                    opp = get_opp_bat(gdf)
                    pa_ct, hits, hrs, bbs, ks = count_pa_results(gdf)
                    bat_games.append({
                        'date':   gdate.strftime('%b %d, %Y'),
                        'opp':    opp,
                        'pa':     pa_ct,
                        'h':      hits,
                        'hr':     hrs,
                        'bb':     bbs,
                        'k':      ks,
                        '_gdate': gdate,
                        '_gpk':   gpk,
                        '_gdf':   gdf,
                    })
                bat_games.sort(key=lambda g: g['_gdate'], reverse=True)

                st.markdown(f"**{len(bat_games)} game(s) in {season}**")

                for gi, game in enumerate(bat_games):
                    label = (
                        f"📅 **{game['date']}**  {game['opp']}  ·  "
                        f"PA: **{game['pa']}**  H: **{game['h']}**  "
                        f"HR: **{game['hr']}**  BB: **{game['bb']}**  K: **{game['k']}**"
                    )
                    with st.expander(label):
                        gdf = game['_gdf'].copy()
                        gpk_val   = game['_gpk']
                        _bat_game_url = f"https://baseballsavant.mlb.com/game?game_pk={gpk_val}"

                        # Group by at-bat
                        if 'at_bat_number' in gdf.columns:
                            at_bats = []
                            for ab_num, ab_df in gdf.groupby('at_bat_number', sort=True):
                                ab_df = ab_df.sort_values('pitch_number') if 'pitch_number' in ab_df.columns else ab_df
                                ev_row  = ab_df[ab_df['events'].notna()]
                                event   = ev_row.iloc[-1]['events'] if len(ev_row) > 0 else ''
                                sv      = ev_row.iloc[-1].get('sv_id',   None) if len(ev_row) > 0 else None
                                play    = ev_row.iloc[-1].get('play_id', None) if len(ev_row) > 0 else None
                                lnk     = savant_url(play, sv)
                                pitches = len(ab_df)
                                # Pitch sequence
                                seq = ', '.join(
                                    [pt_name(r.get('pitch_type','?')) + f" ({r.get('description','?')})"
                                     for _, r in ab_df.iterrows()
                                     if pd.notna(r.get('pitch_type'))]
                                )
                                at_bats.append({
                                    'Pitches': pitches,
                                    'Result':  str(event).replace('_',' ').title() if event else '—',
                                    'Sequence': seq[:80] + ('…' if len(seq) > 80 else ''),
                                    '_link': lnk,
                                })

                            if at_bats:
                                rows_html = ""
                                for ab_row in at_bats:
                                    lnk = ab_row['_link']
                                    if lnk:
                                        vid = (f"<a href='{lnk}' target='_blank' style='color:#3a9dff;"
                                               f"font-size:0.75rem;text-decoration:none;'>▶ Watch</a>")
                                    else:
                                        vid = "—"
                                    rows_html += (
                                        f"<tr>"
                                        f"<td>{ab_row['Pitches']}</td>"
                                        f"<td style='color:#e8eaf0;font-weight:600;'>{ab_row['Result']}</td>"
                                        f"<td style='color:#505a70;font-size:0.72rem;'>{ab_row['Sequence']}</td>"
                                        f"<td>{vid}</td>"
                                        f"</tr>"
                                    )
                                st.markdown(f"""
                                <div style="overflow-x:auto;">
                                <table style="width:100%;border-collapse:collapse;font-size:0.78rem;font-family:'Barlow',sans-serif;">
                                  <thead>
                                    <tr style="background:#1c2230;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">
                                      <th style="padding:5px 8px;">Pitches</th>
                                      <th>Result</th><th>Pitch Sequence</th><th>Video</th>
                                    </tr>
                                  </thead>
                                  <tbody style="color:#e8eaf0;">
                                    {rows_html}
                                  </tbody>
                                </table>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("Detailed at-bat data not available for this game.")

                        st.markdown(
                            f"<a href='{_bat_game_url}' target='_blank' "
                            f"style='color:#505a70;font-size:0.72rem;'>📊 Full Game on Baseball Savant ↗</a>",
                            unsafe_allow_html=True,
                        )
