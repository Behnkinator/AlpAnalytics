import streamlit as st
import json, os
from utils import THEME_CSS

_POSTS_FILE = os.path.join(os.path.dirname(__file__), "posts.json")

def _recent_posts(n=3):
    try:
        with open(_POSTS_FILE) as f:
            posts = json.load(f)
        published = [p for p in posts if p.get('published', True)]
        return sorted(published, key=lambda p: p.get('date', ''), reverse=True)[:n]
    except Exception:
        return []

st.set_page_config(
    page_title="AlpAnalytics",
    layout="wide",
    page_icon="⚾",
    initial_sidebar_state="expanded"
)

st.markdown(THEME_CSS, unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<div class="sidebar-logo">⚾ AlpAnalytics</div>', unsafe_allow_html=True)

# ── Header row: title left, player search right ────────────────────────────────
header_col, search_col = st.columns([3, 1])

with header_col:
    st.markdown("""
    <div class="app-header" style="margin-bottom:0.5rem;">
        <p class="app-title">AlpAnalytics</p>
        <p class="app-subtitle">Pitch Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

with search_col:
    st.markdown("""
    <div style="background:#141820;border:1px solid #2a3348;border-top:3px solid #b47fff;
                border-radius:8px;padding:1rem 1.2rem;margin-top:0.6rem;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;font-weight:700;
                    color:#b47fff;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">
            🔍 Player Search
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.form("home_search_form", clear_on_submit=False):
        home_query = st.text_input(
            "Player",
            placeholder="e.g. Cole  or  Gerrit Cole",
            label_visibility="collapsed",
            key="home_query",
        )
        submitted = st.form_submit_button("Find Player →", use_container_width=True)
    if submitted:
        if home_query.strip():
            st.session_state['_query'] = home_query.strip()
            st.switch_page("pages/4_Player_Lookup.py")
        else:
            st.toast("Enter a player name to search.", icon="⚠️")

# ── App cards ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1.5rem;margin-top:1.5rem;">

  <div style="background:#141820;border:1px solid #2a3348;border-top:4px solid #f0c040;border-radius:8px;padding:2rem 1.8rem;">
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:800;color:#f0c040;letter-spacing:0.05em;">PITCHING REPORTS</div>
    <div style="color:#8a94aa;font-size:0.88rem;margin-top:0.6rem;line-height:1.6;">
      Pull live MLB game data or upload a Trackman / Statcast CSV. Get full game summaries, per-pitcher reports, movement charts, and Stuff+ grades for every pitch.
    </div>
    <div style="margin-top:1.5rem;display:flex;gap:0.75rem;flex-wrap:wrap;">
      <span style="background:rgba(240,192,64,0.12);color:#f0c040;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:0.3rem 0.7rem;border-radius:3px;">MLB Live</span>
      <span style="background:rgba(240,192,64,0.12);color:#f0c040;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:0.3rem 0.7rem;border-radius:3px;">CSV Upload</span>
      <span style="background:rgba(240,192,64,0.12);color:#f0c040;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:0.3rem 0.7rem;border-radius:3px;">Stuff+</span>
    </div>
  </div>

  <div style="background:#141820;border:1px solid #2a3348;border-top:4px solid #3dcc7c;border-radius:8px;padding:2rem 1.8rem;">
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:800;color:#3dcc7c;letter-spacing:0.05em;">STUFF+ CALCULATOR</div>
    <div style="color:#8a94aa;font-size:0.88rem;margin-top:0.6rem;line-height:1.6;">
      Score any pitch using the trained Stuff+ model. Input velocity, spin, movement, and release parameters to instantly get a Stuff+ grade benchmarked against all MLB pitches.
    </div>
    <div style="margin-top:1.5rem;display:flex;gap:0.75rem;flex-wrap:wrap;">
      <span style="background:rgba(61,204,124,0.12);color:#3dcc7c;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:0.3rem 0.7rem;border-radius:3px;">Single Pitch</span>
      <span style="background:rgba(61,204,124,0.12);color:#3dcc7c;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:0.3rem 0.7rem;border-radius:3px;">Instant Score</span>
      <span style="background:rgba(61,204,124,0.12);color:#3dcc7c;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:0.3rem 0.7rem;border-radius:3px;">Gauge Chart</span>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

# Navigation links
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/1_Pitching_Reports.py", label="Open Pitching Reports →", icon="📊")
with c2:
    st.page_link("pages/3_Stuff_Plus_Calculator.py", label="Open Stuff+ Calculator →", icon="🎯")

# ── Blog section ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;align-items:baseline;justify-content:space-between;
            border-bottom:1px solid #2a3348;padding-bottom:0.5rem;margin-bottom:1.2rem;">
  <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;font-weight:700;
               text-transform:uppercase;letter-spacing:0.1em;color:#f0c040;">Latest Posts</span>
</div>
""", unsafe_allow_html=True)

_posts = _recent_posts(3)

if not _posts:
    st.markdown(
        "<div style='color:#505a70;font-size:0.85rem;'>No posts yet. "
        "Head to the Blog to write your first post.</div>",
        unsafe_allow_html=True,
    )
else:
    _blog_cols = st.columns(len(_posts)) if len(_posts) > 1 else st.columns([1, 2])
    for _bi, _post in enumerate(_posts):
        _tags = [t.strip() for t in _post.get('tags', '').split(',') if t.strip()]
        _tag_html = ''.join(
            f"<span style='background:rgba(58,157,255,0.12);color:#3a9dff;font-size:0.66rem;"
            f"font-weight:700;letter-spacing:0.08em;text-transform:uppercase;"
            f"padding:0.18rem 0.5rem;border-radius:3px;margin-right:3px;'>{t}</span>"
            for t in _tags[:3]
        )
        _preview = _post.get('content', '')
        _preview = ' '.join(_preview.split())
        _preview = (_preview[:160] + '…') if len(_preview) > 160 else _preview
        with _blog_cols[_bi]:
            st.markdown(f"""
            <div style="background:#141820;border:1px solid #2a3348;border-top:3px solid #3a9dff;
                        border-radius:8px;padding:1.3rem 1.4rem;height:100%;">
              <div style="color:#505a70;font-size:0.72rem;font-weight:600;
                          letter-spacing:0.08em;margin-bottom:0.5rem;">{_post.get('date','')}</div>
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.25rem;font-weight:800;
                          color:#e8eaf0;line-height:1.2;margin-bottom:0.5rem;">{_post.get('title','')}</div>
              <div style="color:#8a94aa;font-size:0.82rem;line-height:1.55;margin-bottom:0.8rem;">
                {_preview}</div>
              <div>{_tag_html}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
st.page_link("pages/5_Blog.py", label="Go to Blog →", icon="✍️")
