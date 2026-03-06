"""Blog — AlpAnalytics. Write and read baseball analytics posts."""
import streamlit as st
import json
import os
import uuid
from datetime import date
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import THEME_CSS

POSTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "posts.json")

st.set_page_config(
    page_title="Blog · AlpAnalytics",
    layout="wide",
    page_icon="✍️",
    initial_sidebar_state="expanded",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-logo">⚾ AlpAnalytics</div>', unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <p class="app-title">Blog</p>
    <p class="app-subtitle">AlpAnalytics · Baseball Analytics · Pitch Intelligence</p>
</div>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_posts():
    if not os.path.exists(POSTS_FILE):
        return []
    try:
        with open(POSTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def _save_posts(posts):
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=2)

def _tag_chips(tags_str, color='#3a9dff'):
    tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    if not tags:
        return ''
    chips = ''.join(
        f"<span style='background:rgba(58,157,255,0.12);color:{color};font-size:0.68rem;"
        f"font-weight:700;letter-spacing:0.08em;text-transform:uppercase;"
        f"padding:0.2rem 0.55rem;border-radius:3px;margin-right:4px;'>{t}</span>"
        for t in tags
    )
    return chips


# ── Admin access via secret URL param: add ?admin=1 to the URL ─────────────────
_is_admin = st.query_params.get("admin") == "1"

# ── Session state ──────────────────────────────────────────────────────────────
if 'blog_mode' not in st.session_state:
    st.session_state.blog_mode = 'read'
if 'edit_id' not in st.session_state:
    st.session_state.edit_id = None

posts = _load_posts()
published = sorted(
    [p for p in posts if p.get('published', True)],
    key=lambda p: p.get('date', ''), reverse=True
)

# ── Mode toggle ────────────────────────────────────────────────────────────────
if _is_admin:
    _m1, _m2, _spacer = st.columns([1, 1, 5])
    with _m1:
        if st.button("✍️ Write New Post", use_container_width=True):
            st.session_state.blog_mode = 'write'
            st.session_state.edit_id = None
            st.rerun()
    with _m2:
        if st.button("📖 View All Posts", use_container_width=True):
            st.session_state.blog_mode = 'read'
            st.session_state.edit_id = None
            st.rerun()
else:
    if st.button("📖 View All Posts", use_container_width=True):
        st.session_state.blog_mode = 'read'
        st.session_state.edit_id = None
        st.rerun()

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WRITE MODE
# ══════════════════════════════════════════════════════════════════════════════
if _is_admin and st.session_state.blog_mode in ('write', 'edit'):
    is_edit = st.session_state.edit_id is not None
    existing = next((p for p in posts if p['id'] == st.session_state.edit_id), {}) if is_edit else {}

    st.markdown(
        f'<div class="section-header">{"Edit Post" if is_edit else "New Post"}</div>',
        unsafe_allow_html=True,
    )

    with st.form("post_form", clear_on_submit=False):
        title = st.text_input(
            "Title",
            value=existing.get('title', ''),
            placeholder="Post title",
        )
        col_date, col_tags = st.columns(2)
        with col_date:
            post_date = st.text_input(
                "Date",
                value=existing.get('date', str(date.today())),
                placeholder="YYYY-MM-DD",
            )
        with col_tags:
            tags = st.text_input(
                "Tags (comma-separated)",
                value=existing.get('tags', ''),
                placeholder="analytics, pitching, MLB",
            )

        content = st.text_area(
            "Content (Markdown supported)",
            value=existing.get('content', ''),
            height=420,
            placeholder=(
                "Write your post here. Markdown is fully supported.\n\n"
                "## Subheading\n\n"
                "Use **bold**, *italic*, `inline code`, and standard markdown.\n\n"
                "| Column | Value |\n|--------|-------|\n| ERA | 2.50 |"
            ),
        )

        published_flag = st.checkbox(
            "Published (visible on home page and blog)",
            value=existing.get('published', True),
        )

        _fs1, _fs2, _fs3 = st.columns([2, 1, 1])
        with _fs1:
            submitted = st.form_submit_button("💾 Save Post", use_container_width=True)
        with _fs2:
            preview_btn = st.form_submit_button("👁 Preview", use_container_width=True)
        with _fs3:
            cancel_btn = st.form_submit_button("Cancel", use_container_width=True)

    if cancel_btn:
        st.session_state.blog_mode = 'read'
        st.session_state.edit_id = None
        st.rerun()

    if submitted or preview_btn:
        if not title.strip():
            st.error("Title is required.")
        elif not content.strip():
            st.error("Content is required.")
        else:
            if submitted:
                if is_edit:
                    for p in posts:
                        if p['id'] == st.session_state.edit_id:
                            p['title']     = title.strip()
                            p['date']      = post_date.strip()
                            p['tags']      = tags.strip()
                            p['content']   = content.strip()
                            p['published'] = published_flag
                    _save_posts(posts)
                    st.success("Post updated.")
                else:
                    new_post = {
                        'id':        str(uuid.uuid4())[:8],
                        'title':     title.strip(),
                        'date':      post_date.strip(),
                        'tags':      tags.strip(),
                        'content':   content.strip(),
                        'published': published_flag,
                    }
                    posts.insert(0, new_post)
                    _save_posts(posts)
                    st.success("Post saved.")
                st.session_state.blog_mode = 'read'
                st.session_state.edit_id = None
                st.rerun()

            # Preview
            if preview_btn:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Preview</div>', unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-family:\"Barlow Condensed\",sans-serif;font-size:2.2rem;"
                    f"font-weight:800;color:#e8eaf0;line-height:1.1;margin-bottom:0.4rem;'>{title}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='color:#505a70;font-size:0.78rem;margin-bottom:0.8rem;'>"
                    f"{post_date} &nbsp;·&nbsp; {_tag_chips(tags)}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='border-top:1px solid #2a3348;margin-bottom:1.2rem;'></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(content)


# ══════════════════════════════════════════════════════════════════════════════
# READ MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    if not published:
        st.markdown(
            "<div style='color:#505a70;font-size:0.9rem;margin-top:1rem;'>"
            "No posts yet. Click <b>Write New Post</b> above to get started.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{len(published)} post(s)**")
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        for post in published:
            pid   = post.get('id', '')
            ptitle = post.get('title', 'Untitled')
            pdate  = post.get('date', '')
            ptags  = post.get('tags', '')
            pcontent = post.get('content', '')

            # Card header always visible
            with st.expander(f"**{ptitle}** — {pdate}", expanded=False):
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;flex-wrap:wrap;'>"
                    f"<span style='color:#505a70;font-size:0.78rem;'>{pdate}</span>"
                    f"<span style='color:#2a3348;'>·</span>"
                    f"{_tag_chips(ptags)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='border-top:1px solid #2a3348;margin-bottom:1rem;'></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(pcontent)

                # Edit / Delete row (admin only)
                if _is_admin:
                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                    _ea, _da, _spacer2 = st.columns([1, 1, 5])
                    with _ea:
                        if st.button("✏️ Edit", key=f"edit_{pid}"):
                            st.session_state.blog_mode = 'edit'
                            st.session_state.edit_id = pid
                            st.rerun()
                    with _da:
                        if st.button("🗑 Delete", key=f"del_{pid}"):
                            posts = [p for p in posts if p['id'] != pid]
                            _save_posts(posts)
                            st.rerun()
