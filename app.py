import streamlit as st
import fitz
import docx
import re
import plotly.graph_objects as go
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CareerLens AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 0 !important; max-width: 100% !important;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
[data-testid="stSidebar"] * {color: #cbd5e1 !important;}
[data-testid="stSidebarNav"] {display:none;}

/* Body background */
[data-testid="stAppViewContainer"] > .main {
    background: #f1f5f9;
}

/* Cards */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.card-sm {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Progress bar row */
.prog-row { margin: 8px 0; }
.prog-label {
    display: flex; justify-content: space-between;
    font-size: 13px; color: #374151; margin-bottom: 3px;
}
.prog-track {
    background: #e5e7eb; border-radius: 99px; height: 8px; width:100%;
}
.prog-fill {
    height: 8px; border-radius: 99px;
}

/* Skill tag */
.skill-matched {
    display:inline-flex; align-items:center; gap:5px;
    color: #16a34a; font-size:13px; padding: 2px 0;
}
.skill-missing {
    display:inline-flex; align-items:center; gap:5px;
    color: #dc2626; font-size:13px; padding: 2px 0;
}

/* Insight card */
.insight-row {
    display:flex; align-items:flex-start; gap:10px;
    padding: 10px 0; border-bottom: 1px solid #f1f5f9;
    cursor: pointer;
}
.insight-icon {
    width:32px; height:32px; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:16px; flex-shrink:0;
}
.insight-title { font-size:13px; font-weight:600; color:#1e293b; }
.insight-body  { font-size:12px; color:#64748b; margin-top:2px; }

/* Feedback card */
.fb-card {
    display:flex; align-items:flex-start; gap:10px;
    background:#fff; border-radius:10px; padding:12px 14px;
    margin-bottom:9px;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
.fb-dot {
    width:10px; height:10px; border-radius:50%;
    margin-top:4px; flex-shrink:0;
}
.fb-title { font-size:13px; font-weight:600; }
.fb-body  { font-size:12px; color:#64748b; margin-top:3px; }

/* Nav item */
.nav-item {
    display:flex; align-items:center; gap:10px;
    padding: 11px 16px; border-radius:10px;
    font-size:14px; font-weight:500; color:#94a3b8;
    cursor:pointer; margin: 2px 0;
}
.nav-item.active {
    background:#1e3a5f; color:#ffffff !important;
}
.nav-item:hover { background:#1e293b; color:#e2e8f0 !important; }

/* Analyze button */
.analyze-btn {
    background: #1e3a5f;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-size:14px;
    font-weight:600;
    cursor:pointer;
    white-space:nowrap;
}

/* Score circle */
.score-ring-wrap {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
}

/* Header bar */
.topbar {
    background:#fff;
    border-bottom: 1px solid #e5e7eb;
    padding: 14px 28px;
    display:flex; justify-content:space-between; align-items:center;
}
.topbar-title { font-size:22px; font-weight:700; color:#1e293b; }
.topbar-title span { color:#3b82f6; }
.topbar-sub { font-size:13px; color:#64748b; margin-top:2px; }
.topbar-user {
    display:flex; align-items:center; gap:8px;
    font-size:14px; color:#374151; font-weight:500;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed #c7d2fe;
    border-radius:12px;
    background:#f8faff;
    padding: 18px 20px;
    text-align:left;
}
.upload-title { font-size:15px; font-weight:700; color:#1e293b; }
.upload-sub   { font-size:12px; color:#94a3b8; }

/* Badge */
.badge-green {
    background:#dcfce7; color:#16a34a;
    font-size:12px; font-weight:600; padding:2px 10px; border-radius:99px;
}
.badge-yellow {
    background:#fef9c3; color:#b45309;
    font-size:12px; font-weight:600; padding:2px 10px; border-radius:99px;
}
.badge-red {
    background:#fee2e2; color:#dc2626;
    font-size:12px; font-weight:600; padding:2px 10px; border-radius:99px;
}

/* Section title */
.sec-title {
    font-size:15px; font-weight:700; color:#1e293b; margin-bottom:4px;
}
.sec-link {
    font-size:12px; color:#3b82f6; cursor:pointer; font-weight:500;
}

/* CTA card bottom sidebar */
.cta-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
    border-radius:14px;
    padding:20px;
    color:white;
    font-size:16px;
    font-weight:700;
    line-height:1.5;
    margin-top:16px;
    position:relative;
    overflow:hidden;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════
if "scores" not in st.session_state:
    st.session_state.scores = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = ""
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def extract_pdf(f):
    doc = fitz.open(stream=f.read(), filetype="pdf")
    return "\n".join(p.get_text() for p in doc)

def extract_docx(f):
    d = docx.Document(f)
    return "\n".join(p.text for p in d.paragraphs)

def extract_text(f):
    return extract_pdf(f) if f.name.lower().endswith(".pdf") else extract_docx(f)

TECH_SKILLS = [
    "python","java","javascript","typescript","c++","c#","go","rust","kotlin",
    "swift","ruby","php","scala","r","sql","bash","html","css","sass",
    "react","angular","vue","next.js","svelte","bootstrap","tailwind","jquery",
    "node.js","django","flask","fastapi","spring","express","rails","laravel",
    "machine learning","deep learning","nlp","computer vision","tensorflow",
    "pytorch","keras","scikit-learn","pandas","numpy","matplotlib","opencv",
    "aws","azure","gcp","docker","kubernetes","terraform","jenkins","ci/cd",
    "mysql","postgresql","mongodb","redis","oracle","cassandra","elasticsearch",
    "firebase","snowflake","bigquery","spark","hadoop","kafka","airflow","dbt",
    "git","linux","agile","scrum","rest","graphql","microservices","tableau",
    "power bi","excel","statistics","data analysis","communication",
    "problem solving","team collaboration","leadership","project management",
]

def get_skills(text):
    low = text.lower()
    return {s for s in TECH_SKILLS if re.search(r"\b" + re.escape(s) + r"\b", low)}

def get_exp(text):
    hits = re.findall(r"(\d+)\+?\s*years?", text.lower())
    return max((float(x) for x in hits), default=0)

def get_edu(text):
    low = text.lower()
    for kw, lvl in [("phd",4),("ph.d",4),("doctorate",4),
                    ("master",3),("mba",3),("m.s",3),("m.tech",3),
                    ("bachelor",2),("b.s",2),("b.tech",2),("b.e",2),
                    ("associate",1),("diploma",1)]:
        if kw in low: return lvl
    return 0

def cosine_sim(a, b):
    try:
        v = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        m = v.fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except:
        return 0.0

def compute(resume, jd):
    rs = get_skills(resume); js = get_skills(jd)
    matched = sorted(rs & js); missing = sorted(js - rs)
    sk = (len(matched)/len(js)*100) if js else 0
    re_ = get_exp(resume); je = get_exp(jd)
    ex = (min(re_/je,1)*100) if je else 85.0
    rl = get_edu(resume); jl = get_edu(jd)
    ed = (min(rl/jl,1)*100) if jl else 80.0
    ats = min(cosine_sim(resume, jd)*2.5*100, 100)
    ov = sk*0.40 + ex*0.25 + ed*0.15 + ats*0.20
    return dict(overall=round(ov,1), skills=round(sk,1),
                experience=round(ex,1), education=round(ed,1),
                ats=round(ats,1), matched=matched, missing=missing,
                resume_exp=re_, jd_exp=je)

def score_color(v):
    return "#22c55e" if v>=75 else ("#f59e0b" if v>=50 else "#ef4444")

def score_badge(v):
    if v>=75: return '<span class="badge-green">Good Match</span>'
    if v>=50: return '<span class="badge-yellow">Fair Match</span>'
    return '<span class="badge-red">Weak Match</span>'

def prog_bar(label, pct, color):
    return f"""
<div class="prog-row">
  <div class="prog-label"><span>{label}</span><span style="font-weight:600">{pct}%</span></div>
  <div class="prog-track"><div class="prog-fill" style="width:{pct}%;background:{color}"></div></div>
</div>"""

def donut_chart(value):
    color = score_color(value)
    fig = go.Figure(go.Pie(
        values=[value, 100-value],
        hole=0.72,
        marker_colors=[color, "#e5e7eb"],
        textinfo="none",
        hoverinfo="skip",
        sort=False,
    ))
    fig.add_annotation(
        text=f"<b>{value}%</b><br><span style='font-size:11px;color:#64748b'>Overall Match</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#1e293b"),
        align="center",
    )
    fig.update_layout(
        showlegend=False, margin=dict(l=0,r=0,t=0,b=0),
        height=200, width=200, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def radar_chart(sc):
    cats = ["Skills","Experience","Education","Keywords","Compatibility"]
    vals = [sc["skills"], sc["experience"], sc["education"], sc["ats"],
            round((sc["skills"]+sc["ats"])/2,1)]
    vc = vals + [vals[0]]; cc = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[100]*6, theta=cc, fill="toself",
        fillcolor="rgba(219,234,254,0.25)",
        line=dict(color="#bfdbfe", width=1, dash="dot"),
        name="Job Requirement", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=cc, fill="toself",
        fillcolor="rgba(59,130,246,0.18)",
        line=dict(color="#3b82f6", width=2),
        name="You",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(size=9,color="#94a3b8"),
                            gridcolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=11,color="#374151")),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    font=dict(size=11)),
        height=310, margin=dict(l=30,r=30,t=30,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def ai_suggestions(api_key, resume, jd, sc):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a senior career coach. Given this resume and job description, return EXACTLY 4 feedback items as a JSON array.
Each item: {{"type":"error|warning|info|tip","title":"short title","body":"one sentence fix"}}
Types map to: error=red(weak), warning=orange(missing achievement), info=blue(action verbs), tip=purple(keywords).

JD: {jd[:2000]}
Resume: {resume[:2000]}
Missing skills: {', '.join(sc['missing'][:10])}
ATS score: {sc['ats']}%

Return only valid JSON array, no markdown.
"""
    try:
        r = model.generate_content(prompt)
        import json
        text = r.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(text)
    except:
        return [
            {"type":"error",   "title":"Weak Bullet Points",
             "body":"Replace vague duties with quantified impact statements."},
            {"type":"warning", "title":"Missing Measurable Achievements",
             "body":"Add numbers or results to show your impact."},
            {"type":"info",    "title":"Use Stronger Action Verbs",
             "body":"Use: Increased, Implemented, Developed instead of 'did' or 'worked'."},
            {"type":"tip",     "title":"Missing Keywords",
             "body":f"Include JD keywords: {', '.join(sc['missing'][:5]) or 'analysis, reporting'}."},
        ]

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="padding:20px 16px 10px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <div style="background:#1d4ed8;border-radius:50%;width:36px;height:36px;
         display:flex;align-items:center;justify-content:center;font-size:18px;">🎯</div>
    <div>
      <div style="color:#ffffff;font-size:15px;font-weight:700;">CareerLens AI</div>
      <div style="color:#64748b;font-size:11px;">Resume & Job Readiness Analyzer</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    pages = [("🏠", "Dashboard"), ("📤", "Upload & Analyze"),
             ("📊", "Skill Gap Analysis"), ("💬", "Resume Feedback"),
             ("📖", "Learning Roadmap")]
    for icon, name in pages:
        active = "active" if st.session_state.page == name else ""
        if st.button(f"{icon}  {name}", key=f"nav_{name}",
                     use_container_width=True,
                     type="secondary" if not active else "primary"):
            st.session_state.page = name

    # Gemini key
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#475569;font-size:12px;padding:0 16px;margin-bottom:4px;">🔑 Gemini API Key</div>',
                unsafe_allow_html=True)
    api_key = st.text_input("", type="password", placeholder="AIza...",
                             key="api_key", label_visibility="collapsed")

    st.markdown("""
<div class="cta-card">
  <div style="font-size:22px;margin-bottom:6px;">🎯</div>
  Better Skills.<br>Better Matches.<br>A Brighter Career.
  <div style="position:absolute;bottom:0;right:0;opacity:0.15;font-size:60px;">🚀</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════
main = st.container()
with main:
    # ── Top bar ──────────────────────────────────────────────────
    st.markdown("""
<div class="topbar">
  <div>
    <div class="topbar-title">Welcome to <span>CareerLens AI</span></div>
    <div class="topbar-sub">Upload your resume and job description to get instant insights, match scores and personalized suggestions.</div>
  </div>
  <div class="topbar-user">
    <div style="background:#e0e7ff;border-radius:50%;width:34px;height:34px;
         display:flex;align-items:center;justify-content:center;font-size:16px;">👤</div>
    Career Seeker ▾
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # ── Three-column layout ───────────────────────────────────────
    left_col, mid_col, right_col = st.columns([2.3, 2.3, 1.4], gap="small")

    # ══════════════════════════════════════════════════════════════
    # LEFT COLUMN
    # ══════════════════════════════════════════════════════════════
    with left_col:
        # Upload + JD card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        up_c, or_c, jd_c = st.columns([1.1, 0.15, 1.1])

        with up_c:
            st.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
  <div style="font-size:22px;">📄</div>
  <div>
    <div class="upload-title">Upload Your Resume</div>
    <div class="upload-sub">PDF or DOCX (Max 5MB)</div>
  </div>
</div>""", unsafe_allow_html=True)
            uploaded = st.file_uploader("", type=["pdf","docx"],
                                         label_visibility="collapsed",
                                         key="resume_upload")
            if uploaded:
                st.markdown(f"""
<div style="display:flex;align-items:center;gap:6px;margin-top:4px;
     background:#f0fdf4;border-radius:8px;padding:6px 10px;">
  <span style="color:#16a34a;font-size:13px;font-weight:600;">✔</span>
  <span style="font-size:13px;color:#374151;">{uploaded.name}</span>
</div>""", unsafe_allow_html=True)

        with or_c:
            st.markdown('<div style="height:52px"></div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align:center;color:#94a3b8;font-size:12px;font-weight:600;">OR</div>',
                        unsafe_allow_html=True)

        with jd_c:
            st.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
  <div style="font-size:22px;">📋</div>
  <div>
    <div class="upload-title">Paste Job Description</div>
    <div class="upload-sub">Copy and paste the job description here</div>
  </div>
</div>""", unsafe_allow_html=True)
            jd_input = st.text_area("", height=90,
                                     placeholder="Paste the job description...",
                                     label_visibility="collapsed", key="jd_input")

        st.markdown("</div>", unsafe_allow_html=True)

        # Analyze button (full-width)
        analyze = st.button("Analyze Now  →", type="primary",
                             use_container_width=True, key="analyze_btn",
                             disabled=(not uploaded or not jd_input.strip()))

        if analyze and uploaded and jd_input.strip():
            with st.spinner("Analyzing…"):
                resume_text = extract_text(uploaded)
                st.session_state.scores = compute(resume_text, jd_input)
                st.session_state.resume_text = resume_text
                st.session_state.jd_text = jd_input
                if api_key:
                    st.session_state.fb_items = ai_suggestions(
                        api_key, resume_text, jd_input, st.session_state.scores)
                else:
                    sc = st.session_state.scores
                    st.session_state.fb_items = [
                        {"type":"error",   "title":"Weak Bullet Points",
                         "body":'\"Responsible for recruitment activities.\" → \"Screened 100+ resumes and coordinated interviews across multiple hiring requirements.\"'},
                        {"type":"warning", "title":"Missing Measurable Achievement",
                         "body":"Add numbers or results to show your impact."},
                        {"type":"info",    "title":"Use Stronger Action Verbs",
                         "body":"Consider using: Increased, Implemented, Developed, Managed instead of basic verbs like \"did\" or \"worked\"."},
                        {"type":"tip",     "title":"Missing Keywords",
                         "body":f"Include more JD keywords like: {', '.join(sc['missing'][:4]) or 'analysis, stakeholder, reporting, problem solving'}."},
                    ]

        sc = st.session_state.scores

        # ── Job Match Score card ──────────────────────────────────
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        d1, d2 = st.columns([1, 1.6])
        with d1:
            if sc:
                st.plotly_chart(donut_chart(sc["overall"]),
                                use_container_width=False, config={"displayModeBar":False})
            else:
                st.markdown("""
<div style="width:200px;height:200px;border-radius:50%;border:16px solid #e5e7eb;
     display:flex;align-items:center;justify-content:center;margin:auto;">
  <div style="text-align:center;">
    <div style="font-size:28px;font-weight:700;color:#94a3b8;">—</div>
    <div style="font-size:11px;color:#94a3b8;">Overall Match</div>
  </div>
</div>""", unsafe_allow_html=True)

        with d2:
            if sc:
                badge = score_badge(sc["overall"])
                st.markdown(f"""
<div style="margin-bottom:10px;">
  <span style="font-size:17px;font-weight:700;color:#1e293b;">Job Match Score</span>
  &nbsp;{badge}
</div>
<div style="font-size:13px;color:#64748b;margin-bottom:14px;">
  {"Your resume aligns well with the job description. A few skills and keywords can be improved for a stronger match." if sc["overall"]>=65 else "Your resume has significant gaps. Review the suggestions to improve your match."}
</div>
""", unsafe_allow_html=True)
                st.markdown(
                    prog_bar("🛠  Skills Match",     sc["skills"],     "#6366f1") +
                    prog_bar("💼  Experience Match", sc["experience"], "#8b5cf6") +
                    prog_bar("🎓  Education Match",  sc["education"],  "#22c55e") +
                    prog_bar("🔑  ATS / Keyword Match", sc["ats"],    "#f59e0b"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("""
<div style="font-size:17px;font-weight:700;color:#1e293b;margin-bottom:8px;">Job Match Score</div>
<div style="font-size:13px;color:#94a3b8;">Upload your resume and paste a job description, then click Analyze Now.</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Matched & Missing Skills ──────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sh1, sh2 = st.columns([1,1])
        with sh1:
            st.markdown('<div class="sec-title">👥 Matched &amp; Missing Skills</div>', unsafe_allow_html=True)
        with sh2:
            st.markdown('<div style="text-align:right"><span class="sec-link">View All</span></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown('<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:6px;">Matched Skills</div>', unsafe_allow_html=True)
            items = (st.session_state.scores["matched"][:6]
                     if sc else ["Excel","SQL","Power BI","Communication","Team Collaboration"])
            for s in items:
                st.markdown(f'<div class="skill-matched">✅ {s.title()}</div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:6px;">Missing Skills</div>', unsafe_allow_html=True)
            miss = (st.session_state.scores["missing"][:6]
                    if sc else ["Python","Tableau","Data Analysis","Statistics","Problem Solving"])
            for s in miss:
                st.markdown(f'<div class="skill-missing">❌ {s.title()}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # MIDDLE COLUMN
    # ══════════════════════════════════════════════════════════════
    with mid_col:
        # Profile Overview radar
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📡 Your Profile Overview</div>', unsafe_allow_html=True)
        if sc:
            st.plotly_chart(radar_chart(sc), use_container_width=True,
                            config={"displayModeBar":False})
        else:
            placeholder_sc = dict(skills=60,experience=55,education=70,ats=50)
            st.plotly_chart(radar_chart(placeholder_sc), use_container_width=True,
                            config={"displayModeBar":False})
            st.markdown('<div style="text-align:center;font-size:12px;color:#94a3b8;margin-top:-10px;">Analyze a resume to see your real scores</div>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # AI Resume Feedback
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fb1, fb2 = st.columns([1,1])
        with fb1:
            st.markdown('<div class="sec-title">💬 AI Resume Feedback</div>', unsafe_allow_html=True)
        with fb2:
            st.markdown('<div style="text-align:right"><span class="sec-link">View All</span></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        TYPE_MAP = {
            "error":   ("#ef4444", "#fef2f2"),
            "warning": ("#f59e0b", "#fffbeb"),
            "info":    ("#3b82f6", "#eff6ff"),
            "tip":     ("#8b5cf6", "#f5f3ff"),
        }
        fb_default = [
            {"type":"error",   "title":"Weak Bullet Point",
             "body":'\"Responsible for recruitment activities.\" → \"Screened 100+ resumes and coordinated interviews across multiple hiring requirements.\"'},
            {"type":"warning", "title":"Missing Measurable Achievement",
             "body":"Add numbers or results to show your impact."},
            {"type":"info",    "title":"Use Stronger Action Verbs",
             "body":"Consider using: Increased, Implemented, Developed, Managed instead of basic verbs like \"did\" or \"worked\"."},
            {"type":"tip",     "title":"Missing Keywords",
             "body":"Include more JD keywords like: analysis, stakeholder, reporting, problem solving."},
        ]
        fb_items = getattr(st.session_state, "fb_items", fb_default)
        for item in fb_items[:4]:
            color, bg = TYPE_MAP.get(item["type"], ("#6366f1","#eef2ff"))
            st.markdown(f"""
<div class="fb-card" style="border-left:4px solid {color};background:{bg};">
  <div>
    <div class="fb-title" style="color:{color};">{item["title"]}</div>
    <div class="fb-body">{item["body"]}</div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # RIGHT COLUMN
    # ══════════════════════════════════════════════════════════════
    with right_col:
        # Quick Insights
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;"><span style="font-size:18px;">💡</span><span class="sec-title" style="margin:0">Quick Insights</span></div>', unsafe_allow_html=True)

        if sc:
            insights = []
            edu_score = sc["education"]
            if edu_score >= 80:
                insights.append(("🎓","#fef9c3","You're a great fit in Education",
                                  f"({edu_score:.0f}%) Consider highlighting your relevant certifications more."))
            if sc["missing"]:
                top_miss = sc["missing"][0].title()
                insights.append(("📄","#fee2e2",f"{top_miss} is missing from your resume",
                                  "Add hands-on projects or certifications to strengthen your profile."))
            insights.append(("📋","#eff6ff",f"Your ATS score is {sc['ats']:.0f}%",
                              "Include more relevant keywords from the job description."))
        else:
            insights = [
                ("🎓","#fef9c3","You're a great fit in Education (90%)",
                 "Consider highlighting your relevant certifications more."),
                ("📄","#fee2e2","Python is missing from your resume",
                 "Add hands-on projects or certifications to strengthen your profile."),
                ("📋","#eff6ff","Your ATS score is 71%",
                 "Include more relevant keywords from the job description."),
            ]
        for icon, bg, title, body in insights:
            st.markdown(f"""
<div class="insight-row">
  <div class="insight-icon" style="background:{bg}">{icon}</div>
  <div>
    <div class="insight-title">{title}</div>
    <div class="insight-body">{body}</div>
  </div>
  <div style="margin-left:auto;color:#94a3b8;font-size:16px;">›</div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recommended Skills
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;"><span style="font-size:18px;">📖</span><span class="sec-title" style="margin:0">Recommended Skills to Learn</span></div>', unsafe_allow_html=True)
        skills_to_learn = (sc["missing"][:3] if sc and sc["missing"]
                           else ["Python","Statistics","Tableau"])
        for i, s in enumerate(skills_to_learn, 1):
            st.markdown(f'<div style="font-size:13px;color:#374151;padding:5px 0;border-bottom:1px solid #f1f5f9;">{i}.&nbsp;&nbsp;<b>{s.title()}</b></div>',
                        unsafe_allow_html=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.button("View Learning Roadmap  →", use_container_width=True,
                  type="primary", key="roadmap_btn")

        # Motivational
        st.markdown("""
<div style="margin-top:10px;padding:16px;background:#f8faff;border-radius:12px;
     text-align:center;position:relative;overflow:hidden;">
  <div style="font-size:26px;margin-bottom:4px;">🚀</div>
  <div style="font-size:13px;font-weight:600;color:#1e293b;line-height:1.5;">
    Small improvements<br>lead to big opportunities.
  </div>
  <div style="position:absolute;bottom:-10px;right:-10px;font-size:60px;opacity:0.08;">🏔</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
