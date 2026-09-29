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
/* ── Force light mode across the entire app ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
section.main > div,
.stApp {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
}

/* Kill any dark-mode overrides Streamlit injects */
[data-theme="dark"] { --background-color: #f1f5f9 !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 0 !important; max-width: 100% !important;}

/* All native Streamlit text elements → dark on light */
.stMarkdown, .stMarkdown p, .stMarkdown li,
p, li, span, label, div {
    color: #1e293b;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
    background-color: transparent;
}
[data-testid="stSidebarNav"] {display:none;}

/* Body background */
[data-testid="stAppViewContainer"] > .main {
    background: #f1f5f9 !important;
}

/* Cards — always white */
.card {
    background: #ffffff !important;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.card-sm {
    background: #ffffff !important;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Progress bar row */
.prog-row { margin: 8px 0; }
.prog-label {
    display: flex; justify-content: space-between;
    font-size: 13px; color: #1e293b !important; margin-bottom: 3px;
    font-weight: 500;
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
    color: #059669 !important; font-size:13px; font-weight:600; padding: 2px 0;
}
.skill-missing {
    display:inline-flex; align-items:center; gap:5px;
    color: #dc2626 !important; font-size:13px; font-weight:600; padding: 2px 0;
}

/* Insight card */
.insight-row {
    display:flex; align-items:flex-start; gap:10px;
    padding: 10px 0; border-bottom: 1px solid #e5e7eb;
    cursor: pointer;
}
.insight-icon {
    width:34px; height:34px; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:16px; flex-shrink:0;
}
.insight-title { font-size:13px; font-weight:700; color:#0f172a !important; }
.insight-body  { font-size:12px; color:#475569 !important; margin-top:2px; }

/* Feedback card */
.fb-card {
    display:flex; align-items:flex-start; gap:10px;
    background:#fff !important; border-radius:10px; padding:12px 14px;
    margin-bottom:9px;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
.fb-dot {
    width:10px; height:10px; border-radius:50%;
    margin-top:4px; flex-shrink:0;
}
.fb-title { font-size:13px; font-weight:700 !important; }
.fb-body  { font-size:12px; color:#475569 !important; margin-top:3px; }

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
    background:#ffffff !important;
    border-bottom: 1px solid #e5e7eb;
    padding: 14px 28px;
    display:flex; justify-content:space-between; align-items:center;
}
.topbar-title { font-size:22px; font-weight:800; color:#0f172a !important; }
.topbar-title span { color:#2563eb !important; font-weight:800; }
.topbar-sub { font-size:13px; color:#475569 !important; margin-top:2px; }
.topbar-user {
    display:flex; align-items:center; gap:8px;
    font-size:14px; color:#1e293b !important; font-weight:600;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed #c7d2fe;
    border-radius:12px;
    background:#f8faff;
    padding: 18px 20px;
    text-align:left;
}
.upload-title { font-size:15px; font-weight:800; color:#0f172a !important; }
.upload-sub   { font-size:12px; color:#64748b !important; }

/* Badge */
.badge-green {
    background:#d1fae5; color:#065f46 !important;
    font-size:12px; font-weight:700; padding:3px 12px; border-radius:99px;
}
.badge-yellow {
    background:#fef3c7; color:#92400e !important;
    font-size:12px; font-weight:700; padding:3px 12px; border-radius:99px;
}
.badge-red {
    background:#fee2e2; color:#991b1b !important;
    font-size:12px; font-weight:700; padding:3px 12px; border-radius:99px;
}

/* Section title — vibrant and always visible */
.sec-title {
    font-size:15px; font-weight:800; color:#0f172a !important; margin-bottom:4px;
}
.sec-link {
    font-size:12px; color:#2563eb !important; cursor:pointer; font-weight:600;
}

/* Matched / Missing column headings */
.col-head-matched { font-size:13px; font-weight:700; color:#065f46 !important; margin-bottom:6px; }
.col-head-missing { font-size:13px; font-weight:700; color:#991b1b !important; margin-bottom:6px; }

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
    try:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        if not text.strip():
            raise ValueError("No text found in PDF — it may be image-based.")
        return text
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")

def extract_docx(f):
    try:
        d = docx.Document(f)
        text = "\n".join(p.text for p in d.paragraphs)
        if not text.strip():
            raise ValueError("DOCX appears to be empty.")
        return text
    except Exception as e:
        raise ValueError(f"Could not read DOCX: {e}")

def extract_text(f):
    name = f.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf(f)
    elif name.endswith(".docx"):
        return extract_docx(f)
    raise ValueError("Unsupported file type. Please upload PDF or DOCX.")

# ── Skill bank — broad vocabulary so we catch what's actually in documents ──
TECH_SKILLS = [
    # Languages
    "python","java","javascript","typescript","c++","c#","golang","go","rust",
    "kotlin","swift","ruby","php","scala","r","sql","pl/sql","bash","shell",
    "html","css","sass","xml","json","yaml",
    # Frontend
    "react","angular","vue","next.js","nuxt","svelte","bootstrap","tailwind",
    "jquery","redux","webpack","vite",
    # Backend
    "node.js","django","flask","fastapi","spring","spring boot","express",
    "rails","laravel","asp.net",".net","nestjs","graphql","rest api","restful",
    # Data / ML / AI
    "machine learning","deep learning","nlp","natural language processing",
    "computer vision","tensorflow","pytorch","keras","scikit-learn","pandas",
    "numpy","matplotlib","seaborn","opencv","hugging face","langchain",
    "transformers","llm","generative ai","bert","gpt","rag","xgboost",
    "random forest","neural network","data science","feature engineering",
    # Cloud & DevOps
    "aws","azure","gcp","google cloud","docker","kubernetes","terraform",
    "ansible","jenkins","ci/cd","github actions","gitlab ci","helm",
    "prometheus","grafana","linux","unix","bash scripting",
    # Databases
    "mysql","postgresql","mongodb","redis","sqlite","oracle","cassandra",
    "elasticsearch","neo4j","dynamodb","firebase","snowflake","bigquery",
    "cosmos db","supabase",
    # Data Engineering
    "spark","hadoop","kafka","airflow","dbt","etl","data pipeline",
    "data warehouse","databricks","flink","hive","presto",
    # BI / Analytics
    "tableau","power bi","looker","qlik","excel","google sheets",
    "data visualization","data analysis","statistics","statistical analysis",
    "a/b testing","hypothesis testing","regression","forecasting",
    # Tools & Practices
    "git","github","gitlab","jira","confluence","agile","scrum","kanban",
    "microservices","api","swagger","postman","unit testing","tdd","bdd",
    # Soft skills
    "communication","problem solving","team collaboration","leadership",
    "project management","stakeholder management","critical thinking",
    "time management","presentation","client management",
]

def get_skills(text: str) -> set:
    low = text.lower()
    return {s for s in TECH_SKILLS if re.search(r"\b" + re.escape(s) + r"\b", low)}

def get_partial_skills(resume: str, jd_skills: set) -> list:
    """Skills mentioned in JD but only in passing/weakly in resume (1 mention)."""
    low = resume.lower()
    partial = []
    for s in jd_skills:
        pattern = r"\b" + re.escape(s) + r"\b"
        hits = re.findall(pattern, low)
        if len(hits) == 1:
            partial.append(s)
    return sorted(partial)

def get_exp(text: str) -> float:
    hits = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)", text.lower())
    return max((float(x) for x in hits), default=0)

def get_edu_level(text: str) -> int:
    low = text.lower()
    for kw, lvl in [
        ("phd",4),("ph.d",4),("doctorate",4),("d.phil",4),
        ("master",3),("mba",3),("m.s",3),("m.sc",3),("m.tech",3),("m.e",3),("meng",3),
        ("bachelor",2),("b.s",2),("b.sc",2),("b.tech",2),("b.e",2),("b.eng",2),("undergraduate",2),
        ("associate",1),("diploma",1),("certificate",1),
    ]:
        if kw in low:
            return lvl
    return 0

def get_edu_label(lvl: int) -> str:
    return {0:"Not specified",1:"Diploma/Certificate",2:"Bachelor's",3:"Master's/MBA",4:"PhD/Doctorate"}.get(lvl,"Not specified")

def cosine_sim(a: str, b: str) -> float:
    try:
        v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        m = v.fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except Exception:
        return 0.0

def keyword_overlap(resume: str, jd: str) -> float:
    """Word-level Jaccard overlap as a secondary ATS signal."""
    stop = {"the","a","an","and","or","is","in","to","of","for","with","that","this","as","at","by","on"}
    r_words = {w for w in re.findall(r"\b[a-z]{3,}\b", resume.lower()) if w not in stop}
    j_words = {w for w in re.findall(r"\b[a-z]{3,}\b", jd.lower()) if w not in stop}
    if not j_words:
        return 0.0
    return len(r_words & j_words) / len(j_words)

def compute(resume: str, jd: str) -> dict:
    # Skills
    r_skills = get_skills(resume)
    j_skills  = get_skills(jd)
    matched   = sorted(r_skills & j_skills)
    missing   = sorted(j_skills - r_skills)
    partial   = get_partial_skills(resume, j_skills - r_skills)  # weak mentions only
    # Remove from missing if they appear as partial
    missing   = [s for s in missing if s not in partial]

    sk_score  = (len(matched) / len(j_skills) * 100) if j_skills else 0.0

    # Experience
    r_exp = get_exp(resume)
    j_exp = get_exp(jd)
    if j_exp == 0:
        ex_score = 85.0          # JD doesn't specify — give benefit of doubt
    else:
        ex_score = min(r_exp / j_exp, 1.0) * 100

    # Education
    r_edu = get_edu_level(resume)
    j_edu = get_edu_level(jd)
    if j_edu == 0:
        ed_score = 80.0
    else:
        ed_score = min(r_edu / j_edu, 1.0) * 100

    # ATS — blend cosine + Jaccard for a more meaningful score
    cos  = cosine_sim(resume, jd)
    jac  = keyword_overlap(resume, jd)
    ats  = min((cos * 1.6 + jac * 0.9) * 100, 100)

    # Overall weighted
    overall = sk_score*0.40 + ex_score*0.25 + ed_score*0.15 + ats*0.20

    return dict(
        overall    = round(overall, 1),
        skills     = round(sk_score, 1),
        experience = round(ex_score, 1),
        education  = round(ed_score, 1),
        ats        = round(ats, 1),
        matched    = matched,
        missing    = missing,
        partial    = partial,
        resume_exp = r_exp,
        jd_exp     = j_exp,
        resume_edu = get_edu_label(r_edu),
        jd_edu     = get_edu_label(j_edu),
        total_jd_skills = len(j_skills),
    )

# ── Rule-based fallback feedback (uses actual sc data, no hardcoded strings) ─
def _rule_based_feedback(sc: dict, resume: str, jd: str) -> list:
    items = []
    # 1. Missing skills
    if sc["missing"]:
        top = ", ".join(sc["missing"][:4])
        items.append({"type":"tip","title":"Missing JD Keywords",
                      "body":f"Add these skills to your resume (found in JD but absent): {top}."})
    # 2. Partial / weak skills
    if sc["partial"]:
        top = ", ".join(sc["partial"][:3])
        items.append({"type":"warning","title":"Weakly Mentioned Skills",
                      "body":f"You mention {top} only once. Expand with concrete examples or projects."})
    # 3. ATS score issue
    if sc["ats"] < 55:
        items.append({"type":"error","title":f"Low ATS Score ({sc['ats']:.0f}%)",
                      "body":"Your resume text has low keyword overlap with the JD. Mirror the JD's exact phrasing for key requirements."})
    elif sc["ats"] < 70:
        items.append({"type":"warning","title":f"ATS Score Can Improve ({sc['ats']:.0f}%)",
                      "body":"Add more JD-specific terms naturally throughout your experience bullets."})
    # 4. Experience gap
    if sc["jd_exp"] > 0 and sc["resume_exp"] < sc["jd_exp"]:
        gap = sc["jd_exp"] - sc["resume_exp"]
        items.append({"type":"error","title":"Experience Gap Detected",
                      "body":f"JD requires ~{sc['jd_exp']:.0f} yrs; resume shows ~{sc['resume_exp']:.0f} yrs. Highlight relevant contract/freelance/project work to close the {gap:.0f}-yr gap."})
    # 5. Education gap
    if sc["resume_edu"] == "Not specified":
        items.append({"type":"warning","title":"Education Not Found in Resume",
                      "body":f"JD expects {sc['jd_edu']}. Add your degree/certification to the Education section."})
    # 6. Skills coverage praise
    if sc["skills"] >= 75:
        items.append({"type":"info","title":f"Strong Skills Match ({sc['skills']:.0f}%)",
                      "body":f"You match {len(sc['matched'])} of {sc['total_jd_skills']} required skills. Keep quantifying each skill with impact numbers."})
    elif not items:  # fallback if nothing triggered
        items.append({"type":"info","title":"Use Stronger Action Verbs",
                      "body":"Replace weak verbs (responsible for, helped, worked on) with impact verbs: Led, Delivered, Reduced, Increased, Automated."})
    return items[:4]

def ai_suggestions(api_key: str, resume: str, jd: str, sc: dict) -> list:
    import json
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a senior ATS specialist and career coach.
Analyze the RESUME against the JOB DESCRIPTION below and return EXACTLY 4 specific, actionable feedback items.

RULES:
- Every item must reference specific content from the resume or JD.
- Do NOT give generic advice. Be precise.
- For "Weak Bullet Points": quote an actual weak line from the resume and provide a rewritten version.
- For "Missing Keywords": list the exact missing terms from the JD.
- For "Missing Achievements": reference a specific role or responsibility and suggest a metric.
- Return ONLY a valid JSON array, no markdown, no explanation.

JSON schema for each item:
{{"type": "error|warning|info|tip", "title": "<short issue title>", "body": "<specific explanation with example>"}}

type meanings: error=critical issue, warning=improvement needed, info=best practice tip, tip=keyword/ATS tip

=== JOB DESCRIPTION (first 2500 chars) ===
{jd[:2500]}

=== RESUME (first 2500 chars) ===
{resume[:2500]}

=== COMPUTED ANALYSIS ===
Skills Match: {sc['skills']}% ({len(sc['matched'])} matched, {len(sc['missing'])} missing)
Missing skills: {', '.join(sc['missing'][:12]) or 'none'}
Partial/weak skills: {', '.join(sc['partial'][:8]) or 'none'}
ATS score: {sc['ats']}%
Experience: Resume={sc['resume_exp']}yrs, JD requires={sc['jd_exp']}yrs
Education: Resume={sc['resume_edu']}, JD expects={sc['jd_edu']}
"""
    try:
        r = model.generate_content(prompt)
        text = r.text.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        items = json.loads(text.strip())
        # Validate shape
        if isinstance(items, list) and all("type" in i and "title" in i and "body" in i for i in items):
            return items[:4]
        raise ValueError("Bad shape")
    except Exception:
        return _rule_based_feedback(sc, resume, jd)

# ══════════════════════════════════════════════════════════════════
# CHART & UI HELPERS
# ══════════════════════════════════════════════════════════════════
def score_color(v: float) -> str:
    return "#22c55e" if v >= 75 else ("#f59e0b" if v >= 50 else "#ef4444")

def score_badge(v: float) -> str:
    if v >= 75: return '<span class="badge-green">Good Match</span>'
    if v >= 50: return '<span class="badge-yellow">Fair Match</span>'
    return '<span class="badge-red">Weak Match</span>'

def prog_bar(label: str, pct: float, color: str) -> str:
    return f"""
<div class="prog-row">
  <div class="prog-label"><span>{label}</span><span style="font-weight:600">{pct}%</span></div>
  <div class="prog-track"><div class="prog-fill" style="width:{pct}%;background:{color}"></div></div>
</div>"""

def donut_chart(value: float):
    color = score_color(value)
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
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
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=200, width=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def radar_chart(sc: dict):
    cats = ["Skills", "Experience", "Education", "Keywords", "Compatibility"]
    vals = [
        sc["skills"], sc["experience"], sc["education"], sc["ats"],
        round((sc["skills"] + sc["ats"]) / 2, 1),
    ]
    vc = vals + [vals[0]]
    cc = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[100] * 6, theta=cc, fill="toself",
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
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=9, color="#94a3b8"),
                gridcolor="#e2e8f0",
            ),
            angularaxis=dict(tickfont=dict(size=11, color="#374151")),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, font=dict(size=11)),
        height=310,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

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
# MAIN CONTENT — page router
# ══════════════════════════════════════════════════════════════════
current_page = st.session_state.page

# ── Shared top-bar (shown on every page) ─────────────────────────
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

# ── Read scores once ──────────────────────────────────────────────
sc = st.session_state.scores

if current_page == "Dashboard":
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
  <span style="color:#059669;font-size:13px;font-weight:700;">✔</span>
  <span style="font-size:13px;color:#0f172a;font-weight:600;">{uploaded.name}</span>
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
            try:
                with st.spinner("Extracting and analyzing your resume…"):
                    resume_text = extract_text(uploaded)
                    if not jd_input.strip():
                        st.error("Please paste a Job Description before analyzing.")
                        st.stop()
                    _sc = compute(resume_text, jd_input)
                    st.session_state.scores = _sc
                    st.session_state.resume_text = resume_text
                    st.session_state.jd_text = jd_input
                    # Feedback — Gemini if key provided, else rule-based from actual sc
                    if api_key:
                        st.session_state.fb_items = ai_suggestions(
                            api_key, resume_text, jd_input, _sc)
                    else:
                        st.session_state.fb_items = _rule_based_feedback(_sc, resume_text, jd_input)
                st.rerun()
            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"Unexpected error during analysis: {e}")

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
  <span style="font-size:17px;font-weight:800;color:#0f172a;">Job Match Score</span>
  &nbsp;{badge}
</div>
<div style="font-size:13px;color:#475569;margin-bottom:14px;font-weight:500;">
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
<div style="font-size:17px;font-weight:800;color:#0f172a;margin-bottom:8px;">Job Match Score</div>
<div style="font-size:13px;color:#64748b;font-weight:500;">Upload your resume and paste a job description, then click Analyze Now.</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Matched & Missing Skills ──────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sh1, sh2 = st.columns([1,1])
        with sh1:
            st.markdown('<div class="sec-title">👥 Matched &amp; Missing Skills</div>', unsafe_allow_html=True)
        with sh2:
            if sc:
                st.markdown(f'<div style="text-align:right;font-size:12px;color:#64748b;">{len(sc["matched"])} matched · {len(sc["missing"])} missing · {len(sc["partial"])} partial</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        if sc:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown('<div class="col-head-matched">✅ Matched Skills</div>', unsafe_allow_html=True)
                if sc["matched"]:
                    for s in sc["matched"][:8]:
                        st.markdown(f'<div class="skill-matched">✅ {s.title()}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size:12px;color:#94a3b8;">No matching skills found in resume.</div>', unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="col-head-missing">❌ Missing Skills</div>', unsafe_allow_html=True)
                if sc["missing"]:
                    for s in sc["missing"][:8]:
                        st.markdown(f'<div class="skill-missing">❌ {s.title()}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size:12px;color:#059669;font-weight:600;">No missing skills — full coverage!</div>', unsafe_allow_html=True)
            # Partial skills row
            if sc["partial"]:
                st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:13px;font-weight:700;color:#b45309;margin-bottom:4px;">⚠️ Partially Mentioned Skills (strengthen these)</div>', unsafe_allow_html=True)
                partial_html = " ".join(
                    f'<span style="background:#fef3c7;color:#92400e;border-radius:99px;padding:2px 10px;font-size:12px;font-weight:600;margin:2px;display:inline-block;">~{s.title()}</span>'
                    for s in sc["partial"][:8]
                )
                st.markdown(partial_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:13px;color:#94a3b8;padding:8px 0;">Upload a resume and paste a JD, then click Analyze Now to see skills.</div>', unsafe_allow_html=True)
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
            st.markdown('<div style="text-align:center;font-size:12px;color:#64748b;font-weight:500;margin-top:-10px;">Analyze a resume to see your real scores</div>',
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
        fb_items = getattr(st.session_state, "fb_items", None)
        if fb_items:
            for item in fb_items[:4]:
                color, bg = TYPE_MAP.get(item["type"], ("#6366f1","#eef2ff"))
                st.markdown(f"""
<div class="fb-card" style="border-left:4px solid {color};background:{bg};">
  <div>
    <div class="fb-title" style="color:{color};">{item["title"]}</div>
    <div class="fb-body">{item["body"]}</div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:13px;color:#94a3b8;padding:8px 0;">Analyze a resume to see personalized AI feedback.</div>', unsafe_allow_html=True)
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
            # 1. Overall verdict
            if sc["overall"] >= 75:
                insights.append(("🎯","#dcfce7",f"Strong Match: {sc['overall']:.0f}% Overall",
                                  f"Your resume aligns well. Focus on the {len(sc['missing'])} missing skills to push higher."))
            elif sc["overall"] >= 50:
                insights.append(("🎯","#fef3c7",f"Fair Match: {sc['overall']:.0f}% Overall",
                                  f"Good foundation. Add missing skills and mirror JD keywords to improve."))
            else:
                insights.append(("🎯","#fee2e2",f"Weak Match: {sc['overall']:.0f}% Overall",
                                  "Significant gaps exist. Review missing skills and rewrite bullets to match the JD."))
            # 2. Skills breakdown
            if sc["skills"] >= 70:
                insights.append(("🛠️","#dcfce7",f"Skills: {sc['skills']:.0f}% — {len(sc['matched'])} matched",
                                  f"Strong skills alignment. {len(sc['partial'])} skills are weak — expand those with examples."))
            else:
                top_miss = ", ".join(sc["missing"][:3]) if sc["missing"] else "none"
                insights.append(("🛠️","#fee2e2",f"Skills Gap: {len(sc['missing'])} required skills missing",
                                  f"Key missing skills: {top_miss}. Add these to relevant experience bullets."))
            # 3. Experience
            if sc["jd_exp"] > 0:
                if sc["resume_exp"] >= sc["jd_exp"]:
                    insights.append(("💼","#dcfce7",f"Experience: {sc['resume_exp']:.0f} yrs (JD needs {sc['jd_exp']:.0f} yrs)",
                                      "You meet the experience requirement. Quantify impact in each role."))
                else:
                    insights.append(("💼","#fee2e2",f"Experience Gap: {sc['resume_exp']:.0f} yrs vs {sc['jd_exp']:.0f} yrs required",
                                      "Highlight freelance, contract, or project work to compensate for the gap."))
            # 4. ATS
            if sc["ats"] < 50:
                insights.append(("📋","#fee2e2",f"ATS Score: {sc['ats']:.0f}% — Needs Work",
                                  "Very low keyword overlap. Use exact phrases from the JD in your bullets."))
            elif sc["ats"] < 70:
                insights.append(("📋","#fef3c7",f"ATS Score: {sc['ats']:.0f}% — Can Improve",
                                  "Sprinkle more JD-specific terms throughout your experience section."))
            else:
                insights.append(("📋","#dcfce7",f"ATS Score: {sc['ats']:.0f}% — Good",
                                  "Good keyword coverage. Ensure each key skill appears in context."))
        else:
            insights = [
                ("🎯","#e0e7ff","Awaiting Analysis",
                 "Upload your resume and paste a JD, then click Analyze Now."),
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
        if sc and (sc["missing"] or sc["partial"]):
            # Prioritize: missing first, then partial
            priority = sc["missing"][:3] + [s for s in sc["partial"][:3] if s not in sc["missing"][:3]]
            priority = priority[:4]
            for i, s in enumerate(priority, 1):
                is_partial = s in sc["partial"] and s not in sc["missing"]
                label = f"~{s.title()} (strengthen)" if is_partial else s.title()
                st.markdown(
                    f'<div style="font-size:13px;color:#0f172a;font-weight:600;padding:6px 0;border-bottom:1px solid #e5e7eb;">'
                    f'{i}.&nbsp;&nbsp;<span style="color:#2563eb;font-weight:700;">{label}</span></div>',
                    unsafe_allow_html=True)
        elif sc:
            st.markdown('<div style="font-size:13px;color:#059669;font-weight:600;padding:8px 0;">🎉 You have great coverage of the required skills!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:13px;color:#94a3b8;padding:8px 0;">Analyze a resume to see skill recommendations.</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        if st.button("View Learning Roadmap  →", use_container_width=True,
                     type="primary", key="roadmap_btn"):
            st.session_state.page = "Learning Roadmap"
            st.rerun()

        # Motivational
        st.markdown("""
<div style="margin-top:10px;padding:16px;background:#eef2ff;border-radius:12px;
     text-align:center;position:relative;overflow:hidden;border:1px solid #c7d2fe;">
  <div style="font-size:26px;margin-bottom:4px;">🚀</div>
  <div style="font-size:13px;font-weight:700;color:#1e40af;line-height:1.5;">
    Small improvements<br>lead to big opportunities.
  </div>
  <div style="position:absolute;bottom:-10px;right:-10px;font-size:60px;opacity:0.08;">🏔</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LEARNING ROADMAP PAGE
# ══════════════════════════════════════════════════════════════════
elif current_page == "Learning Roadmap":
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    if not sc:
        st.markdown("""
<div class="card" style="text-align:center;padding:40px;">
  <div style="font-size:48px;margin-bottom:12px;">🗺️</div>
  <div style="font-size:18px;font-weight:700;color:#0f172a;margin-bottom:8px;">No Analysis Yet</div>
  <div style="font-size:14px;color:#64748b;">Upload your resume and paste a job description on the Dashboard, then click Analyze Now to generate your personalized learning roadmap.</div>
</div>""", unsafe_allow_html=True)
        if st.button("← Go to Dashboard", key="goto_dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
    else:
        # ── Header ────────────────────────────────────────────────
        st.markdown("""
<div class="card" style="background:linear-gradient(135deg,#1e3a5f,#1d4ed8)!important;">
  <div style="color:white;font-size:20px;font-weight:800;margin-bottom:4px;">🗺️ Your Personalized Learning Roadmap</div>
  <div style="color:#bfdbfe;font-size:13px;">Based on your resume analysis — focus on these skills to land your target role.</div>
</div>""", unsafe_allow_html=True)

        if st.button("← Back to Dashboard", key="back_dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        # ── Score summary strip ───────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "Overall Match",   sc["overall"],    score_color(sc["overall"])),
            (c2, "Skills Match",    sc["skills"],     score_color(sc["skills"])),
            (c3, "Experience",      sc["experience"], score_color(sc["experience"])),
            (c4, "ATS Score",       sc["ats"],        score_color(sc["ats"])),
        ]:
            col.markdown(f"""
<div class="card" style="text-align:center;padding:14px 8px;">
  <div style="font-size:22px;font-weight:800;color:{color};">{val}%</div>
  <div style="font-size:12px;color:#64748b;font-weight:500;margin-top:2px;">{label}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

        road_l, road_r = st.columns([1.6, 1], gap="large")

        with road_l:
            # ── Missing skills roadmap ────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">🎯 Skills to Learn (from JD)</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:12px;color:#64748b;margin-bottom:12px;">These skills are required in the job description but missing or weak in your resume. Learn them in priority order.</div>', unsafe_allow_html=True)

            priority_skills = sc["missing"][:6] + [s for s in sc["partial"][:4] if s not in sc["missing"]]

            if priority_skills:
                for i, skill in enumerate(priority_skills, 1):
                    is_partial = skill in sc["partial"] and skill not in sc["missing"]
                    status_label = "⚠️ Strengthen" if is_partial else "❌ Missing"
                    status_color = "#92400e" if is_partial else "#991b1b"
                    status_bg    = "#fef3c7" if is_partial else "#fee2e2"
                    priority_tag = "High Priority" if i <= 2 else ("Medium" if i <= 4 else "Nice to Have")
                    priority_color = "#dc2626" if i <= 2 else ("#f59e0b" if i <= 4 else "#22c55e")

                    # Resource links per skill type
                    skill_lower = skill.lower()
                    if any(k in skill_lower for k in ["python","java","javascript","typescript","sql","r","scala","go","rust","kotlin"]):
                        resources = "📚 freeCodeCamp · LeetCode · Codecademy"
                    elif any(k in skill_lower for k in ["machine learning","deep learning","nlp","tensorflow","pytorch","keras","scikit","data science","neural"]):
                        resources = "📚 fast.ai · Coursera ML · Kaggle"
                    elif any(k in skill_lower for k in ["aws","azure","gcp","docker","kubernetes","terraform","devops","ci/cd"]):
                        resources = "📚 AWS Free Tier · A Cloud Guru · Linux Foundation"
                    elif any(k in skill_lower for k in ["tableau","power bi","looker","visualization","analytics","statistics"]):
                        resources = "📚 Tableau Public · DataCamp · StatQuest"
                    elif any(k in skill_lower for k in ["react","angular","vue","node","django","flask","fastapi","spring"]):
                        resources = "📚 Official Docs · The Odin Project · freeCodeCamp"
                    elif any(k in skill_lower for k in ["spark","kafka","airflow","hadoop","dbt","snowflake","bigquery"]):
                        resources = "📚 DataBricks Academy · dbt Learn · Coursera"
                    else:
                        resources = "📚 LinkedIn Learning · Coursera · YouTube"

                    st.markdown(f"""
<div style="background:#f8faff;border-radius:12px;padding:14px 16px;margin-bottom:10px;
     border-left:4px solid {priority_color};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <div style="font-size:14px;font-weight:700;color:#0f172a;">{i}. {skill.title()}</div>
    <div style="display:flex;gap:6px;align-items:center;">
      <span style="background:{status_bg};color:{status_color};font-size:11px;font-weight:600;
            padding:2px 8px;border-radius:99px;">{status_label}</span>
      <span style="background:#f1f5f9;color:{priority_color};font-size:11px;font-weight:600;
            padding:2px 8px;border-radius:99px;">{priority_tag}</span>
    </div>
  </div>
  <div style="font-size:12px;color:#475569;">{resources}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div style="text-align:center;padding:24px;color:#059669;font-weight:600;font-size:14px;">
  🎉 You have full coverage of required skills! Focus on deepening expertise and adding portfolio projects.
</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Matched skills to highlight ───────────────────────
            if sc["matched"]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="sec-title">✅ Matched Skills — Showcase These</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:12px;color:#64748b;margin-bottom:10px;">You already have these. Make sure each one appears with a measurable achievement in your resume.</div>', unsafe_allow_html=True)
                tags = " ".join(
                    f'<span style="background:#d1fae5;color:#065f46;border-radius:99px;padding:4px 12px;font-size:12px;font-weight:600;margin:3px;display:inline-block;">✅ {s.title()}</span>'
                    for s in sc["matched"]
                )
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with road_r:
            # ── Radar chart ───────────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">📊 Your Profile vs Job Requirement</div>', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(sc), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Action plan ───────────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">📋 30-Day Action Plan</div>', unsafe_allow_html=True)
            weeks = [
                ("Week 1–2", "🔴", "Add missing skills to your resume with project examples"),
                ("Week 2–3", "🟡", "Take one online course for the top missing skill"),
                ("Week 3–4", "🟢", "Build a small project using the missing skill & add to portfolio"),
            ]
            for week, dot, action in weeks:
                st.markdown(f"""
<div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid #f1f5f9;">
  <span style="font-size:16px;flex-shrink:0;">{dot}</span>
  <div>
    <div style="font-size:12px;font-weight:700;color:#0f172a;">{week}</div>
    <div style="font-size:12px;color:#475569;margin-top:1px;">{action}</div>
  </div>
</div>""", unsafe_allow_html=True)
            # Personalized tip
            if sc["missing"]:
                top = sc["missing"][0].title()
                st.markdown(f'<div style="margin-top:10px;font-size:12px;color:#2563eb;font-weight:600;">💡 Start with: {top}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Stats ─────────────────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">📈 Your Gap Summary</div>', unsafe_allow_html=True)
            total = sc["total_jd_skills"] if sc["total_jd_skills"] > 0 else 1
            gap_pct = round(len(sc["missing"]) / total * 100)
            st.markdown(
                prog_bar(f"Skills Covered ({len(sc['matched'])}/{total})", sc["skills"], "#22c55e") +
                prog_bar(f"Skills Gap ({len(sc['missing'])} missing)", gap_pct, "#ef4444") +
                prog_bar("ATS Keyword Match", sc["ats"], "#f59e0b"),
                unsafe_allow_html=True,
            )
            # Experience row
            if sc["jd_exp"] > 0:
                exp_pct = min(round(sc["resume_exp"] / sc["jd_exp"] * 100), 100)
                st.markdown(prog_bar(f"Experience ({sc['resume_exp']:.0f}/{sc['jd_exp']:.0f} yrs)", exp_pct, "#6366f1"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
