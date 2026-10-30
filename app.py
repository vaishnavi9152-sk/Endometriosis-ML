import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EndoScan",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root{
    --bg1:#fdf8ff;
    --bg2:#f8fbff;
    --text:#241733;
    --muted:#7a6c8c;
    --line:#eadff5;
    --purple:#7b4bc4;
    --green:#2f8b6c;
    --amber:#a66b10;
    --red:#b94545;
    --shadow:0 12px 35px rgba(91,54,143,0.10);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"]{
    background:
        radial-gradient(circle at top left, #f6ebff 0%, transparent 28%),
        radial-gradient(circle at top right, #eaf4ff 0%, transparent 24%),
        linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%) !important;
    color:var(--text) !important;
    font-family:'Inter', sans-serif !important;
}

[data-testid="stHeader"], [data-testid="stSidebar"], #MainMenu, footer {
    display:none !important;
}

.block-container{
    max-width:1200px !important;
    padding-top:1.2rem !important;
    padding-bottom:3rem !important;
}

h1,h2,h3,h4{ font-family:'Cormorant Garamond', serif !important; color:var(--text); }

/* ── hero ── */
.hero{
    background:linear-gradient(135deg,rgba(255,255,255,0.86),rgba(255,255,255,0.72));
    border:1px solid rgba(234,223,245,0.9);
    border-radius:28px; padding:2.1rem 2rem;
    box-shadow:var(--shadow); backdrop-filter:blur(8px); margin-bottom:1.1rem;
}
.hero-title{
    font-family:'Cormorant Garamond',serif; font-size:3.1rem;
    line-height:1; margin-bottom:0.3rem; color:#2a173a;
}
.hero-sub{ font-size:1rem; color:var(--muted); line-height:1.75; max-width:820px; }
.stats-row{ display:flex; flex-wrap:wrap; gap:0.8rem; margin-top:1.3rem; }
.stat-chip{
    background:linear-gradient(135deg,#ffffff,#faf5ff);
    border:1px solid var(--line); border-radius:16px;
    padding:0.9rem 1rem; min-width:150px;
    box-shadow:0 6px 18px rgba(91,54,143,0.06);
}
.stat-num{ font-family:'Cormorant Garamond',serif; font-size:1.8rem; color:var(--purple); line-height:1; }
.stat-label{ font-size:0.76rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.12em; margin-top:0.25rem; }

/* ── panels ── */
.panel{
    background:linear-gradient(180deg,rgba(255,255,255,0.82),rgba(255,255,255,0.74));
    border:1px solid rgba(234,223,245,0.95); border-radius:24px;
    padding:1.4rem 1.3rem; box-shadow:var(--shadow);
    backdrop-filter:blur(8px); margin-bottom:1rem;
}
.section-tag{ font-size:0.72rem; color:var(--purple); text-transform:uppercase; letter-spacing:0.18em; font-weight:700; margin-bottom:0.35rem; }
.section-title{ font-family:'Cormorant Garamond',serif; font-size:2rem; color:#2a173a; margin-bottom:0.2rem; }
.section-sub{ color:var(--muted); font-size:0.94rem; line-height:1.7; margin-bottom:1rem; }

/* ── question blocks ── */
.question-wrap{
    background:rgba(255,255,255,0.75); border:1px solid #efe5f7;
    border-radius:18px; padding:1rem 1rem 0.8rem 1rem; margin-bottom:0.9rem;
}
.question-title{ font-weight:700; color:#29193a; margin-bottom:0.2rem; font-size:1rem; }
.question-sci{ color:var(--purple); font-size:0.78rem; font-style:italic; margin-bottom:0.2rem; }
.question-hint{ color:var(--muted); font-size:0.84rem; line-height:1.6; margin-bottom:0.35rem; }

/* ── Range badge — replaces "(0)" "(3)" etc ── */
.range-badge{
    display:inline-block;
    background:#f0eafc; border:1px solid #dcd0f5;
    border-radius:6px; padding:0.18rem 0.6rem;
    font-size:0.74rem; color:#5a3d9a; font-weight:600;
    letter-spacing:0.04em; margin-bottom:0.5rem;
}

/* ── CTA ── */
.cta-box{
    background:linear-gradient(135deg,#f7efff,#fff4fb);
    border:1px solid #e5d7f6; border-radius:22px;
    padding:1.35rem; text-align:center; margin-top:1rem;
}
.cta-title{ font-family:'Cormorant Garamond',serif; font-size:1.7rem; color:#2a173a; margin-bottom:0.2rem; }
.cta-sub{ color:var(--muted); font-size:0.92rem; line-height:1.7; }

/* ── result cards ── */
.result-card{
    background:rgba(255,255,255,0.9); border:1px solid var(--line);
    border-radius:22px; padding:1.2rem; box-shadow:var(--shadow); height:100%;
}
.result-label{
    font-size:0.72rem; text-transform:uppercase; letter-spacing:0.16em;
    color:var(--muted); margin-bottom:0.45rem; font-weight:700;
}
.result-big{ font-family:'Cormorant Garamond',serif; font-size:3rem; line-height:1; margin-bottom:0.35rem; }
.kpi-sub{ color:var(--muted); font-size:0.82rem; line-height:1.6; }

/* ── Risk pill badge (replaces red progress bar) ── */
.risk-pill{
    display:inline-block;
    padding:0.3rem 1rem; border-radius:999px;
    font-size:0.8rem; font-weight:700; letter-spacing:0.07em;
    text-transform:uppercase; margin-top:0.5rem; margin-bottom:0.4rem;
}

.soft-box{
    background:rgba(255,255,255,0.82); border:1px solid var(--line);
    border-radius:20px; padding:1rem; margin-bottom:1rem;
}

/* ── Flags ── */
.flag{
    display:flex; gap:0.7rem; align-items:flex-start;
    background:#fff7f7; border:1px solid #f0d4d4;
    border-radius:14px; padding:0.75rem 0.85rem;
    margin-bottom:0.5rem; color:#622222; font-size:0.88rem;
}
.flag-mark{
    width:10px; height:10px; border-radius:2px;
    background:#c84e4e; flex-shrink:0; margin-top:3px;
}

/* ── Patient "say" cards ── */
.say-card{
    border-left:4px solid var(--green); background:#f4fbf8;
    border-radius:0 14px 14px 0; padding:0.8rem 1rem;
    margin-bottom:0.5rem; color:#194334; font-size:0.9rem; line-height:1.7;
}
.note{
    background:#fffef7; border:1px solid #f1e5ba;
    border-radius:16px; padding:0.95rem 1rem;
    color:#6a5712; font-size:0.86rem; line-height:1.65;
}
.comorbid-note{
    background:#f5f0ff; border:1px solid #ddd0f5;
    border-radius:16px; padding:0.95rem 1rem;
    color:#3d2265; font-size:0.86rem; line-height:1.65; margin-bottom:0.8rem;
}
.disclaimer{
    background:rgba(255,255,255,0.82); border:1px solid var(--line);
    border-radius:18px; padding:0.95rem 1rem; color:var(--muted);
    font-size:0.82rem; line-height:1.7; text-align:center;
}

/* ── CLINICAL TAB — professional table layout ── */
.doc-header{
    border-bottom:2px solid #ede8f5; padding-bottom:0.8rem; margin-bottom:1.2rem;
}
.doc-title{
    font-family:'Cormorant Garamond',serif; font-size:1.6rem; color:#1e1030; margin-bottom:0.1rem;
}
.doc-subtitle{ font-size:0.8rem; color:#9b8fb5; letter-spacing:0.05em; }

.doc-section-head{
    font-size:0.68rem; text-transform:uppercase; letter-spacing:0.2em;
    color:#9b8fb5; font-weight:700; margin:1.2rem 0 0.5rem 0;
    border-bottom:1px solid #ede8f5; padding-bottom:0.3rem;
}
.doc-row{
    display:flex; justify-content:space-between; align-items:baseline;
    padding:0.48rem 0; border-bottom:1px solid #f5f1fb; font-size:0.87rem;
}
.doc-row:last-child{ border-bottom:none; }
.doc-row-label{ color:#6d6180; font-weight:500; }
.doc-row-val{ color:#1e1030; font-weight:600; text-align:right; }
.doc-box{
    background:#ffffff; border:1px solid #e4dff0; border-radius:14px;
    padding:1.1rem 1.2rem; margin-bottom:0.9rem;
    box-shadow:0 3px 12px rgba(91,54,143,0.05);
}
.doc-flag{
    display:flex; gap:0.6rem; align-items:flex-start;
    padding:0.5rem 0.7rem; margin-bottom:0.35rem;
    border-left:3px solid #9b6fc7;
    background:#faf7ff; border-radius:0 8px 8px 0;
    font-size:0.84rem; color:#2e1f50;
}
.doc-flag-mark{
    width:6px; height:6px; border-radius:50%;
    background:#9b6fc7; flex-shrink:0; margin-top:5px;
}
.doc-action{
    background:#f5f0ff; border:1px solid #ddd0f5; border-radius:12px;
    padding:1rem 1.1rem; color:#2a1a50; font-size:0.9rem;
    line-height:1.7; font-weight:500;
}
.doc-learn{
    background:#f8fbff; border:1px solid #d8e8f8; border-radius:12px;
    padding:0.9rem 1.1rem; color:#1a2f50; font-size:0.84rem; line-height:1.7;
    margin-top:0.8rem;
}
.doc-tier{
    display:inline-block; padding:0.28rem 0.85rem;
    border-radius:6px; font-size:0.75rem; font-weight:700;
    letter-spacing:0.1em; text-transform:uppercase;
}
.doc-cluster-desc{
    font-size:0.86rem; color:#3d2e55; line-height:1.75;
}

/* ── slider range labels ── */
.slider-range{
    font-size:0.74rem; color:#9b8fb5;
    display:flex; justify-content:space-between;
    margin-top:-0.6rem; margin-bottom:0.6rem; padding:0 0.1rem;
}
.dfl{ font-size:0.87rem; font-weight:600; color:#2e1f50; margin-top:0.8rem; margin-bottom:0.1rem; display:block; }
.dfs{ font-size:0.72rem; color:#9b6fc7; font-style:italic; margin-bottom:0.25rem; display:block; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"]{
    background:#f2ebfb !important; border-radius:16px !important;
    padding:5px !important; gap:6px !important;
}
.stTabs [data-baseweb="tab"]{
    border-radius:12px !important; padding:0.7rem 1.2rem !important;
    color:#6d6180 !important; font-weight:600 !important; background:transparent !important;
}
.stTabs [aria-selected="true"]{
    background:white !important; color:var(--purple) !important;
    box-shadow:0 4px 12px rgba(91,54,143,0.08) !important;
}

[data-testid="stNumberInput"] input, .stTextInput input{
    background:rgba(255,255,255,0.9) !important;
    border:1.5px solid #e7dcf3 !important; border-radius:14px !important; color:var(--text) !important;
}
.stRadio>label, .stCheckbox>label{ color:var(--text) !important; }
.stButton>button{
    width:100% !important; border:none !important; border-radius:999px !important;
    padding:0.95rem 1.4rem !important;
    background:linear-gradient(135deg,#7b4bc4,#d66aa8) !important;
    color:white !important; font-size:1rem !important; font-weight:700 !important;
    box-shadow:0 10px 24px rgba(123,75,196,0.28) !important;
}
.stButton>button:hover{
    transform:translateY(-1px);
    box-shadow:0 14px 30px rgba(123,75,196,0.34) !important;
}
[data-testid="stMetric"]{
    background:rgba(255,255,255,0.85) !important; border:1px solid var(--line) !important;
    border-radius:18px !important; padding:0.8rem !important;
}
hr{ border-color:#ecdef8 !important; }
.small-muted{ color:var(--muted); font-size:0.82rem; line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    m = {}
    try:
        m["endo"]    = joblib.load("models/diagnosis_model.joblib")
        m["endo_sc"] = joblib.load("models/diagnosis_scaler.joblib")
        with open("models/diagnosis_features.json", "r") as f:
            m["ef"] = json.load(f)
        m["endo_ok"] = True
    except Exception as e:
        m["endo_ok"] = False
        m["endo_err"] = str(e)

    try:
        m["km"]    = joblib.load("models/kmeans_model.joblib")
        m["km_sc"] = joblib.load("models/kmeans_scaler.joblib")
        m["km_ok"] = True
    except Exception as e:
        m["km_ok"] = False
        m["km_err"] = str(e)

    return m


MDL = load_models()


# =========================
# HELPERS
# =========================
def pred_endo(age, irr, pain, hormone, inf, bmi):
    if not MDL["endo_ok"]:
        return None, "not_loaded"
    X = pd.DataFrame([{
        "Age": age, "Menstrual_Irregularity": irr,
        "Chronic_Pain_Level": pain,
        "Hormone_Level_Abnormality": hormone,
        "Infertility": inf, "BMI": bmi
    }])[MDL["ef"]]
    prob = float(MDL["endo"].predict_proba(MDL["endo_sc"].transform(X))[0][1])
    return prob, "diagnosis_model.joblib"


def pred_cluster(dys, abd, icp, dyu, dyc, pd_, cst, pb, ms):
    if not MDL["km_ok"]:
        return None, "not_loaded"
    X = np.array([[dys, abd, icp, dyu, dyc, pd_, cst, pb, ms]])
    return int(MDL["km"].predict(MDL["km_sc"].transform(X))[0]), "kmeans_model.joblib"


def pdpct(score):
    pts = [(0,0),(16,22),(35,46),(50,74),(65,91),(70,100)]
    for i in range(len(pts)-1):
        if pts[i][0] <= score <= pts[i+1][0]:
            f = (score-pts[i][0]) / (pts[i+1][0]-pts[i][0]+1e-9)
            return int(pts[i][1] + f*(pts[i+1][1]-pts[i][1]))
    return 100


def risk_style(score):
    if score >= 60: return "#b94545", "Higher Risk",  "#fff7f7", "#f0d0d0"
    if score >= 35: return "#a66b10", "Moderate Risk", "#fffaf1", "#f0dfba"
    return "#2f8b6c", "Lower Risk", "#f4fbf8", "#cfe9df"


# =========================
# CLUSTER DATA  (ROXWELL n=699)
# =========================
CL = {
    0: {
        "name":   "Cluster 0 — High pain + severe psychological burden",
        "color":  "#b94545", "bg": "#fff8f8", "border": "#e6c7c7",
        "dep": 92, "anx": 90, "cw": 18, "n": 144,
        "avg_delay": "8–10 years",
        "desc":   "Severe physical pain and severe psychological burden co-occur in this group. "
                  "This profile is associated with the longest diagnostic delays in our study.",
        "missed": "Depression and anxiety in this subtype are often consequences of undiagnosed "
                  "endometriosis rather than separate primary diagnoses.",
        "action": "Urgent referral to a gynaecology specialist. Mental health evaluation is also recommended.",
    },
    1: {
        "name":   "Cluster 1 — Moderate pain, psychologically resilient",
        "color":  "#406ea8", "bg": "#f7faff", "border": "#d6e3f5",
        "dep": 18, "anx": 17, "cw": 9, "n": 233,
        "avg_delay": "5–7 years",
        "desc":   "Moderate symptoms with lower psychological burden. This may represent an "
                  "earlier stage where timely intervention can make a meaningful difference.",
        "missed": "Symptoms are frequently dismissed because the person appears emotionally well "
                  "and functional, despite significant physical pain.",
        "action": "Referral for further evaluation is recommended. Monitor for symptom progression.",
    },
    2: {
        "name":   "Cluster 2 — Widespread multi-site physical pain",
        "color":  "#b94545", "bg": "#fff8f8", "border": "#e6c7c7",
        "dep": 69, "anx": 67, "cw": 15, "n": 134,
        "avg_delay": "7–10 years",
        "desc":   "High physical pain burden across bowel, bladder, and pelvis. This profile is "
                  "more consistent with deep infiltrating endometriosis.",
        "missed": "Bowel and bladder symptoms are commonly attributed to IBS or recurrent UTI "
                  "before endometriosis is considered — the main source of delay in this group.",
        "action": "Urgent specialist referral. Multi-organ involvement warrants imaging evaluation.",
    },
    3: {
        "name":   "Cluster 3 — High pain, psychologically coping",
        "color":  "#a66b10", "bg": "#fffaf2", "border": "#eedebc",
        "dep": 12, "anx": 21, "cw": 5, "n": 188,
        "avg_delay": "6–8 years",
        "desc":   "Physical pain is high but the person appears to be coping psychologically. "
                  "This can lead to underreporting and delayed presentation.",
        "missed": "Pain may be normalised over many years. Ask specifically whether symptoms have "
                  "been worsening year on year.",
        "action": "Referral recommended. Focus on pain management and monitoring for worsening.",
    },
}

# =========================
# COMORBIDITIES LIST
# =========================
COMORBIDITIES = [
    "Irritable Bowel Syndrome (IBS)",
    "Interstitial Cystitis (painful bladder syndrome)",
    "Fibromyalgia",
    "Migraine / chronic headaches",
    "Autoimmune condition (e.g. lupus, thyroid disease, rheumatoid arthritis)",
    "Polycystic Ovary Syndrome (PCOS)",
    "Pelvic floor dysfunction",
    "Anxiety disorder (clinically diagnosed)",
    "Depression (clinically diagnosed)",
    "Adenomyosis",
    "Ovarian cysts (endometrioma)",
    "Chronic Fatigue Syndrome (ME/CFS)",
    "Asthma or allergies",
    "None of the above",
]


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">EndoScan</div>
    <div class="hero-sub">
        A research prototype for endometriosis symptom assessment and clinical decision support.
        Complete the sections below to generate a personalised risk estimate and patient subtype summary.
    </div>
    <div class="stats-row">
        <div class="stat-chip"><div class="stat-num">1 in 10</div><div class="stat-label">Women affected</div></div>
        <div class="stat-chip"><div class="stat-num">7–10 yrs</div><div class="stat-label">Average delay</div></div>
        <div class="stat-chip"><div class="stat-num">699</div><div class="stat-label">Real patients</div></div>
        <div class="stat-chip"><div class="stat-num">4</div><div class="stat-label">Subtypes found</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# CHANGE 1: Only show model errors, never show success message
if not MDL["endo_ok"]:
    st.error(f"Diagnosis model could not be loaded: {MDL.get('endo_err')}. Run src/03_endo_model.py first.")
if not MDL["km_ok"]:
    st.error(f"Clustering model could not be loaded: {MDL.get('km_err')}. Run src/02_clustering.py first.")

tab_p, tab_d = st.tabs(["Patient Assessment", "Clinical Decision Support"])


# =========================
# PATIENT TAB
# =========================
with tab_p:
    st.markdown("""
    <div class="panel">
        <div class="section-tag">Patient Flow</div>
        <div class="section-title">Endometriosis Symptom Assessment</div>
        <div class="section-sub">
            Answer each section honestly. The options describe the impact of each symptom —
            choose whichever best matches your experience. You do not need to know exact numbers.
            Your results will appear at the bottom.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CHANGE 2: helper now accepts range_note and shows it as a badge, no numbers in labels
    def prq(lbl, sci, hint, opts, key, range_note=None):
        range_html = (
            f'<div class="range-badge">Range: {range_note}</div>' if range_note else ""
        )
        st.markdown(
            f'<div class="question-wrap">'
            f'<div class="question-title">{lbl}</div>'
            f'<div class="question-sci">{sci}</div>'
            f'<div class="question-hint">{hint}</div>'
            f'{range_html}',
            unsafe_allow_html=True
        )
        value = opts[st.radio("", list(opts.keys()), key=key, label_visibility="collapsed")]
        st.markdown("</div>", unsafe_allow_html=True)
        return value

    # CHANGE 2: All labels are descriptive — no "(0)" "(3)" "(5)" etc.
    PAIN = {
        "No pain — completely normal":                       0,
        "Mild — uncomfortable but I kept going":             2,
        "Moderate — slowed me down significantly":           5,
        "Severe — difficult to leave the house or function": 7,
        "Unbearable — incapacitating, worst pain I have felt": 10,
    }
    FREQ = {
        "Never — no pain between periods":                   0,
        "Rarely — a few days per month":                     3,
        "Often — more than a week every month":              6,
        "Almost always — pain is present nearly every day":  9,
    }
    DISAB = {
        "No days — I functioned completely normally":        0,
        "A couple of days — mild disruption":               16,
        "Several days — missed work or normal activities":  35,
        "More than a week — significant time lost":         50,
        "Almost every day — pain dominates my month":       65,
    }
    BLEED = {
        "Light to normal — change protection every few hours":  0,
        "Heavy — change protection every couple of hours":      1,
        "Very heavy — soak through in under an hour":           2,
    }
    FAT = {
        "Normal energy — I feel like myself":               0,
        "Mild — needed more rest than usual":               1,
        "Significant — fatigue affects most of my day":     3,
        "Severe — I was not able to function normally":     4,
    }

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="panel"><div class="section-tag">Section 1</div><div class="section-title">Pain Assessment</div></div>', unsafe_allow_html=True)

        period_pain = prq(
            "Dysmenorrhoea (Menstrual / Period Pain)",
            "Pain occurring during menstruation — a primary endometriosis symptom.",
            "How strongly does your period pain affect your ability to function?",
            PAIN, "pp",
            range_note="No pain → Mild → Moderate → Severe → Unbearable"
        )

        pelvic_pain = prq(
            "Chronic Pelvic Pain (Lower Abdominal Pain)",
            "Persistent pain in the lower abdomen, even outside the menstrual period.",
            "How often do you experience pelvic pain?",
            FREQ, "pv",
            range_note="Never → Rarely → Often → Almost always"
        )

        bowel_pain = prq(
            "Dyschezia (Bowel / Rectal Pain)",
            "Pain during bowel movements, often worse around menstruation.",
            "How severe is your bowel pain when it occurs?",
            PAIN, "bp",
            range_note="No pain → Mild → Moderate → Severe → Unbearable"
        )

        bladder_pain = prq(
            "Dysuria (Bladder / Urination Pain)",
            "Pain, burning, or pressure when urinating.",
            "How severe is your bladder or urination pain?",
            PAIN, "du",
            range_note="No pain → Mild → Moderate → Severe → Unbearable"
        )

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Dyspareunia (Intercourse Pain)</div>'
            '<div class="question-sci">Pain during or after intercourse — commonly linked with deep infiltrating endometriosis.</div>'
            '<div class="question-hint">Choose the option that best matches your experience.</div>'
            '<div class="range-badge">Range: Not applicable → No pain → Moderate → Severe</div>',
            unsafe_allow_html=True
        )
        sx = st.radio("", [
            "Not applicable to me",
            "No pain",
            "Moderate pain — uncomfortable but continued",
            "Severe pain — had to stop or I avoid intercourse entirely",
        ], key="sx", label_visibility="collapsed")
        sx_map = {
            "Not applicable to me":                                (0, 0),
            "No pain":                                             (0, 1),
            "Moderate pain — uncomfortable but continued":         (4, 1),
            "Severe pain — had to stop or I avoid intercourse entirely": (8, 1),
        }
        intercourse_pain, intercourse_reported = sx_map[sx]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="section-tag">Section 2</div><div class="section-title">Systemic Symptoms</div></div>', unsafe_allow_html=True)

        fatigue = prq(
            "Chronic Fatigue (Persistent Tiredness)",
            "Persistent tiredness not fully relieved by rest — affects up to 88% of endo patients.",
            "How would you describe your fatigue from symptoms?",
            FAT, "fat",
            range_note="Normal → Mild → Significant → Severe"
        )

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Abdominal Bloating ("Endo Belly")</div>'
            '<div class="question-sci">Visible abdominal swelling associated with endometriosis — distinct from normal bloating.</div>'
            '<div class="question-hint">Do you experience severe abdominal swelling, especially around your period?</div>',
            unsafe_allow_html=True
        )
        bloating = st.radio("", [
            "No",
            "Yes — noticeable bloating",
            "Yes — severe, my abdomen visibly swells",
        ], key="bl", label_visibility="collapsed")
        bloating_val = 0 if bloating == "No" else 1
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Mittelschmerz (Mid-Cycle / Ovulation Pain)</div>'
            '<div class="question-sci">Pain occurring mid-cycle, approximately 2 weeks before menstruation — a specific endometriosis indicator.</div>'
            '<div class="question-hint">Do you notice pain around the middle of your cycle?</div>',
            unsafe_allow_html=True
        )
        ovul = st.radio("", [
            "No / not sure",
            "Yes, occasionally",
            "Yes, regularly every cycle",
        ], key="ov", label_visibility="collapsed")
        ovul_val = 0 if ovul == "No / not sure" else 1
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Referred Back or Leg Pain</div>'
            '<div class="question-sci">Pain radiating to the lower back, buttocks, or legs — may reflect deeper pelvic nerve involvement.</div>'
            '<div class="question-hint">Do you feel pain traveling to your lower back or legs during your period?</div>',
            unsafe_allow_html=True
        )
        back_leg = st.radio("", [
            "No",
            "Yes — pain travels into my lower back or legs",
        ], key="blp", horizontal=True, label_visibility="collapsed")
        back_leg_val = 1 if back_leg != "No" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="section-tag">Section 3</div><div class="section-title">Menstrual & Reproductive History</div></div>', unsafe_allow_html=True)

        heavy_bleed = prq(
            "Menorrhagia (Heavy Menstrual Bleeding)",
            "Abnormally heavy menstrual bleeding — may indicate endometrial involvement.",
            "How heavy is your menstrual bleeding?",
            BLEED, "hb",
            range_note="Light / Normal → Heavy → Very heavy"
        )

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Menstrual Irregularity</div>'
            '<div class="question-sci">Irregular cycle length, spotting between periods, or unpredictable timing.</div>'
            '<div class="question-hint">How predictable and regular are your periods?</div>',
            unsafe_allow_html=True
        )
        irreg = st.radio("", [
            "Yes — always regular and predictable",
            "Sometimes irregular",
            "Often irregular or completely unpredictable",
        ], key="irr", label_visibility="collapsed")
        irreg_val = 0 if irreg == "Yes — always regular and predictable" else 1
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Progressive Worsening of Symptoms Over Time</div>'
            '<div class="question-sci">Worsening year on year is the strongest clinical differentiator between endometriosis and primary dysmenorrhoea.</div>'
            '<div class="question-hint">Have your symptoms become worse over the years?</div>',
            unsafe_allow_html=True
        )
        worse = st.radio("", [
            "Staying about the same",
            "Getting slightly worse each year",
            "Getting significantly worse — much harder than when it started",
        ], key="ws", label_visibility="collapsed")
        worse_val = [
            "Staying about the same",
            "Getting slightly worse each year",
            "Getting significantly worse — much harder than when it started",
        ].index(worse)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="section-tag">Section 4</div><div class="section-title">Life Impact & Wellbeing</div></div>', unsafe_allow_html=True)

        pain_disability = prq(
            "Functional Disability (Pain Disability Index)",
            "How much pain interferes with daily life, work, and activities.",
            "In the last month, how many days did pain stop you from functioning normally?",
            DISAB, "pd",
            range_note="No days → A couple → Several days → More than a week → Almost every day"
        )

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Depression Screening (PHQ-2)</div>'
            '<div class="question-sci">A validated 2-item depression screen. 41% of endo patients screen positive vs 8% in the general population.</div>'
            '<div class="question-hint">In the last 2 weeks, have you felt down, hopeless, or lost interest in things you usually enjoy?</div>',
            unsafe_allow_html=True
        )
        dq = st.radio("", [
            "No — I have felt like myself",
            "Yes, sometimes — low mood on some days",
            "Yes, often — most days have felt difficult or hopeless",
        ], key="dq", label_visibility="collapsed")
        depression = {
            "No — I have felt like myself": 0,
            "Yes, sometimes — low mood on some days": 2,
            "Yes, often — most days have felt difficult or hopeless": 4,
        }[dq]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Anxiety Screening (GAD-2)</div>'
            '<div class="question-sci">A validated 2-item anxiety screen. 43% of endo patients screen positive vs 8% in the general population.</div>'
            '<div class="question-hint">In the last 2 weeks, have you felt anxious, on edge, or unable to control your worry?</div>',
            unsafe_allow_html=True
        )
        aq = st.radio("", [
            "No — I have felt calm and in control",
            "Yes, sometimes — more anxious than usual",
            "Yes, often — persistent worry I cannot control",
        ], key="aq", label_visibility="collapsed")
        anxiety = {
            "No — I have felt calm and in control": 0,
            "Yes, sometimes — more anxious than usual": 2,
            "Yes, often — persistent worry I cannot control": 4,
        }[aq]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="section-tag">Section 5</div><div class="section-title">Personal & Clinical Background</div></div>', unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca: age = st.number_input("Age", 14, 60, 28)
        with cb: bmi = st.number_input("BMI (leave as 23 if unknown)", 14.0, 45.0, 23.0, step=0.1)

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Hormone Level Abnormality</div>'
            '<div class="question-sci">Known hormonal dysregulation — e.g. abnormal oestrogen, progesterone, LH or FSH levels.</div>',
            unsafe_allow_html=True
        )
        ha = st.radio("", ["No / Not tested", "Yes"], key="ha", horizontal=True, label_visibility="collapsed")
        hormone_val = 1 if ha == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Infertility / Subfertility</div>'
            '<div class="question-sci">Difficulty conceiving is associated with endometriosis in up to 50% of infertile women.</div>',
            unsafe_allow_html=True
        )
        inf = st.radio("", ["No / Not applicable", "Yes"], key="inf", horizontal=True, label_visibility="collapsed")
        infertility_val = 1 if inf == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="question-wrap">'
            '<div class="question-title">Family History of Endometriosis</div>'
            '<div class="question-sci">First-degree relatives (mother, sister) with endometriosis confer approximately 7-fold increased risk.</div>',
            unsafe_allow_html=True
        )
        fhx = st.radio("", ["No / I do not know", "Yes"], key="fhx", horizontal=True, label_visibility="collapsed")
        family_hx_val = 1 if fhx == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

        # CHANGE 3: COMORBIDITIES SECTION
        st.markdown('<div class="panel"><div class="section-tag">Section 6</div><div class="section-title">Other Medical Conditions</div><div class="section-sub">Select any conditions you have been diagnosed with or are currently experiencing. These commonly co-occur with endometriosis and help build a fuller clinical picture.</div></div>', unsafe_allow_html=True)

        selected_comorbidities = st.multiselect(
            "Select all that apply:",
            COMORBIDITIES,
            key="comorbidities"
        )
        has_ibs        = "Irritable Bowel Syndrome (IBS)" in selected_comorbidities
        has_ic         = "Interstitial Cystitis (painful bladder syndrome)" in selected_comorbidities
        has_autoimmune = "Autoimmune condition (e.g. lupus, thyroid disease, rheumatoid arthritis)" in selected_comorbidities
        n_comorbid     = len([c for c in selected_comorbidities if c != "None of the above"])

    # ── CTA ──
    st.markdown("""
    <div class="cta-box">
        <div class="cta-title">Ready to generate your assessment?</div>
        <div class="cta-sub">
            Your answers will be processed through the trained diagnosis and clustering models.
            Results will appear below instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    run = st.button("Generate My Endometriosis Assessment", key="run", use_container_width=True)

    if run:
        st.session_state["show_results"] = True

    if st.session_state.get("show_results", False):
        st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
        components.html("""
        <script>
            setTimeout(function(){
                var el = window.parent.document.getElementById("results-anchor");
                if(el){ el.scrollIntoView({behavior:"smooth", block:"start"}); }
            }, 150);
        </script>""", height=0)

        st.markdown("""
        <div class="panel">
            <div class="section-tag">Results</div>
            <div class="section-title">Your Assessment Summary</div>
            <div class="section-sub">
                These results come from the trained ML models and the symptom pattern you entered.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CALL MODELS ──
        raw_prob, esrc = pred_endo(age, irreg_val, pelvic_pain, hormone_val, infertility_val, bmi)
        if raw_prob is None:
            st.error("Diagnosis model not loaded. Check your model files first.")
            st.stop()

        ep = raw_prob
        if family_hx_val:   ep = min(.97, ep * 1.35)
        if worse_val >= 2:  ep = min(.97, ep * 1.18)
        elif worse_val == 1:ep = min(.97, ep * 1.08)
        if ovul_val:        ep = min(.97, ep * 1.05)
        if has_ibs or has_ic:          ep = min(.97, ep * 1.05)
        if has_autoimmune:             ep = min(.97, ep * 1.04)
        rp = round(ep * 100)

        pb  = round(float(np.mean([period_pain, pelvic_pain, intercourse_pain, bladder_pain, bowel_pain])), 2)
        ms  = 1 if (bladder_pain >= 5 and bowel_pain >= 5) else 0
        cst = int(min(35, max(6, 12 + (depression/4)*10 + (period_pain/10)*8)))

        cl, ksrc = pred_cluster(period_pain, pelvic_pain, intercourse_pain,
                                 bladder_pain, bowel_pain, pain_disability, cst, pb, ms)
        if cl is None: cl = 1
        cp  = CL[cl]
        pct = pdpct(pain_disability)
        rc, rt, rbg, rbr = risk_style(rp)

        # ── METRIC CARDS ──
        m1, m2, m3 = st.columns(3)

        # CHANGE 5 & 6: No red progress bar — use pill badge instead
        with m1:
            st.markdown(f"""
            <div class="result-card" style="background:{rbg};border-color:{rbr};">
                <div class="result-label">Endometriosis Risk Score</div>
                <div class="result-big" style="color:{rc};">{rp}%</div>
                <div class="risk-pill" style="background:{rbr};color:{rc};">{rt}</div>
                <div class="kpi-sub" style="margin-top:0.5rem;">
                    Model output: {raw_prob:.3f} &rarr; adjusted: {ep:.3f}<br>
                    Source: <code>{esrc}</code>
                </div>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Patient Subtype</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:1.55rem;
                            color:{cp['color']};line-height:1.25;">{cp['name']}</div>
                <div class="kpi-sub" style="margin-top:0.5rem;">
                    Source: <code>{ksrc}</code><br>
                    Derived from real patient clustering (n={cp['n']})
                </div>
            </div>""", unsafe_allow_html=True)

        with m3:
            comorbid_note = f"<br>{n_comorbid} comorbidities reported." if n_comorbid > 0 else ""
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Pain Disability Percentile</div>
                <div class="result-big" style="color:#7b4bc4;">
                    {pct}<span style="font-size:1.4rem;">th</span>
                </div>
                <div class="kpi-sub">
                    Compared against endometriosis patient reference population.{comorbid_note}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── CLUSTER DETAIL ──
        st.markdown(f"""
        <div class="soft-box" style="background:{cp['bg']};border-color:{cp['border']};">
            <div class="result-label">Subtype Interpretation (ROXWELL Study, n=699)</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.75rem;
                        color:{cp['color']};margin-bottom:0.45rem;">{cp['name']}</div>
            <div class="small-muted" style="color:#4f415f;">{cp['desc']}</div>
        </div>""", unsafe_allow_html=True)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Major depression in subtype", f"{cp['dep']}%", delta="vs 8% general pop")
        sc2.metric("Anxiety in subtype",          f"{cp['anx']}%", delta="vs 8% general pop")
        sc3.metric("Unable to work",              f"{cp['cw']}%")
        sc4.metric("Avg. diagnostic delay",       cp["avg_delay"])

        # ── FLAGS ──
        flags = []
        if period_pain >= 7:   flags.append("Severe dysmenorrhoea — pain rated severe or unbearable.")
        if bowel_pain >= 6:    flags.append("Significant dyschezia — possible deep infiltrating disease.")
        if bladder_pain >= 6:  flags.append("Severe dysuria — may indicate bladder endometriosis.")
        if back_leg_val:       flags.append("Referred lower back or leg pain — possible deeper pelvic involvement.")
        if heavy_bleed >= 2:   flags.append("Very heavy bleeding — soaking through protection in under one hour.")
        if family_hx_val:      flags.append("Family history present — approximately 7-fold inherited risk.")
        if worse_val >= 1:     flags.append("Symptoms are progressively worsening over time.")
        if ovul_val:           flags.append("Mid-cycle ovulation pain reported — specific endometriosis indicator.")
        if fatigue >= 3:       flags.append("Severe fatigue associated with symptom burden.")
        if depression >= 4:    flags.append("PHQ-2 positive for major depressive symptom pattern.")
        if has_ibs:            flags.append("IBS reported — may co-occur with or mask bowel endometriosis.")
        if has_ic:             flags.append("Interstitial cystitis reported — may co-exist with bladder endometriosis.")
        if has_autoimmune:     flags.append("Autoimmune condition present — associated with higher disease severity.")
        if n_comorbid >= 3:    flags.append(f"{n_comorbid} comorbidities reported — consider multidisciplinary review.")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="panel"><div class="section-tag">Clinical Signals</div><div class="section-title">Important Red Flags</div></div>', unsafe_allow_html=True)
            if flags:
                for fl in flags:
                    # CHANGE 7: square mark, no emoji
                    st.markdown(f'<div class="flag"><div class="flag-mark"></div><div>{fl}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="soft-box">No major red flags detected at the current response levels.</div>', unsafe_allow_html=True)

            # Comorbidities summary
            if n_comorbid > 0:
                items_html = "".join(
                    f"<li>{c}</li>"
                    for c in selected_comorbidities if c != "None of the above"
                )
                st.markdown(f"""
                <div class="comorbid-note">
                    <strong>Reported comorbidities:</strong>
                    <ul style="margin:0.4rem 0 0 1rem;padding:0;line-height:1.9;">{items_html}</ul>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="soft-box" style="background:{cp['bg']};border-color:{cp['border']};">
                <div class="result-label">Recommended Action</div>
                <div style="font-weight:700;color:{cp['color']};line-height:1.8;">{cp['action']}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="note"><strong>Commonly missed in this subtype:</strong> {cp["missed"]}</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel"><div class="section-tag">Symptom Profile</div><div class="section-title">Pain & Fatigue Overview</div></div>', unsafe_allow_html=True)

            fig_p, ax_p = plt.subplots(figsize=(6, 3.4))
            fig_p.patch.set_facecolor("white")
            ax_p.set_facecolor("white")
            ns = ["Menstrual\npain","Pelvic\npain","Bowel\npain","Bladder\npain","Intercourse\npain","Fatigue"]
            vs = [period_pain, pelvic_pain, bowel_pain, bladder_pain, intercourse_pain, fatigue*2.5]
            colors_bar = ["#b94545" if v>=7 else "#a66b10" if v>=4 else "#2f8b6c" for v in vs]
            bars = ax_p.barh(ns, vs, height=0.58, color=colors_bar)
            ax_p.set_xlim(0, 10)
            ax_p.axvline(7, linestyle="--", alpha=0.25, linewidth=1)
            ax_p.axvline(4, linestyle="--", alpha=0.25, linewidth=1)
            ax_p.set_xlabel("Score (0 = none, 10 = worst)", fontsize=9)
            for bar, v in zip(bars, vs):
                if v > 0:
                    ax_p.text(v+0.12, bar.get_y()+bar.get_height()/2, f"{v:.0f}", va="center", fontsize=8)
            ax_p.spines["top"].set_visible(False)
            ax_p.spines["right"].set_visible(False)
            ax_p.spines["left"].set_color("#e6dbf1")
            ax_p.spines["bottom"].set_color("#e6dbf1")
            ax_p.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close()

            st.markdown('<div class="panel"><div class="section-tag">Appointment Support</div><div class="section-title">What To Say At Your Visit</div></div>', unsafe_allow_html=True)

            scripts = [
                f'"My period pain is {"severe and stops me from functioning" if period_pain>=7 else "moderate and clearly affecting my life" if period_pain>=5 else "noticeable and concerning to me"}."',
                f'"My pelvic pain occurs {"almost every day, not just during my period" if pelvic_pain>=9 else "frequently — more than a week every month" if pelvic_pain>=6 else "often enough to concern me"}."',
            ]
            if bowel_pain >= 3:   scripts.append('"I have pain during bowel movements, especially around my period."')
            if bladder_pain >= 3: scripts.append('"I experience painful urination that keeps returning and does not feel fully explained."')
            if back_leg_val:      scripts.append('"During my period, pain travels into my lower back or legs."')
            if family_hx_val:     scripts.append('"I have a family history of endometriosis and want that taken into account."')
            if worse_val >= 1:    scripts.append('"My symptoms have been getting worse over time, not staying the same."')
            if n_comorbid > 0:
                comorb_list = ", ".join(c for c in selected_comorbidities if c != "None of the above")
                scripts.append(f'"I also have {comorb_list}, which may be relevant to my symptoms."')
            scripts.append('"I would like a referral to a gynaecologist or endometriosis specialist."')

            for s in scripts:
                st.markdown(f'<div class="say-card">{s}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            EndoScan is a research prototype and not a clinical diagnostic tool.
            A definitive diagnosis of endometriosis requires evaluation by a qualified clinician.
            Developed at Purdue University Fort Wayne | Harvard HSIL Hackathon 2026.
        </div>""", unsafe_allow_html=True)


# =========================
# CLINICAL TAB — fully redesigned professional layout
# =========================
with tab_d:
    st.markdown("""
    <div class="panel">
        <div class="section-tag">Clinical Decision Support</div>
        <div class="section-title">Structured Clinical Assessment</div>
        <div class="section-sub">
            Enter assessed symptom scores and clinical indicators. The system will generate a
            structured clinical summary referenced against 699 real endometriosis patients
            (ROXWELL Baseline Study, Meyrose et al.).
        </div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns([1, 1], gap="large")

    with d1:
        # Demographics
        st.markdown('<div class="doc-section-head">Patient Demographics</div>', unsafe_allow_html=True)
        dca, dcb = st.columns(2)
        with dca: da = st.number_input("Age (years)", 14, 60, 28, key="da")
        with dcb: db = st.number_input("BMI", 14.0, 45.0, 23.0, key="db", step=0.1)

        # CHANGE 8: Range shown next to every slider label, no emojis
        st.markdown('<div class="doc-section-head" style="margin-top:1.2rem;">Symptom Scores</div>', unsafe_allow_html=True)
        st.caption("All pain scores: Range 0 (no pain) to 10 (worst possible pain)")

        st.markdown('<span class="dfl">Dysmenorrhoea — menstrual pain</span><span class="dfs">Dysmenorrhoea &nbsp;|&nbsp; Range: 0–10 &nbsp;|&nbsp; 0 = no pain, 10 = worst possible</span>', unsafe_allow_html=True)
        dp  = st.slider("", 0, 10, 6,  key="dp_s",  label_visibility="collapsed")

        st.markdown('<span class="dfl">Chronic pelvic pain</span><span class="dfs">Chronic Pelvic Pain &nbsp;|&nbsp; Range: 0–10</span>', unsafe_allow_html=True)
        dpe = st.slider("", 0, 10, 5,  key="dpe_s", label_visibility="collapsed")

        st.markdown('<span class="dfl">Dyschezia — bowel pain</span><span class="dfs">Dyschezia &nbsp;|&nbsp; Range: 0–10</span>', unsafe_allow_html=True)
        ddc = st.slider("", 0, 10, 2,  key="ddc_s", label_visibility="collapsed")

        st.markdown('<span class="dfl">Dysuria — bladder pain</span><span class="dfs">Dysuria &nbsp;|&nbsp; Range: 0–10</span>', unsafe_allow_html=True)
        ddu = st.slider("", 0, 10, 2,  key="ddu_s", label_visibility="collapsed")

        st.markdown('<span class="dfl">Dyspareunia — intercourse pain</span><span class="dfs">Dyspareunia &nbsp;|&nbsp; Range: 0–10 &nbsp;|&nbsp; Enter 0 if not applicable</span>', unsafe_allow_html=True)
        ddy = st.slider("", 0, 10, 3,  key="ddy_s", label_visibility="collapsed")

        st.markdown('<span class="dfl">Pain Disability Index (PDI)</span><span class="dfs">PDI &nbsp;|&nbsp; Range: 0–70 &nbsp;|&nbsp; 0 = no disability, 70 = complete disability</span>', unsafe_allow_html=True)
        dpd = st.slider("", 0, 70, 30, key="dpd_s", label_visibility="collapsed")

        st.markdown('<div class="doc-section-head" style="margin-top:1.2rem;">Psychosocial Scores</div>', unsafe_allow_html=True)

        st.markdown('<span class="dfl">Depression score (PHQ)</span><span class="dfs">PHQ-2 &nbsp;|&nbsp; Range: 0–6 &nbsp;|&nbsp; Cut-off for probable major depression: 3 or above</span>', unsafe_allow_html=True)
        ddep = st.slider("", 0, 6, 2, key="ddep_s", label_visibility="collapsed")

        st.markdown('<span class="dfl">Anxiety score (GAD)</span><span class="dfs">GAD-2 &nbsp;|&nbsp; Range: 0–6 &nbsp;|&nbsp; Cut-off for probable anxiety disorder: 3 or above</span>', unsafe_allow_html=True)
        danx = st.slider("", 0, 6, 2, key="danx_s", label_visibility="collapsed")

        # Clinical checkboxes — no emojis
        st.markdown('<div class="doc-section-head" style="margin-top:1.2rem;">Clinical Indicators</div>', unsafe_allow_html=True)
        dha  = st.checkbox("Hormone level abnormality (oestrogen, progesterone, LH, FSH)", key="dha")
        dinf = st.checkbox("Infertility / subfertility", key="dinf")
        dfh  = st.checkbox("First-degree family history of endometriosis", key="dfh")
        dhb  = st.checkbox("Menorrhagia — soaking through protection in under 1 hour", key="dhb")
        dir_ = st.checkbox("Menstrual irregularity", key="dir_")
        dbk  = st.checkbox("Referred lumbar or sciatic pain", key="dbk")
        dov  = st.checkbox("Mittelschmerz — mid-cycle ovulation pain", key="dov")
        dws  = st.checkbox("Progressive worsening of symptoms over time", key="dws")

        # Comorbidities
        st.markdown('<div class="doc-section-head" style="margin-top:1.2rem;">Comorbidities</div>', unsafe_allow_html=True)
        doc_comorbid = st.multiselect(
            "Select confirmed comorbidities:",
            COMORBIDITIES,
            key="doc_comorbid"
        )
        doc_n_comorbid = len([c for c in doc_comorbid if c != "None of the above"])

        # CHANGE 9: Button text has no emoji
        gen = st.button("Generate Clinical Report", key="gen", use_container_width=True)

    with d2:
        if not gen:
            st.markdown("""
            <div class="doc-box" style="min-height:500px;display:flex;align-items:center;
                        justify-content:center;text-align:center;">
                <div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:1.9rem;
                                color:#5a4a70;margin-bottom:0.4rem;">Clinical Report</div>
                    <div class="small-muted" style="max-width:300px;margin:0 auto;">
                        Complete the symptom scores on the left and click
                        Generate Clinical Report to produce the structured summary.
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        if gen:
            # CHANGE 4: Auto-scroll to report on the clinical side
            components.html("""
            <script>
                setTimeout(function(){
                    window.parent.scrollTo({top: 400, behavior: 'smooth'});
                }, 200);
            </script>""", height=0)

            # ── CALL MODELS ──
            drp_raw, desrc = pred_endo(da, int(dir_), dpe, int(dha), int(dinf), db)
            if drp_raw is None:
                st.error("Diagnosis model not loaded. Check model files.")
                st.stop()

            dep_ = drp_raw
            if dfh:  dep_ = min(.97, dep_ * 1.35)
            if dws:  dep_ = min(.97, dep_ * 1.15)
            if dov:  dep_ = min(.97, dep_ * 1.05)
            if "Irritable Bowel Syndrome (IBS)" in doc_comorbid:
                dep_ = min(.97, dep_ * 1.05)
            if "Interstitial Cystitis (painful bladder syndrome)" in doc_comorbid:
                dep_ = min(.97, dep_ * 1.04)
            drp = round(dep_ * 100)

            dpb_  = round(float(np.mean([dp, dpe, ddy, ddu, ddc])), 2)
            dms_  = 1 if (ddu >= 5 and ddc >= 5) else 0
            dcst_ = int(min(35, max(6, 12 + (ddep/6)*10 + (dp/10)*8)))
            dcl, dksrc = pred_cluster(dp, dpe, ddy, ddu, ddc, dpd, dcst_, dpb_, dms_)
            if dcl is None: dcl = 1
            dcp   = CL[dcl]
            dpct_ = pdpct(dpd)
            drc, drt, drbg, drbr = risk_style(drp)

            # ── REPORT HEADER — clean professional ──
            st.markdown(f"""
            <div class="doc-box" style="border-left:4px solid {drc};">
                <div class="doc-header">
                    <div class="doc-title">Clinical Assessment Report</div>
                    <div class="doc-subtitle">EndoScan &nbsp;|&nbsp; ROXWELL Baseline Study (Meyrose et al., n=699)</div>
                </div>
                <div style="display:flex;align-items:baseline;gap:1.2rem;margin-bottom:0.6rem;">
                    <div style="font-family:'Cormorant Garamond',serif;font-size:3rem;
                                line-height:1;color:{drc};">{drp}%</div>
                    <div class="doc-tier" style="background:{drbg};color:{drc};border:1px solid {drbr};">
                        {drt}
                    </div>
                </div>
                <div style="font-size:0.8rem;color:#9b8fb5;line-height:1.6;">
                    Diagnosis model: <code>{desrc}</code> &nbsp;|&nbsp;
                    Raw output: {drp_raw:.3f} &rarr; adjusted: {dep_:.3f}
                </div>
            </div>""", unsafe_allow_html=True)

            # ── KEY METRICS TABLE — professional rows ──
            st.markdown(f"""
            <div class="doc-box">
                <div class="doc-section-head">Patient Summary</div>
                <div class="doc-row">
                    <span class="doc-row-label">Patient subtype</span>
                    <span class="doc-row-val" style="color:{dcp['color']};">{dcp['name']}</span>
                </div>
                <div class="doc-row">
                    <span class="doc-row-label">Clustering model</span>
                    <span class="doc-row-val"><code>{dksrc}</code></span>
                </div>
                <div class="doc-row">
                    <span class="doc-row-label">Pain disability percentile</span>
                    <span class="doc-row-val">{dpct_}th (vs endo population, n=699)</span>
                </div>
                <div class="doc-row">
                    <span class="doc-row-label">Typical diagnostic delay in this subtype</span>
                    <span class="doc-row-val">{dcp['avg_delay']}</span>
                </div>
                <div class="doc-row">
                    <span class="doc-row-label">Depression rate in this subtype</span>
                    <span class="doc-row-val">{dcp['dep']}% &nbsp;vs 8% general population</span>
                </div>
                <div class="doc-row">
                    <span class="doc-row-label">Anxiety disorder rate in this subtype</span>
                    <span class="doc-row-val">{dcp['anx']}% &nbsp;vs 8% general population</span>
                </div>
                <div class="doc-row">
                    <span class="doc-row-label">Unable to work — this subtype</span>
                    <span class="doc-row-val">{dcp['cw']}%</span>
                </div>
                {"<div class='doc-row'><span class='doc-row-label'>Comorbidities recorded</span><span class='doc-row-val'>" + str(doc_n_comorbid) + " condition(s)</span></div>" if doc_n_comorbid > 0 else ""}
            </div>""", unsafe_allow_html=True)

            # ── RED FLAGS — professional, no emojis ──
            dflags = []
            if dp   >= 7:  dflags.append("Severe dysmenorrhoea (score 7–10/10)")
            if ddc  >= 5:  dflags.append("Significant dyschezia — consider deep infiltrating endometriosis")
            if ddu  >= 5:  dflags.append("Dysuria score 5–10 — exclude recurrent UTI; consider vesical endo")
            if dbk:        dflags.append("Referred lumbar or sciatic pain — possible nerve root involvement")
            if dhb:        dflags.append("Menorrhagia — soaking through protection in under 1 hour")
            if dfh:        dflags.append("First-degree family history — approximately 7-fold genetic risk")
            if dws:        dflags.append("Progressive worsening — strongest differentiator from primary dysmenorrhoea")
            if dov:        dflags.append("Mittelschmerz — mid-cycle pain; a specific endometriosis indicator")
            if ddep >= 4:  dflags.append("PHQ depression score 4–6 — probable major depression")
            if danx >= 4:  dflags.append("GAD anxiety score 4–6 — probable anxiety disorder")
            if "Irritable Bowel Syndrome (IBS)" in doc_comorbid:
                dflags.append("IBS recorded — may co-occur with or conceal bowel endometriosis")
            if "Autoimmune condition (e.g. lupus, thyroid disease, rheumatoid arthritis)" in doc_comorbid:
                dflags.append("Autoimmune condition recorded — associated with elevated disease severity")
            if doc_n_comorbid >= 3:
                dflags.append(f"{doc_n_comorbid} comorbidities recorded — multidisciplinary review may be appropriate")

            if dflags:
                flags_rows = "".join(
                    f'<div class="doc-flag"><div class="doc-flag-mark"></div><div>{f}</div></div>'
                    for f in dflags
                )
                st.markdown(f"""
                <div class="doc-box">
                    <div class="doc-section-head">Clinical Red Flags</div>
                    {flags_rows}
                </div>""", unsafe_allow_html=True)

            # ── SUBTYPE NARRATIVE ──
            st.markdown(f"""
            <div class="doc-box">
                <div class="doc-section-head">Subtype Clinical Narrative</div>
                <div class="doc-cluster-desc">{dcp['desc']}</div>
                <div class="doc-learn">
                    <strong>Physician learning note:</strong> {dcp['missed']}
                    Asking specifically about progressive worsening year on year and bowel or bladder
                    involvement in patients presenting with pelvic pain significantly improves
                    endometriosis detection rates in primary care.
                </div>
            </div>""", unsafe_allow_html=True)

            # ── COMORBIDITIES IN REPORT ──
            if doc_n_comorbid > 0:
                comorb_rows = "".join(
                    f'<div class="doc-row"><span class="doc-row-label">{c}</span>'
                    f'<span class="doc-row-val">Confirmed</span></div>'
                    for c in doc_comorbid if c != "None of the above"
                )
                st.markdown(f"""
                <div class="doc-box">
                    <div class="doc-section-head">Recorded Comorbidities</div>
                    {comorb_rows}
                </div>""", unsafe_allow_html=True)

            # ── RECOMMENDED ACTION ──
            st.markdown(f"""
            <div class="doc-action">
                <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.18em;
                            color:#9b6fc7;font-weight:700;margin-bottom:0.4rem;">
                    Recommended Action
                </div>
                {dcp['action']}
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer" style="margin-top:1rem;">
                Clinical decision support only. Definitive diagnosis requires evaluation by a
                qualified clinician and, where indicated, laparoscopic confirmation.
                ROXWELL Baseline Study (Meyrose et al., n=699) |
                Harvard HSIL Hackathon 2026 | Purdue University Fort Wayne.
            </div>""", unsafe_allow_html=True)
