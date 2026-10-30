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
    page_icon="🌸",
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
    --card:#ffffffcc;
    --text:#241733;
    --muted:#7a6c8c;
    --line:#eadff5;
    --purple:#7b4bc4;
    --purple2:#b187e8;
    --rose:#d66aa8;
    --green:#2f8b6c;
    --amber:#a66b10;
    --red:#b94545;
    --shadow:0 12px 35px rgba(91, 54, 143, 0.10);
    --radius:20px;
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

h1, h2, h3, h4{
    font-family:'Cormorant Garamond', serif !important;
    color:var(--text);
}

.hero{
    background:linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,255,255,0.72));
    border:1px solid rgba(234,223,245,0.9);
    border-radius:28px;
    padding:2.1rem 2rem;
    box-shadow:var(--shadow);
    backdrop-filter:blur(8px);
    margin-bottom:1.1rem;
}

.hero-title{
    font-family:'Cormorant Garamond', serif;
    font-size:3.1rem;
    line-height:1;
    margin-bottom:0.3rem;
    color:#2a173a;
}

.hero-sub{
    font-size:1rem;
    color:var(--muted);
    line-height:1.75;
    max-width:820px;
}

.stats-row{
    display:flex;
    flex-wrap:wrap;
    gap:0.8rem;
    margin-top:1.3rem;
}

.stat-chip{
    background:linear-gradient(135deg, #ffffff, #faf5ff);
    border:1px solid var(--line);
    border-radius:16px;
    padding:0.9rem 1rem;
    min-width:150px;
    box-shadow:0 6px 18px rgba(91,54,143,0.06);
}

.stat-num{
    font-family:'Cormorant Garamond', serif;
    font-size:1.8rem;
    color:var(--purple);
    line-height:1;
}

.stat-label{
    font-size:0.76rem;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:0.12em;
    margin-top:0.25rem;
}

.panel{
    background:linear-gradient(180deg, rgba(255,255,255,0.82), rgba(255,255,255,0.74));
    border:1px solid rgba(234,223,245,0.95);
    border-radius:24px;
    padding:1.4rem 1.3rem;
    box-shadow:var(--shadow);
    backdrop-filter:blur(8px);
    margin-bottom:1rem;
}

.section-tag{
    font-size:0.72rem;
    color:var(--purple);
    text-transform:uppercase;
    letter-spacing:0.18em;
    font-weight:700;
    margin-bottom:0.35rem;
}

.section-title{
    font-family:'Cormorant Garamond', serif;
    font-size:2rem;
    color:#2a173a;
    margin-bottom:0.2rem;
}

.section-sub{
    color:var(--muted);
    font-size:0.94rem;
    line-height:1.7;
    margin-bottom:1rem;
}

.question-wrap{
    background:rgba(255,255,255,0.75);
    border:1px solid #efe5f7;
    border-radius:18px;
    padding:1rem 1rem 0.8rem 1rem;
    margin-bottom:0.9rem;
}

.question-title{
    font-weight:700;
    color:#29193a;
    margin-bottom:0.2rem;
    font-size:1rem;
}

.question-sci{
    color:var(--purple);
    font-size:0.78rem;
    font-style:italic;
    margin-bottom:0.2rem;
}

.question-hint{
    color:var(--muted);
    font-size:0.84rem;
    line-height:1.6;
    margin-bottom:0.45rem;
}

.cta-box{
    background:linear-gradient(135deg, #f7efff, #fff4fb);
    border:1px solid #e5d7f6;
    border-radius:22px;
    padding:1.35rem;
    text-align:center;
    margin-top:1rem;
}

.cta-title{
    font-family:'Cormorant Garamond', serif;
    font-size:1.7rem;
    color:#2a173a;
    margin-bottom:0.2rem;
}

.cta-sub{
    color:var(--muted);
    font-size:0.92rem;
    line-height:1.7;
}

.result-card{
    background:rgba(255,255,255,0.9);
    border:1px solid var(--line);
    border-radius:22px;
    padding:1.2rem;
    box-shadow:var(--shadow);
    height:100%;
}

.result-label{
    font-size:0.72rem;
    text-transform:uppercase;
    letter-spacing:0.16em;
    color:var(--muted);
    margin-bottom:0.45rem;
    font-weight:700;
}

.result-big{
    font-family:'Cormorant Garamond', serif;
    font-size:3rem;
    line-height:1;
    margin-bottom:0.35rem;
}

.kpi-sub{
    color:var(--muted);
    font-size:0.82rem;
    line-height:1.6;
}

.soft-box{
    background:rgba(255,255,255,0.82);
    border:1px solid var(--line);
    border-radius:20px;
    padding:1rem 1rem;
    margin-bottom:1rem;
}

.flag{
    display:flex;
    gap:0.7rem;
    align-items:flex-start;
    background:#fff7f7;
    border:1px solid #f0d4d4;
    border-radius:14px;
    padding:0.75rem 0.85rem;
    margin-bottom:0.5rem;
    color:#622222;
    font-size:0.88rem;
}

.flag-dot{
    width:8px;
    height:8px;
    border-radius:50%;
    background:#c84e4e;
    margin-top:0.4rem;
    flex-shrink:0;
}

.say-card{
    border-left:4px solid var(--green);
    background:#f4fbf8;
    border-radius:0 14px 14px 0;
    padding:0.8rem 1rem;
    margin-bottom:0.5rem;
    color:#194334;
    font-size:0.9rem;
    line-height:1.7;
}

.note{
    background:#fffef7;
    border:1px solid #f1e5ba;
    border-radius:16px;
    padding:0.95rem 1rem;
    color:#6a5712;
    font-size:0.86rem;
    line-height:1.65;
}

.disclaimer{
    background:rgba(255,255,255,0.82);
    border:1px solid var(--line);
    border-radius:18px;
    padding:0.95rem 1rem;
    color:var(--muted);
    font-size:0.82rem;
    line-height:1.7;
    text-align:center;
}

.stTabs [data-baseweb="tab-list"]{
    background:#f2ebfb !important;
    border-radius:16px !important;
    padding:5px !important;
    gap:6px !important;
}
.stTabs [data-baseweb="tab"]{
    border-radius:12px !important;
    padding:0.7rem 1.2rem !important;
    color:#6d6180 !important;
    font-weight:600 !important;
    background:transparent !important;
}
.stTabs [aria-selected="true"]{
    background:white !important;
    color:var(--purple) !important;
    box-shadow:0 4px 12px rgba(91,54,143,0.08) !important;
}

[data-testid="stNumberInput"] input,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div{
    background:rgba(255,255,255,0.9) !important;
    border:1.5px solid #e7dcf3 !important;
    border-radius:14px !important;
    color:var(--text) !important;
}

.stRadio > label, .stCheckbox > label{
    color:var(--text) !important;
}

.stButton > button{
    width:100% !important;
    border:none !important;
    border-radius:999px !important;
    padding:0.95rem 1.4rem !important;
    background:linear-gradient(135deg, #7b4bc4, #d66aa8) !important;
    color:white !important;
    font-size:1rem !important;
    font-weight:700 !important;
    box-shadow:0 10px 24px rgba(123,75,196,0.28) !important;
}

.stButton > button:hover{
    transform:translateY(-1px);
    box-shadow:0 14px 30px rgba(123,75,196,0.34) !important;
}

[data-testid="stMetric"]{
    background:rgba(255,255,255,0.85) !important;
    border:1px solid var(--line) !important;
    border-radius:18px !important;
    padding:0.8rem !important;
}

hr{
    border-color:#ecdef8 !important;
}

.small-muted{
    color:var(--muted);
    font-size:0.82rem;
    line-height:1.7;
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    m = {}

    try:
        m["endo"] = joblib.load("models/diagnosis_model.joblib")
        m["endo_sc"] = joblib.load("models/diagnosis_scaler.joblib")
        with open("models/diagnosis_features.json", "r") as f:
            m["ef"] = json.load(f)
        m["endo_ok"] = True
    except Exception as e:
        m["endo_ok"] = False
        m["endo_err"] = str(e)

    try:
        m["km"] = joblib.load("models/kmeans_model.joblib")
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
        "Age": age,
        "Menstrual_Irregularity": irr,
        "Chronic_Pain_Level": pain,
        "Hormone_Level_Abnormality": hormone,
        "Infertility": inf,
        "BMI": bmi
    }])

    X = X[MDL["ef"]]
    X_scaled = MDL["endo_sc"].transform(X)
    prob = float(MDL["endo"].predict_proba(X_scaled)[0][1])
    return prob, "diagnosis_model.joblib"


def pred_cluster(dys, abd, icp, dyu, dyc, pd_, cst, pb, ms):
    if not MDL["km_ok"]:
        return None, "not_loaded"

    X = np.array([[dys, abd, icp, dyu, dyc, pd_, cst, pb, ms]])
    X_scaled = MDL["km_sc"].transform(X)
    cluster = int(MDL["km"].predict(X_scaled)[0])
    return cluster, "kmeans_model.joblib"


def pdpct(score):
    pts = [(0, 0), (16, 22), (35, 46), (50, 74), (65, 91), (70, 100)]
    for i in range(len(pts) - 1):
        if pts[i][0] <= score <= pts[i + 1][0]:
            f = (score - pts[i][0]) / (pts[i + 1][0] - pts[i][0] + 1e-9)
            return int(pts[i][1] + f * (pts[i + 1][1] - pts[i][1]))
    return 100


def risk_color(score):
    if score >= 60:
        return "#b94545", "Higher Risk", "#fff7f7", "#f0d0d0"
    if score >= 35:
        return "#a66b10", "Moderate Risk", "#fffaf1", "#f0dfba"
    return "#2f8b6c", "Lower Risk", "#f4fbf8", "#cfe9df"


CL = {
    0: {
        "name": "Cluster 0 — High pain + severe psychological burden",
        "color": "#b94545",
        "bg": "#fff8f8",
        "border": "#e6c7c7",
        "dep": 92,
        "anx": 90,
        "cw": 18,
        "n": 144,
        "desc": "Severe physical pain and severe psychological burden appear together in this group. This profile often experiences the longest diagnostic delays.",
        "missed": "Depression and anxiety may be consequences of undiagnosed endometriosis rather than separate primary problems.",
        "action": "Urgent referral to a gynaecology specialist. Mental health evaluation should also be considered."
    },
    1: {
        "name": "Cluster 1 — Moderate pain, psychologically resilient",
        "color": "#406ea8",
        "bg": "#f7faff",
        "border": "#d6e3f5",
        "dep": 18,
        "anx": 17,
        "cw": 9,
        "n": 233,
        "desc": "Symptoms are moderate, with lower psychological burden. This may represent a stage where earlier intervention can make a big difference.",
        "missed": "Pain is often dismissed because the person appears emotionally well and functional.",
        "action": "Referral for further evaluation is recommended. Monitor progression closely."
    },
    2: {
        "name": "Cluster 2 — Widespread multi-site physical pain",
        "color": "#b94545",
        "bg": "#fff8f8",
        "border": "#e6c7c7",
        "dep": 69,
        "anx": 67,
        "cw": 15,
        "n": 134,
        "desc": "High pain burden involving bowel, bladder, and pelvis. This profile is more consistent with deep infiltrating disease.",
        "missed": "Symptoms are commonly attributed to IBS or recurrent UTI before endometriosis is considered.",
        "action": "Urgent specialist referral. Imaging and multi-organ evaluation may be needed."
    },
    3: {
        "name": "Cluster 3 — High pain, psychologically coping",
        "color": "#a66b10",
        "bg": "#fffaf2",
        "border": "#eedebc",
        "dep": 12,
        "anx": 21,
        "cw": 5,
        "n": 188,
        "desc": "Physical pain is high, but the person appears to be coping psychologically. This can lead to underreporting and delayed care.",
        "missed": "The pain may be normalized for years, especially if symptoms gradually worsen over time.",
        "action": "Referral recommended. Pain management and monitoring of worsening symptoms are important."
    },
}


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">EndoScan</div>
    <div class="hero-sub">
        A clean research prototype for endometriosis symptom assessment and clinical decision support.
        Answer the questions below and generate an instant risk estimate and patient subtype summary.
    </div>
    <div class="stats-row">
        <div class="stat-chip"><div class="stat-num">1 in 10</div><div class="stat-label">Women affected</div></div>
        <div class="stat-chip"><div class="stat-num">7–10 yrs</div><div class="stat-label">Average delay</div></div>
        <div class="stat-chip"><div class="stat-num">699</div><div class="stat-label">Real patients</div></div>
        <div class="stat-chip"><div class="stat-num">4</div><div class="stat-label">Subtypes found</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

if MDL["endo_ok"] and MDL["km_ok"]:
    st.success("ML models loaded successfully. Risk prediction and subtype clustering are active.")
else:
    if not MDL["endo_ok"]:
        st.error(f"Diagnosis model missing: {MDL.get('endo_err')}")
    if not MDL["km_ok"]:
        st.error(f"KMeans model missing: {MDL.get('km_err')}")

tab_p, tab_d = st.tabs([" Patient Assessment", "Clinical Decision Support"])


# =========================
# PATIENT TAB
# =========================
with tab_p:
    st.markdown("""
    <div class="panel">
        <div class="section-tag">Patient Flow</div>
        <div class="section-title">Endometriosis Assessment Form</div>
        <div class="section-sub">
            This version is designed to feel simpler and cleaner. Fill out the sections below and click the button at the bottom to generate your assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    def prq(lbl, sci, hint, opts, key):
        st.markdown(
            f'<div class="question-wrap"><div class="question-title">{lbl}</div><div class="question-sci">{sci}</div><div class="question-hint">{hint}</div>',
            unsafe_allow_html=True
        )
        value = opts[st.radio("", list(opts.keys()), key=key, label_visibility="collapsed")]
        st.markdown("</div>", unsafe_allow_html=True)
        return value

    PAIN = {
        "No pain at all — completely normal (0)": 0,
        "Mild (2) — uncomfortable but manageable": 2,
        "Moderate (5) — slowed me down significantly": 5,
        "Severe (7) — difficult to function normally": 7,
        "Unbearable (10) — incapacitating": 10
    }
    FREQ = {
        "Never (0)": 0,
        "Rarely (3) — a few days per month": 3,
        "Often (6) — more than a week per month": 6,
        "Almost always (9) — nearly every day": 9
    }
    DISAB = {
        "No days — functioned normally (0)": 0,
        "1–2 days — mild impact (16)": 16,
        "3–5 days — moderate impact (35)": 35,
        "More than 5 days — severe impact (50)": 50,
        "Almost every day — major impact (65)": 65
    }
    BLEED = {
        "Light to normal — change every 4+ hours (0)": 0,
        "Heavy (1) — change every 2–3 hours": 1,
        "Very heavy (2) — soak through in under 1 hour": 2
    }
    FAT = {
        "Normal energy (0)": 0,
        "Mild (1) — needed extra rest": 1,
        "Significant (3) — affects my full day": 3,
        "Severe (4) — could not function normally": 4
    }

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="panel"><div class="section-tag">Section 1</div><div class="section-title">Pain Assessment</div></div>', unsafe_allow_html=True)

        period_pain = prq(
            "Dysmenorrhoea (Menstrual / Period Pain)",
            "Pain during menstruation; a core endometriosis symptom.",
            "How strongly does your period pain affect your ability to function?",
            PAIN,
            "pp"
        )

        pelvic_pain = prq(
            "Chronic Pelvic Pain",
            "Persistent lower abdominal pain even outside the menstrual period.",
            "How often do you experience pelvic pain?",
            FREQ,
            "pv"
        )

        bowel_pain = prq(
            "Dyschezia (Bowel / Rectal Pain)",
            "Pain during bowel movements, often worse around menstruation.",
            "How severe is your bowel pain?",
            PAIN,
            "bp"
        )

        bladder_pain = prq(
            "Dysuria (Bladder / Urination Pain)",
            "Pain, burning, or pressure with urination.",
            "How severe is your bladder pain?",
            PAIN,
            "du"
        )

        st.markdown('<div class="question-wrap"><div class="question-title">Dyspareunia (Intercourse Pain)</div><div class="question-sci">Pain during or after intercourse; commonly linked with deep infiltrating disease.</div><div class="question-hint">Choose the option that best fits your experience.</div>', unsafe_allow_html=True)
        sx = st.radio("", ["Not applicable", "No pain (0)", "Moderate pain (4)", "Severe pain (8) — avoid or stop intercourse"], key="sx", label_visibility="collapsed")
        sx_map = {
            "Not applicable": (0, 0),
            "No pain (0)": (0, 1),
            "Moderate pain (4)": (4, 1),
            "Severe pain (8) — avoid or stop intercourse": (8, 1)
        }
        intercourse_pain, intercourse_reported = sx_map[sx]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="section-tag">Section 2</div><div class="section-title">Systemic Symptoms</div></div>', unsafe_allow_html=True)

        fatigue = prq(
            "Chronic Fatigue",
            "Persistent tiredness not fully relieved by rest.",
            "How would you describe your fatigue from symptoms?",
            FAT,
            "fat"
        )

        st.markdown('<div class="question-wrap"><div class="question-title">Abdominal Bloating ("Endo Belly")</div><div class="question-sci">Visible abdominal swelling and bloating associated with endometriosis symptoms.</div><div class="question-hint">Select the best match.</div>', unsafe_allow_html=True)
        bloating = st.radio("", ["No", "Yes — noticeable", "Yes — severe and visibly swollen"], key="bl", label_visibility="collapsed")
        bloating_val = 0 if bloating == "No" else 1
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="question-wrap"><div class="question-title">Mittelschmerz (Ovulation Pain)</div><div class="question-sci">Mid-cycle pain can be a useful clue in symptom patterns.</div><div class="question-hint">Do you notice pain around ovulation?</div>', unsafe_allow_html=True)
        ovul = st.radio("", ["No / not sure", "Yes, occasionally", "Yes, regularly every cycle"], key="ov", label_visibility="collapsed")
        ovul_val = 0 if ovul == "No / not sure" else 1
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="question-wrap"><div class="question-title">Referred Back / Leg Pain</div><div class="question-sci">Pain that radiates to the lower back or legs can reflect deeper pelvic involvement.</div><div class="question-hint">Do you feel pain traveling into your back or legs during periods?</div>', unsafe_allow_html=True)
        back_leg = st.radio("", ["No", "Yes — pain travels to lower back or legs"], key="blp", horizontal=True, label_visibility="collapsed")
        back_leg_val = 1 if back_leg != "No" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="section-tag">Section 3</div><div class="section-title">Menstrual & Reproductive History</div></div>', unsafe_allow_html=True)

        heavy_bleed = prq(
            "Menorrhagia (Heavy Menstrual Bleeding)",
            "Abnormally heavy menstrual bleeding.",
            "How heavy is your bleeding?",
            BLEED,
            "hb"
        )

        st.markdown('<div class="question-wrap"><div class="question-title">Menstrual Irregularity</div><div class="question-sci">Irregular cycles, spotting, or unpredictable timing.</div><div class="question-hint">Choose the option that best describes your cycle.</div>', unsafe_allow_html=True)
        irreg = st.radio("", ["Yes — always regular (0)", "Sometimes irregular", "Often irregular or unpredictable (1)"], key="irr", label_visibility="collapsed")
        irreg_val = 0 if irreg == "Yes — always regular (0)" else 1
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="question-wrap"><div class="question-title">Progressive Symptom Worsening</div><div class="question-sci">Worsening over time is an important differentiator from more routine menstrual pain.</div><div class="question-hint">Have symptoms become worse over the years?</div>', unsafe_allow_html=True)
        worse = st.radio("", ["Staying the same", "Getting slightly worse each year", "Getting significantly worse over time"], key="ws", label_visibility="collapsed")
        worse_val = ["Staying the same", "Getting slightly worse each year", "Getting significantly worse over time"].index(worse)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="section-tag">Section 4</div><div class="section-title">Life Impact & Wellbeing</div></div>', unsafe_allow_html=True)

        pain_disability = prq(
            "Functional Disability (Pain Disability Index)",
            "How much pain interferes with daily life, work, and activities.",
            "In the last month, how many days did pain stop normal functioning?",
            DISAB,
            "pd"
        )

        st.markdown('<div class="question-wrap"><div class="question-title">Depression Screening (PHQ-2)</div><div class="question-sci">A short depression screening used in many clinical settings.</div><div class="question-hint">In the last 2 weeks, have you felt down, hopeless, or lost interest?</div>', unsafe_allow_html=True)
        dq = st.radio("", ["No — I have felt like myself (0)", "Yes, sometimes — a few days (2)", "Yes, often — most days feel difficult (4)"], key="dq", label_visibility="collapsed")
        depression = {
            "No — I have felt like myself (0)": 0,
            "Yes, sometimes — a few days (2)": 2,
            "Yes, often — most days feel difficult (4)": 4
        }[dq]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="question-wrap"><div class="question-title">Anxiety Screening (GAD-2)</div><div class="question-sci">A short screening for anxiety symptoms.</div><div class="question-hint">In the last 2 weeks, have you felt anxious, on edge, or unable to stop worrying?</div>', unsafe_allow_html=True)
        aq = st.radio("", ["No — calm and in control (0)", "Yes, sometimes — more worried than usual (2)", "Yes, often — constant worry I cannot control (4)"], key="aq", label_visibility="collapsed")
        anxiety = {
            "No — calm and in control (0)": 0,
            "Yes, sometimes — more worried than usual (2)": 2,
            "Yes, often — constant worry I cannot control (4)": 4
        }[aq]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="section-tag">Section 5</div><div class="section-title">Personal & Clinical Background</div></div>', unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            age = st.number_input("Age", 14, 60, 28)
        with cb:
            bmi = st.number_input("BMI", 14.0, 45.0, 23.0, step=0.1)

        st.markdown('<div class="question-wrap"><div class="question-title">Hormone Level Abnormality</div><div class="question-sci">Known hormonal dysregulation or abnormal testing.</div>', unsafe_allow_html=True)
        ha = st.radio("", ["No / Not tested", "Yes"], key="ha", horizontal=True, label_visibility="collapsed")
        hormone_val = 1 if ha == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="question-wrap"><div class="question-title">Infertility / Subfertility</div><div class="question-sci">Difficulty conceiving may be associated with endometriosis.</div>', unsafe_allow_html=True)
        inf = st.radio("", ["No / Not applicable", "Yes"], key="inf", horizontal=True, label_visibility="collapsed")
        infertility_val = 1 if inf == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="question-wrap"><div class="question-title">Family History of Endometriosis</div><div class="question-sci">First-degree family history increases risk.</div>', unsafe_allow_html=True)
        fhx = st.radio("", ["No / I don't know", "Yes"], key="fhx", horizontal=True, label_visibility="collapsed")
        family_hx_val = 1 if fhx == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="cta-box">
        <div class="cta-title">Ready to generate your assessment?</div>
        <div class="cta-sub">
            Your answers will be processed using the trained diagnosis model and clustering model.
            Results will appear below instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    run = st.button(" Generate My Endometriosis Assessment", key="run", use_container_width=True)

    if run:
        st.session_state["show_results"] = True

    if st.session_state.get("show_results", False):
        st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
        components.html(
            """
            <script>
                const scrollToResults = () => {
                    const el = window.parent.document.getElementById("results-anchor");
                    if (el) {
                        el.scrollIntoView({behavior: "smooth", block: "start"});
                    }
                };
                setTimeout(scrollToResults, 150);
            </script>
            """,
            height=0,
        )

        st.markdown("""
        <div class="panel">
            <div class="section-tag">Results</div>
            <div class="section-title">Your Assessment Summary</div>
            <div class="section-sub">
                These results come from the trained ML models and the symptom pattern you entered.
            </div>
        </div>
        """, unsafe_allow_html=True)

        raw_prob, esrc = pred_endo(age, irreg_val, pelvic_pain, hormone_val, infertility_val, bmi)
        if raw_prob is None:
            st.error("Diagnosis model not loaded. Check your model files first.")
            st.stop()

        ep = raw_prob
        if family_hx_val:
            ep = min(.97, ep * 1.35)
        if worse_val >= 2:
            ep = min(.97, ep * 1.18)
        elif worse_val == 1:
            ep = min(.97, ep * 1.08)
        if ovul_val:
            ep = min(.97, ep * 1.05)

        rp = round(ep * 100)

        pb = round(float(np.mean([period_pain, pelvic_pain, intercourse_pain, bladder_pain, bowel_pain])), 2)
        ms = 1 if (bladder_pain >= 5 and bowel_pain >= 5) else 0
        cst = int(min(35, max(6, 12 + (depression / 4) * 10 + (period_pain / 10) * 8)))

        cl, ksrc = pred_cluster(
            period_pain, pelvic_pain, intercourse_pain,
            bladder_pain, bowel_pain, pain_disability,
            cst, pb, ms
        )
        if cl is None:
            cl = 1

        cp = CL[cl]
        pct = pdpct(pain_disability)
        rc, rt, rbg, rbr = risk_color(rp)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'''
                <div class="result-card" style="background:{rbg}; border-color:{rbr};">
                    <div class="result-label">Endometriosis Risk Score</div>
                    <div class="result-big" style="color:{rc};">{rp}%</div>
                    <div style="height:7px;background:#eadff5;border-radius:999px;overflow:hidden;margin:0.6rem 0 0.5rem 0;">
                        <div style="height:100%;width:{rp}%;background:{rc};border-radius:999px;"></div>
                    </div>
                    <div style="font-weight:700;color:{rc};margin-bottom:0.35rem;">{rt}</div>
                    <div class="kpi-sub">Raw model: {raw_prob:.3f} → adjusted: {ep:.3f}<br>Source: <code>{esrc}</code></div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f'''
                <div class="result-card">
                    <div class="result-label">Patient Subtype</div>
                    <div style="font-family:'Cormorant Garamond', serif; font-size:1.55rem; color:{cp["color"]}; line-height:1.25;">
                        {cp["name"]}
                    </div>
                    <div class="kpi-sub" style="margin-top:0.5rem;">
                        Source: <code>{ksrc}</code><br>
                        Derived from symptom clustering on real patient data.
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                f'''
                <div class="result-card">
                    <div class="result-label">Pain Disability Percentile</div>
                    <div class="result-big" style="color:#7b4bc4;">{pct}<span style="font-size:1.4rem;">th</span></div>
                    <div class="kpi-sub">Compared against the endometriosis patient reference population.</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        st.markdown(
            f'''
            <div class="soft-box" style="background:{cp["bg"]}; border-color:{cp["border"]};">
                <div class="result-label">Subtype Interpretation</div>
                <div style="font-family:'Cormorant Garamond', serif; font-size:1.75rem; color:{cp["color"]}; margin-bottom:0.45rem;">
                    {cp["name"]}
                </div>
                <div class="small-muted" style="color:#4f415f;">
                    {cp["desc"]}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Major depression in subtype", f'{cp["dep"]}%')
        sc2.metric("Anxiety in subtype", f'{cp["anx"]}%')
        sc3.metric("Unable to work", f'{cp["cw"]}%')
        sc4.metric("Cluster size", f'{cp["n"]}')

        flags = []
        if period_pain >= 7:
            flags.append("Severe dysmenorrhoea — pain rated severe or unbearable.")
        if bowel_pain >= 6:
            flags.append("Significant dyschezia — possible deep infiltrating disease.")
        if bladder_pain >= 6:
            flags.append("Severe dysuria — may indicate bladder involvement.")
        if back_leg_val:
            flags.append("Referred lower back or leg pain — possible deeper pelvic involvement.")
        if heavy_bleed >= 2:
            flags.append("Very heavy bleeding — soaking through protection in under 1 hour.")
        if family_hx_val:
            flags.append("Family history present — meaningful inherited risk signal.")
        if worse_val >= 1:
            flags.append("Symptoms are progressively worsening over time.")
        if ovul_val:
            flags.append("Mittelschmerz / ovulation pain reported.")
        if fatigue >= 3:
            flags.append("Severe fatigue linked with symptom burden.")
        if depression >= 4:
            flags.append("PHQ-2 positive for major depressive symptom burden.")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="panel"><div class="section-tag">Clinical Red Flags</div><div class="section-title">Important Signals</div></div>', unsafe_allow_html=True)
            if flags:
                for fl in flags:
                    st.markdown(f'<div class="flag"><div class="flag-dot"></div><div>{fl}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="soft-box">No major red flags detected at the current response levels.</div>', unsafe_allow_html=True)

            st.markdown(
                f'''
                <div class="soft-box" style="background:{cp["bg"]}; border-color:{cp["border"]};">
                    <div class="result-label">Recommended Action</div>
                    <div style="font-weight:700; color:{cp["color"]}; line-height:1.8;">
                        {cp["action"]}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown(
                f'''
                <div class="note">
                    <strong>Commonly missed in this subtype:</strong> {cp["missed"]}
                </div>
                ''',
                unsafe_allow_html=True
            )

        with right:
            st.markdown('<div class="panel"><div class="section-tag">Symptom Profile</div><div class="section-title">Pain & Fatigue Overview</div></div>', unsafe_allow_html=True)

            fig_p, ax_p = plt.subplots(figsize=(6, 3.4))
            fig_p.patch.set_facecolor("white")
            ax_p.set_facecolor("white")

            ns = ["Menstrual\npain", "Pelvic\npain", "Bowel\npain", "Bladder\npain", "Intercourse\npain", "Fatigue"]
            vs = [period_pain, pelvic_pain, bowel_pain, bladder_pain, intercourse_pain, fatigue * 2.5]
            colors = ["#b94545" if v >= 7 else "#a66b10" if v >= 4 else "#2f8b6c" for v in vs]

            bars = ax_p.barh(ns, vs, height=0.58, color=colors)
            ax_p.set_xlim(0, 10)
            ax_p.axvline(7, linestyle="--", alpha=0.25, linewidth=1)
            ax_p.axvline(4, linestyle="--", alpha=0.25, linewidth=1)
            ax_p.set_xlabel("Score (0 = none, 10 = worst)", fontsize=9)

            for bar, v in zip(bars, vs):
                if v > 0:
                    ax_p.text(v + 0.12, bar.get_y() + bar.get_height()/2, f"{v:.0f}", va="center", fontsize=8)

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
                f'"My period pain is {"severe and it stops me from functioning" if period_pain >= 7 else "moderate and clearly affecting my life" if period_pain >= 5 else "noticeable and concerning to me"}."',
                f'"My pelvic pain occurs {"almost every day, not just during my period" if pelvic_pain >= 9 else "frequently and more than just occasionally" if pelvic_pain >= 6 else "often enough that I am concerned"}."'
            ]
            if bowel_pain >= 3:
                scripts.append('"I have pain during bowel movements, especially around my period."')
            if bladder_pain >= 3:
                scripts.append('"I experience painful urination that keeps returning and does not feel fully explained."')
            if back_leg_val:
                scripts.append('"During my period, pain travels into my lower back or legs."')
            if family_hx_val:
                scripts.append('"I have a family history of endometriosis, so I want that taken seriously."')
            if worse_val >= 1:
                scripts.append('"My symptoms have been getting worse over time, not staying the same."')
            scripts.append('"I would like a referral to a gynaecologist or endometriosis specialist."')

            for s in scripts:
                st.markdown(f'<div class="say-card">{s}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            EndoScan is a research prototype and not a clinical diagnostic tool.
            A definitive diagnosis of endometriosis requires evaluation by a qualified clinician.
        </div>
        """, unsafe_allow_html=True)


# =========================
# DOCTOR TAB
# =========================
with tab_d:
    st.markdown("""
    <div class="panel">
        <div class="section-tag">Clinical Flow</div>
        <div class="section-title">Clinical Decision Support</div>
        <div class="section-sub">
            This panel is intended for clinician-style structured symptom entry and fast summary generation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns([1, 1], gap="large")

    with d1:
        da = st.number_input("Age", 14, 60, 28, key="da")
        db = st.number_input("BMI", 14.0, 45.0, 23.0, key="db", step=0.1)
        dp = st.slider("Dysmenorrhoea — menstrual pain", 0, 10, 6, key="dp_s")
        dpe = st.slider("Chronic pelvic pain", 0, 10, 5, key="dpe_s")
        ddc = st.slider("Dyschezia — bowel pain", 0, 10, 2, key="ddc_s")
        ddu = st.slider("Dysuria — bladder pain", 0, 10, 2, key="ddu_s")
        ddy = st.slider("Dyspareunia — intercourse pain", 0, 10, 3, key="ddy_s")
        dpd = st.slider("Pain disability index", 0, 70, 30, key="dpd_s")
        ddep = st.slider("Depression score", 0, 6, 2, key="ddep_s")
        danx = st.slider("Anxiety score", 0, 6, 2, key="danx_s")

        st.markdown("**Clinical Indicators**")
        dha = st.checkbox("Hormone level abnormality", key="dha")
        dinf = st.checkbox("Infertility / subfertility", key="dinf")
        dfh = st.checkbox("Family history of endometriosis", key="dfh")
        dhb = st.checkbox("Menorrhagia — soaking through in <1 hour", key="dhb")
        dir_ = st.checkbox("Menstrual irregularity", key="dir_")
        dbk = st.checkbox("Referred lumbar / sciatic pain", key="dbk")
        dov = st.checkbox("Mittelschmerz — mid-cycle pain", key="dov")
        dws = st.checkbox("Progressive worsening", key="dws")

        gen = st.button("📋 Generate Clinical Report", key="gen", use_container_width=True)

    with d2:
        if not gen:
            st.markdown("""
            <div class="soft-box" style="height:70%; display:flex; align-items:center; justify-content:center; text-align:center;">
                <div>
                    <div style="font-size:2.4rem;">📋</div>
                    <div style="font-family:'Cormorant Garamond', serif; font-size:1.8rem; margin-top:0.4rem;">Clinical report will appear here</div>
                    <div class="small-muted" style="max-width:320px; margin:0.6rem auto 0 auto;">
                        Enter the symptom values on the left and generate the report.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if gen:
            drp_raw, desrc = pred_endo(da, int(dir_), dpe, int(dha), int(dinf), db)
            if drp_raw is None:
                st.error("Diagnosis model not loaded.")
                st.stop()

            dep_ = drp_raw
            if dfh:
                dep_ = min(.97, dep_ * 1.35)
            if dws:
                dep_ = min(.97, dep_ * 1.15)
            if dov:
                dep_ = min(.97, dep_ * 1.05)

            drp = round(dep_ * 100)
            dpb_ = round(float(np.mean([dp, dpe, ddy, ddu, ddc])), 2)
            dms_ = 1 if (ddu >= 5 and ddc >= 5) else 0
            dcst_ = int(min(35, max(6, 12 + (ddep / 6) * 10 + (dp / 10) * 8)))
            dcl, dksrc = pred_cluster(dp, dpe, ddy, ddu, ddc, dpd, dcst_, dpb_, dms_)
            if dcl is None:
                dcl = 1
            dcp = CL[dcl]
            dpct_ = pdpct(dpd)

            _, drt, _, _ = risk_color(drp)

            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric("Endo Risk", f"{drp}%")
            dm2.metric("Tier", drt.upper())
            dm3.metric("Subtype", f"Cluster {dcl}")
            dm4.metric("PDI %ile", f"{dpct_}th")

            st.markdown(
                f'''
                <div class="soft-box" style="background:{dcp["bg"]}; border-color:{dcp["border"]};">
                    <div class="result-label">Clinical Assessment</div>
                    <div class="small-muted" style="color:#423352;">
                        This <strong>{da}-year-old</strong> patient has an estimated endometriosis risk of
                        <strong>{drp}%</strong> ({drt}) via <code>{desrc}</code> (raw {drp_raw:.3f} → adjusted {dep_:.3f}).
                        Symptom profile aligns with <strong>{dcp["name"]}</strong> via <code>{dksrc}</code>.
                        Pain disability is at the <strong>{dpct_}th percentile</strong>.
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown(
                f'''
                <div class="soft-box" style="background:{dcp["bg"]}; border-color:{dcp["border"]};">
                    <div class="result-label">Recommended Action</div>
                    <div style="font-weight:700; color:{dcp["color"]}; line-height:1.8;">
                        {dcp["action"]}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown("""
            <div class="disclaimer">
                Clinical support output only. Definitive diagnosis requires qualified clinical evaluation.
            </div>
            """, unsafe_allow_html=True)