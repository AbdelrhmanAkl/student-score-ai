from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "studentscore_ridge_model.joblib"
DATA_PATH = ROOT / "data" / "StudentPerformanceFactors.csv"
GITHUB_URL = "https://github.com/AbdelrhmanAkl/student-score-ai"

# Palette — calm, warm neutrals with a single deep-teal accent
INK, MUTED, LINE = "#14201F", "#6B7673", "#E6E3DC"
ACCENT, SOFT, TERRA = "#0F766E", "#E6F2EF", "#B4533C"

st.set_page_config(
    page_title="StudentScore AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Instrument+Serif&display=swap');

:root {--bg:#F6F5F2; --ink:#14201F; --muted:#6B7673; --line:#E3E0D8; --accent:#0F766E; --soft:#E6F2EF;}
html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp li {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;}
.stApp {background: var(--bg);}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stAppDeployButton"], #MainMenu, footer {display: none !important;}
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}

/* ---------- top bar & headline ---------- */
.topbar {display:flex; justify-content:space-between; align-items:center; margin-bottom: 2rem;}
.brand {display:flex; align-items:center; gap:.8rem;}
.logo {width:38px; height:38px; border-radius:11px; background:var(--ink); color:#fff; display:flex;
  align-items:center; justify-content:center; font-weight:700; font-size:1.05rem;}
.brand-name {font-weight:700; color:var(--ink); font-size:1rem; line-height:1.2;}
.brand-sub {color:var(--muted); font-size:.78rem;}
.gh {color:var(--ink) !important; text-decoration:none !important; font-weight:600; font-size:.85rem;
  border:1px solid var(--line); padding:.45rem .95rem; border-radius:999px; background:#fff;}
.gh:hover {border-color:var(--accent); color:var(--accent) !important;}
.headline {font-family:'Instrument Serif', Georgia, 'Times New Roman', serif; font-weight:400;
  font-size:3.6rem; line-height:1.04; letter-spacing:-1.2px; color:var(--ink); margin:0;}
.headline em {color:var(--accent); font-style:italic;}
.lead {color:var(--muted); font-size:1.02rem; margin:1rem 0 1.8rem; max-width:600px; line-height:1.6;}

/* ---------- cards (bordered containers that hold a .card-marker) ---------- */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-marker) {
  background:#fff; border:1px solid var(--line) !important; border-radius:18px !important;
  padding:1.3rem 1.5rem !important;
  box-shadow: 0 1px 2px rgba(20,32,31,.04), 0 14px 32px -26px rgba(20,32,31,.45);
}
div[data-testid="stElementContainer"]:has(.card-marker) {display:none;}
.eyebrow {font-size:.72rem; text-transform:uppercase; letter-spacing:1.4px; color:var(--muted); font-weight:600;}
.card-title {font-weight:650; font-size:1.05rem; color:var(--ink); margin:0 0 .15rem;}
.card-sub {color:var(--muted); font-size:.86rem; line-height:1.45;}

.bigscore {font-family:'Instrument Serif', Georgia, serif; font-size:6rem; line-height:1;
  letter-spacing:-3px; color:var(--ink); margin:.55rem 0 .5rem;}
.bigscore span {font-family:'Inter', sans-serif; font-size:1.1rem; color:#9AA3A0; letter-spacing:0; margin-left:.4rem; font-weight:500;}
.pill {display:inline-block; padding:.3rem .9rem; border-radius:999px; font-weight:600; font-size:.82rem;}
.meta {color:var(--muted); font-size:.92rem; margin-top:1rem; line-height:1.7;}
.meta b {color:var(--ink); font-weight:600;}
.track {height:6px; border-radius:999px; background:#EEECE6; margin-top:1.2rem; overflow:hidden;}
.track > div {height:100%; border-radius:999px;}

.lv {display:grid; grid-template-columns: 1.5fr 1fr auto; gap:1rem; align-items:center;
  padding:.8rem 0; border-bottom:1px solid #F0EEE9; font-size:.93rem; color:var(--ink);}
.lv:last-child {border-bottom:none;}
.lv-bar {height:6px; border-radius:999px; background:#EEECE6; overflow:hidden;}
.lv-bar > div {height:100%; background:var(--accent); border-radius:999px;}
.lv-g {font-weight:650; color:var(--accent); min-width:3rem; text-align:right;}

.stat {background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.2rem;}
.stat .v {font-family:'Instrument Serif', Georgia, serif; font-size:2.3rem; color:var(--ink); line-height:1.1; margin-top:.2rem;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {background:#fff; border-right:1px solid var(--line);}
section[data-testid="stSidebar"][data-collapsed="false"] {width:400px !important; min-width:400px !important;}
.side-title {font-weight:700; font-size:1.2rem; color:var(--ink); margin:0;}
.side-sub {color:var(--muted); font-size:.84rem; margin:.15rem 0 .6rem;}
div[data-testid="stExpander"] details {border:1px solid var(--line) !important; border-radius:14px !important; background:#fff !important;}
div[data-testid="stExpander"] summary {font-weight:600; background:transparent !important;}

/* ---------- widgets ---------- */
div[data-testid="stWidgetLabel"] p {font-size:.84rem !important; font-weight:550 !important; color:#2B3836;}
/* slider: Streamlit's default red -> teal (hue shift of rgb(255,75,75) to ~#0F766E) */
div[data-testid="stSlider"] [role="group"] {filter: hue-rotate(175deg) saturate(.65) brightness(.85);}
/* select boxes */
div[data-testid="stSelectbox"] [role="group"] {background:#F6F5F2 !important; border:1px solid var(--line) !important; border-radius:10px !important;}
div[data-testid="stSelectbox"] [role="group"]:focus-within {border-color:var(--accent) !important;}
/* buttons */
div[data-testid="stButton"], div[data-testid="stDownloadButton"] {width:100% !important;}
div[data-testid="stButton"] button {width:100%; border-radius:10px; border:1px solid var(--line); background:#fff;
  color:var(--ink); font-weight:550; font-size:.82rem; padding:.35rem .3rem;}
div[data-testid="stButton"] button:hover {border-color:var(--accent); color:var(--accent); background:var(--soft);}
div[data-testid="stDownloadButton"] button {width:100%; border-radius:12px; border:none; background:var(--ink); color:#fff; font-weight:600; padding:.7rem;}
div[data-testid="stDownloadButton"] button:hover {background:#000; color:#fff;}
/* tabs */
div[data-testid="stTab"] p {font-weight:600; font-size:.95rem;}
div[data-testid="stTab"][aria-selected="true"] p {color:var(--accent) !important;}
.react-aria-SelectionIndicator {background:var(--accent) !important;}

.footer {text-align:center; color:#9AA3A0; font-size:.82rem; margin-top:3rem;}
.footer a {color:var(--ink); text-decoration:none; font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)


def card_marker():
    """Standalone marker: the container that holds it is styled as a white card."""
    st.markdown('<div class="card-marker"></div>', unsafe_allow_html=True)


def card_head(title: str, sub: str = ""):
    st.markdown(
        f'<div class="card-title">{title}</div>' + (f'<div class="card-sub">{sub}</div>' if sub else ""),
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Model & data
# ----------------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error("Model file not found. Expected: models/studentscore_ridge_model.joblib")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


model = load_model()
df = load_data()
HAS_DATA = not df.empty and "Exam_Score" in df.columns
AVG_SCORE = float(df["Exam_Score"].mean()) if HAS_DATA else None

# (min, max, default) -- ranges come from the dataset when available
NUMERIC = {
    "Hours_Studied": (1, 44, 20),
    "Attendance": (60, 100, 80),
    "Sleep_Hours": (4, 10, 7),
    "Previous_Scores": (50, 100, 75),
    "Tutoring_Sessions": (0, 8, 1),
    "Physical_Activity": (0, 6, 3),
}
for _c, (_lo, _hi, _d) in list(NUMERIC.items()):
    if _c in df.columns:
        NUMERIC[_c] = (int(df[_c].min()), int(df[_c].max()), _d)

CATEGORICAL = {
    "Parental_Involvement": (["Low", "Medium", "High"], "Medium"),
    "Access_to_Resources": (["Low", "Medium", "High"], "Medium"),
    "Extracurricular_Activities": (["No", "Yes"], "Yes"),
    "Motivation_Level": (["Low", "Medium", "High"], "Medium"),
    "Internet_Access": (["No", "Yes"], "Yes"),
    "Family_Income": (["Low", "Medium", "High"], "Medium"),
    "Teacher_Quality": (["Low", "Medium", "High"], "Medium"),
    "School_Type": (["Public", "Private"], "Public"),
    "Peer_Influence": (["Negative", "Neutral", "Positive"], "Neutral"),
    "Learning_Disabilities": (["No", "Yes"], "No"),
    "Parental_Education_Level": (["High School", "College", "Postgraduate"], "College"),
    "Distance_from_Home": (["Near", "Moderate", "Far"], "Near"),
    "Gender": (["Male", "Female"], "Male"),
}
OPTIONS = {}
for _c, (_opts, _) in CATEGORICAL.items():
    opts = list(_opts)
    if _c in df.columns:
        opts += [v for v in df[_c].dropna().astype(str).unique() if v not in opts]
    OPTIONS[_c] = opts

FEATURE_ORDER = list(NUMERIC) + list(CATEGORICAL)

LABELS = {
    "Hours_Studied": "Hours studied / week",
    "Attendance": "Attendance (%)",
    "Sleep_Hours": "Sleep (hours / night)",
    "Previous_Scores": "Previous scores",
    "Tutoring_Sessions": "Tutoring sessions / month",
    "Physical_Activity": "Physical activity (hrs / week)",
    "Parental_Involvement": "Parental involvement",
    "Access_to_Resources": "Access to resources",
    "Extracurricular_Activities": "Extracurriculars",
    "Motivation_Level": "Motivation",
    "Internet_Access": "Internet access",
    "Family_Income": "Family income",
    "Teacher_Quality": "Teacher quality",
    "School_Type": "School type",
    "Peer_Influence": "Peer influence",
    "Learning_Disabilities": "Learning disability",
    "Parental_Education_Level": "Parent education",
    "Distance_from_Home": "Distance from home",
    "Gender": "Gender",
}


def label(col: str) -> str:
    return LABELS.get(col, col.replace("_", " "))


# ----------------------------------------------------------------------------
# Session state + presets
# ----------------------------------------------------------------------------
for c, (lo, hi, d) in NUMERIC.items():
    st.session_state.setdefault(f"in_{c}", min(max(d, lo), hi))
for c, (_, d) in CATEGORICAL.items():
    st.session_state.setdefault(f"in_{c}", d)

PRESETS = {
    "Strong": {
        "Hours_Studied": 30, "Attendance": 96, "Sleep_Hours": 8, "Previous_Scores": 90,
        "Tutoring_Sessions": 3, "Physical_Activity": 4, "Parental_Involvement": "High",
        "Access_to_Resources": "High", "Extracurricular_Activities": "Yes",
        "Motivation_Level": "High", "Internet_Access": "Yes", "Teacher_Quality": "High",
        "Peer_Influence": "Positive", "Parental_Education_Level": "Postgraduate",
        "Distance_from_Home": "Near",
    },
    "Typical": {
        "Hours_Studied": 20, "Attendance": 80, "Sleep_Hours": 7, "Previous_Scores": 75,
        "Tutoring_Sessions": 1, "Physical_Activity": 3, "Parental_Involvement": "Medium",
        "Access_to_Resources": "Medium", "Extracurricular_Activities": "Yes",
        "Motivation_Level": "Medium", "Internet_Access": "Yes", "Teacher_Quality": "Medium",
        "Peer_Influence": "Neutral", "Parental_Education_Level": "College",
        "Distance_from_Home": "Near",
    },
    "At risk": {
        "Hours_Studied": 8, "Attendance": 65, "Sleep_Hours": 5, "Previous_Scores": 55,
        "Tutoring_Sessions": 0, "Physical_Activity": 1, "Parental_Involvement": "Low",
        "Access_to_Resources": "Low", "Extracurricular_Activities": "No",
        "Motivation_Level": "Low", "Internet_Access": "No", "Teacher_Quality": "Low",
        "Peer_Influence": "Negative", "Parental_Education_Level": "High School",
        "Distance_from_Home": "Far",
    },
}


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        if k in NUMERIC:
            lo, hi, _ = NUMERIC[k]
            v = min(max(v, lo), hi)
        elif v not in OPTIONS[k]:
            continue
        st.session_state[f"in_{k}"] = v


def reset_all():
    for c, (lo, hi, d) in NUMERIC.items():
        st.session_state[f"in_{c}"] = min(max(d, lo), hi)
    for c, (_, d) in CATEGORICAL.items():
        st.session_state[f"in_{c}"] = d


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def current_values() -> dict:
    return {c: st.session_state[f"in_{c}"] for c in FEATURE_ORDER}


def predict_batch(rows: pd.DataFrame) -> np.ndarray:
    return model.predict(rows[FEATURE_ORDER])


def clamp(x: float) -> float:
    return float(min(max(x, 0.0), 100.0))


def grade(score: float):
    """(label, text colour, soft background)"""
    if score >= 90:
        return "Excellent", "#0F766E", "#DDF0EC"
    if score >= 80:
        return "Very good", "#2F7D5B", "#E1F1E8"
    if score >= 70:
        return "Good", "#2F6A8F", "#E1EDF5"
    if score >= 60:
        return "Fair", "#9A6B12", "#F7EBCF"
    return "Needs support", "#A4452F", "#F6E0DA"


def top_levers(values: dict, base: float, n: int = 5):
    ideas = []

    def num(col, delta, text):
        _, hi, _ = NUMERIC[col]
        new = min(values[col] + delta, hi)
        if new != values[col]:
            ideas.append((text, {col: new}))

    def cat(col, target, text):
        if values[col] != target and target in OPTIONS[col]:
            ideas.append((text, {col: target}))

    num("Hours_Studied", 5, "Study 5 more hours per week")
    num("Attendance", 10, "Raise attendance by 10 points")
    num("Tutoring_Sessions", 2, "Add 2 tutoring sessions a month")
    num("Physical_Activity", 1, "One more hour of physical activity")
    cat("Motivation_Level", "High", "High motivation")
    cat("Parental_Involvement", "High", "High parental involvement")
    cat("Access_to_Resources", "High", "High access to resources")
    cat("Teacher_Quality", "High", "High teacher quality")
    cat("Internet_Access", "Yes", "Reliable internet access")
    cat("Peer_Influence", "Positive", "Positive peer influence")

    if not ideas:
        return []
    rows = pd.DataFrame([{**values, **chg} for _, chg in ideas])
    preds = predict_batch(rows)
    out = [(t, float(p) - base) for (t, _), p in zip(ideas, preds)]
    out = [o for o in out if o[1] > 0.05]
    return sorted(out, key=lambda x: -x[1])[:n]


def sweep(values: dict, col: str) -> pd.DataFrame:
    lo, hi, _ = NUMERIC[col]
    xs = np.linspace(lo, hi, 30)
    rows = pd.DataFrame([{**values, col: float(x)} for x in xs])
    ys = np.clip(predict_batch(rows), 0, 100)
    return pd.DataFrame({"x": xs, "y": ys})


# ---------- charts (Altair, custom styled) ----------
def style(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            domain=False, tickColor=LINE, gridColor="#EFEDE8", labelColor=MUTED,
            titleColor=MUTED, labelFontSize=11, titleFontSize=11, titleFontWeight="normal",
        )
        .configure_axisX(grid=False)
    )


def whatif_chart(values: dict, col: str, score: float):
    data = sweep(values, col)
    lo_y, hi_y = float(data["y"].min()), float(data["y"].max())
    pad = max((hi_y - lo_y) * 0.3, 1.5)
    ylo, yhi = max(lo_y - pad, 0.0), min(hi_y + pad, 100.0)
    data["y0"] = ylo
    cur = pd.DataFrame({"x": [values[col]], "y": [score]})
    xs = alt.X("x:Q", title=label(col), scale=alt.Scale(nice=False))
    ys = alt.Y("y:Q", title=None, scale=alt.Scale(domain=[ylo, yhi], nice=False))
    base = alt.Chart(data).encode(x=xs, y=ys)
    area = base.mark_area(color=ACCENT, opacity=0.09, interpolate="monotone").encode(y2="y0:Q")
    line = base.mark_line(color=ACCENT, strokeWidth=2.5, interpolate="monotone", clip=True)
    dot = (
        alt.Chart(cur)
        .mark_point(filled=True, size=130, color=INK, opacity=1)
        .encode(x="x:Q", y=alt.Y("y:Q", scale=alt.Scale(domain=[ylo, yhi], nice=False)))
    )
    return style(alt.layer(area, line, dot).properties(height=200, width="container"))


def distribution_chart(score: float):
    vc = df["Exam_Score"].round().astype(int).value_counts().sort_index()
    vc = vc.reindex(range(int(vc.index.min()), int(vc.index.max()) + 1), fill_value=0)
    vc = vc.rolling(3, center=True, min_periods=1).mean().reset_index()
    vc.columns = ["score", "count"]
    dom = [int(vc["score"].min()) - 1, int(vc["score"].max()) + 1]
    base = alt.Chart(vc).encode(
        x=alt.X("score:Q", title="Exam score", scale=alt.Scale(domain=dom)),
        y=alt.Y("count:Q", axis=None),
    )
    area = base.mark_area(color=ACCENT, opacity=0.12, interpolate="monotone")
    line = base.mark_line(color=ACCENT, strokeWidth=2, interpolate="monotone")
    xs = min(max(score, dom[0]), dom[1])
    you = pd.DataFrame({"x": [xs], "t": [f"You · {score:.1f}"]})
    avg = pd.DataFrame({"x": [AVG_SCORE], "t": [f"Average · {AVG_SCORE:.1f}"]})
    you_right = xs >= AVG_SCORE
    r_you = alt.Chart(you).mark_rule(color=INK, strokeWidth=2).encode(x="x:Q")
    r_avg = alt.Chart(avg).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q")
    t_you = alt.Chart(you).mark_text(
        align="left" if you_right else "right", dx=6 if you_right else -6, dy=-10,
        color=INK, fontWeight="bold", fontSize=12,
    ).encode(x="x:Q", y=alt.value(0), text="t:N")
    t_avg = alt.Chart(avg).mark_text(
        align="right" if you_right else "left", dx=-6 if you_right else 6, dy=-10,
        color=MUTED, fontSize=12,
    ).encode(x="x:Q", y=alt.value(0), text="t:N")
    ch = alt.layer(area, line, r_avg, r_you, t_avg, t_you).properties(
        height=210, width="container", padding={"top": 28, "left": 4, "right": 10, "bottom": 4}
    )
    return style(ch)


def correlation_chart():
    corr = df.select_dtypes("number").corr()["Exam_Score"].drop("Exam_Score")
    cd = pd.DataFrame({"feature": [label(i) for i in corr.index], "r": corr.values.round(3)})
    ch = (
        alt.Chart(cd)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("r:Q", title="Correlation with exam score"),
            color=alt.condition(alt.datum.r > 0, alt.value(ACCENT), alt.value(TERRA)),
            tooltip=["feature", "r"],
        )
        .properties(height=230, width="container")
    )
    return style(ch)


# ----------------------------------------------------------------------------
# Sidebar — student profile
# ----------------------------------------------------------------------------
def slider(col):
    lo, hi, _ = NUMERIC[col]
    st.slider(label(col), lo, hi, key=f"in_{col}")


def select(col):
    st.selectbox(label(col), OPTIONS[col], key=f"in_{col}")


with st.sidebar:
    st.markdown(
        '<div class="side-title">Student profile</div>'
        '<div class="side-sub">Pick a starting point, then fine-tune.</div>',
        unsafe_allow_html=True,
    )
    bcols = st.columns(4)
    for i, pname in enumerate(PRESETS):
        bcols[i].button(pname, on_click=apply_preset, args=(pname,), key=f"preset_{i}")
    bcols[3].button("Reset", on_click=reset_all, key="reset_btn")
    st.write("")

    with st.expander("Academic", expanded=True):
        slider("Hours_Studied")
        slider("Attendance")
        slider("Previous_Scores")
        slider("Tutoring_Sessions")
        a, b = st.columns(2)
        with a:
            select("Motivation_Level")
        with b:
            select("Learning_Disabilities")

    with st.expander("Lifestyle", expanded=True):
        slider("Sleep_Hours")
        slider("Physical_Activity")
        a, b = st.columns(2)
        with a:
            select("Extracurricular_Activities")
        with b:
            select("Peer_Influence")

    with st.expander("Background", expanded=False):
        a, b = st.columns(2)
        with a:
            select("Parental_Involvement")
            select("Family_Income")
            select("Teacher_Quality")
            select("Internet_Access")
            select("Gender")
        with b:
            select("Parental_Education_Level")
            select("Access_to_Resources")
            select("School_Type")
            select("Distance_from_Home")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="topbar">'
    '<div class="brand"><div class="logo">S</div>'
    '<div><div class="brand-name">StudentScore AI</div>'
    '<div class="brand-sub">Exam score prediction · Ridge Regression</div></div></div>'
    f'<a class="gh" href="{GITHUB_URL}" target="_blank">GitHub ↗</a>'
    "</div>"
    '<div class="headline">Understand what moves<br>a student’s <em>exam score.</em></div>'
    '<div class="lead">A regression model trained on 6,607 student records. '
    "Adjust the profile in the sidebar and the prediction updates instantly.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Live prediction
# ----------------------------------------------------------------------------
values = current_values()
try:
    raw = float(predict_batch(pd.DataFrame([values]))[0])
    score = clamp(raw)
    ok = True
except Exception as exc:  # noqa: BLE001
    ok, err = False, exc

if not ok:
    st.error("Prediction failed. Check that the saved model matches the notebook features.")
    st.exception(err)
    st.stop()

gname, fg, bg = grade(score)

with st.container(border=True):
    card_marker()
    c1, c2 = st.columns([1, 1.45], gap="large")
    with c1:
        meta = []
        if AVG_SCORE is not None:
            diff = score - AVG_SCORE
            sign = "+" if diff >= 0 else "−"
            meta.append(f"<b>{sign}{abs(diff):.1f} pts</b> vs. dataset average ({AVG_SCORE:.1f})")
            pct = float((df["Exam_Score"] < score).mean() * 100)
            meta.append(f"Higher than <b>{pct:.0f}%</b> of students in the dataset")
        st.markdown(
            '<div class="eyebrow">Predicted exam score</div>'
            f'<div class="bigscore">{score:.1f}<span>/ 100</span></div>'
            f'<span class="pill" style="color:{fg};background:{bg}">{gname}</span>'
            f'<div class="meta">{"<br>".join(meta)}</div>'
            f'<div class="track"><div style="width:{score:.1f}%;background:{fg}"></div></div>',
            unsafe_allow_html=True,
        )
        if raw > 100:
            st.caption("Raw model output exceeded 100 and was capped for display.")
    with c2:
        if HAS_DATA:
            card_head("Where this student lands", "Distribution of exam scores across all students.")
            st.altair_chart(distribution_chart(score), theme=None)

st.write("")
tab_drivers, tab_whatif, tab_data, tab_about = st.tabs(["Key drivers", "What-if", "Dataset", "About"])

# ---------- Key drivers
with tab_drivers:
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        with st.container(border=True):
            card_marker()
            card_head("What would move this score most", "Estimated change in the prediction if one factor improved, everything else equal.")
            st.write("")
            levers = top_levers(values, raw)
            if levers:
                peak = max(g for _, g in levers)
                rows = "".join(
                    f'<div class="lv"><div>{t}</div>'
                    f'<div class="lv-bar"><div style="width:{g / peak * 100:.0f}%"></div></div>'
                    f'<div class="lv-g">+{g:.1f}</div></div>'
                    for t, g in levers
                )
                st.markdown(rows, unsafe_allow_html=True)
            else:
                st.info("This profile is already at the top of the tested ranges.")
            st.caption("Statistical associations learned by the model — not guaranteed causal effects.")
    with right:
        with st.container(border=True):
            card_marker()
            card_head("Export", "Save this profile and its prediction as a CSV.")
            st.write("")
            report = pd.DataFrame([{**values, "Predicted_Exam_Score": round(score, 2)}])
            st.download_button(
                "Download prediction (CSV)",
                report.to_csv(index=False).encode("utf-8"),
                file_name="studentscore_prediction.csv",
                mime="text/csv",
            )

# ---------- What-if
with tab_whatif:
    st.caption("Each curve re-runs the model while varying one input. The dark dot is the current profile.")
    try:
        pairs = [("Hours_Studied", "Attendance"), ("Previous_Scores", "Sleep_Hours")]
        for a_col, b_col in pairs:
            ca, cb = st.columns(2, gap="large")
            for holder, col in ((ca, a_col), (cb, b_col)):
                with holder:
                    with st.container(border=True):
                        card_marker()
                        card_head(label(col), "Predicted exam score")
                        st.altair_chart(whatif_chart(values, col, score), theme=None)
    except Exception as exc:  # noqa: BLE001
        st.warning("What-if charts are unavailable for this model.")
        st.exception(exc)

# ---------- Dataset
with tab_data:
    if not HAS_DATA:
        st.info("Dataset file not found at data/StudentPerformanceFactors.csv.")
    else:
        stats = [
            ("Students", f"{len(df):,}"),
            ("Average score", f"{df['Exam_Score'].mean():.1f}"),
            ("Lowest", f"{df['Exam_Score'].min():.0f}"),
            ("Highest", f"{df['Exam_Score'].max():.0f}"),
        ]
        cols = st.columns(4)
        for col_, (k, v) in zip(cols, stats):
            col_.markdown(f'<div class="stat"><div class="eyebrow">{k}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)
        st.write("")
        d1, d2 = st.columns(2, gap="large")
        with d1:
            with st.container(border=True):
                card_marker()
                card_head("Score distribution", "Number of students per exam score.")
                st.altair_chart(distribution_chart(score), theme=None)
        with d2:
            with st.container(border=True):
                card_marker()
                card_head("Numeric factors vs. score", "Pearson correlation with exam score.")
                st.altair_chart(correlation_chart(), theme=None)

# ---------- About
with tab_about:
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            card_marker()
            card_head("How it works", "From inputs to a prediction in four steps.")
            st.markdown(
                "1. **Input** — academic, lifestyle and background features.\n"
                "2. **Preprocessing** — the saved scikit-learn pipeline encodes and scales the data.\n"
                "3. **Prediction** — a Ridge Regression model estimates the exam score.\n"
                "4. **Insights** — what-if sweeps re-run the model to show how each input moves the score."
            )
            st.markdown("`Python` · `scikit-learn` · `Pandas` · `Joblib` · `Altair` · `Streamlit`")
    with b:
        with st.container(border=True):
            card_marker()
            card_head("Responsible use", "Please read before relying on a prediction.")
            st.markdown(
                "- Predictions are **estimates**, not guaranteed results.\n"
                "- Accuracy depends on the training data's quality and representativeness.\n"
                "- Statistical relationships are **not** proof of causation.\n"
                "- Not intended as the sole basis for high-impact academic decisions."
            )
            st.link_button("View source on GitHub", GITHUB_URL)

st.markdown(
    f'<div class="footer">Designed &amp; built by <a href="{GITHUB_URL}" target="_blank">Abdelrahman Akl</a>'
    " · StudentScore AI</div>",
    unsafe_allow_html=True,
)
