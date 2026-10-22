from pathlib import Path

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

st.set_page_config(
    page_title="StudentScore AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {background: radial-gradient(1200px 500px at 50% -80px, #e0e7ff 0%, #f8fafc 60%) fixed;}
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1360px;}

/* ---------- hero ---------- */
.hero {
  background: linear-gradient(115deg, #4338ca 0%, #6d28d9 50%, #be185d 115%);
  border-radius: 28px; padding: 3rem 2.5rem 2.6rem; color: #fff; text-align: center;
  box-shadow: 0 30px 60px -30px rgba(67,56,202,.65); position: relative; overflow: hidden;
}
.hero:before {
  content: ""; position: absolute; width: 420px; height: 420px; right: -120px; top: -180px;
  background: radial-gradient(circle, rgba(255,255,255,.22), transparent 65%);
}
.hero h1 {margin: 0; font-size: 3.2rem; font-weight: 800; letter-spacing: -1px; color: #fff; padding: 0;}
.hero p {margin: .7rem auto 0; font-size: 1.2rem; opacity: .93; color: #fff; max-width: 680px;}
.chips {margin-top: 1.4rem; display: flex; gap: .6rem; flex-wrap: wrap; justify-content: center;}
.chip {
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.32);
  padding: .35rem .95rem; border-radius: 999px; font-size: .85rem; font-weight: 600; color: #fff;
}

/* ---------- stat strip ---------- */
.stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.3rem 0 .4rem;}
.stat {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.05rem 1.2rem;
  box-shadow: 0 10px 28px -20px rgba(15,23,42,.35);
}
.stat .k {font-size: .72rem; text-transform: uppercase; letter-spacing: 1.2px; color: #64748b; font-weight: 700;}
.stat .v {font-size: 1.55rem; font-weight: 800; color: #0f172a; margin-top: .15rem;}
@media (max-width: 800px) {.stats {grid-template-columns: repeat(2, 1fr);} .hero h1 {font-size: 2.2rem;}}

/* ---------- cards ---------- */
.card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 24px;
  padding: 1.6rem 1.6rem; box-shadow: 0 14px 36px -22px rgba(15,23,42,.35);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.sec-title) {
  background: #fff; border-radius: 24px; border: 1px solid #e2e8f0;
  box-shadow: 0 14px 36px -22px rgba(15,23,42,.35); padding: .6rem .8rem;
}
.sec-title {font-size: 1.15rem; font-weight: 800; color: #0f172a; margin: .2rem 0 .1rem;}
.sec-sub {font-size: .85rem; color: #64748b; margin-bottom: .6rem;}

.score-label {font-size: .8rem; text-transform: uppercase; letter-spacing: 1.6px; color: #64748b; font-weight: 700; text-align: center;}
.gauge-box {position: relative; max-width: 400px; margin: .4rem auto 0;}
.gauge-num {position: absolute; left: 0; right: 0; bottom: 2px; text-align: center;
  font-size: 4.2rem; font-weight: 800; line-height: 1; color: #0f172a; letter-spacing: -2px;}
.gauge-num span {font-size: 1.2rem; color: #94a3b8; font-weight: 600; letter-spacing: 0;}
.center {text-align: center;}
.badge {display: inline-block; margin-top: 1rem; padding: .4rem 1.2rem; border-radius: 999px; font-weight: 700; font-size: .95rem;}
.delta {margin-top: .8rem; font-size: .95rem; color: #475569;}

.lever {display: flex; justify-content: space-between; align-items: center;
  padding: .7rem .2rem; border-bottom: 1px dashed #e2e8f0; font-size: .98rem; color: #334155;}
.lever:last-child {border-bottom: none;}
.lever .gain {color: #059669; font-weight: 700; background: #ecfdf5; padding: .15rem .7rem; border-radius: 10px;}
.section-title {font-weight: 800; color: #0f172a; margin: 0 0 .5rem; font-size: 1.1rem;}
.muted {color: #64748b; font-size: .88rem;}

/* ---------- widgets ---------- */
div.stButton > button {width: 100%; border-radius: 14px; font-weight: 600; padding: .6rem .8rem; border: 1px solid #c7d2fe; background: #fff;}
div.stButton > button:hover {border-color: #6366f1; color: #4f46e5; background: #eef2ff;}
div.stDownloadButton > button {width: 100%; border-radius: 14px; font-weight: 700; padding: .65rem; background: #4f46e5; color: #fff; border: none;}
div.stDownloadButton > button:hover {background: #4338ca; color: #fff;}
label p, div[data-testid="stWidgetLabel"] p {font-size: .95rem !important; font-weight: 600 !important; color: #1e293b;}
div[data-baseweb="select"] > div {min-height: 46px; border-radius: 12px;}
div[data-baseweb="slider"] div[role="slider"] {background-color: #6366f1 !important; box-shadow: none !important;}
div[data-testid="stSliderThumbValue"], div[data-testid="stThumbValue"] {color: #4f46e5 !important; font-weight: 700;}
button[data-baseweb="tab"] {font-size: 1.02rem; font-weight: 600; padding: .6rem 1.1rem;}
div[data-baseweb="tab-highlight"] {background-color: #6366f1 !important;}

/* sticky result column */
div[data-testid="stColumn"]:has(.sticky-anchor), div[data-testid="column"]:has(.sticky-anchor) {
  position: sticky; top: 1rem; align-self: flex-start;
}

.footer {text-align: center; color: #94a3b8; font-size: .85rem; margin-top: 2.5rem;}
.footer a {color: #6366f1; text-decoration: none; font-weight: 600;}
</style>
""",
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

# (min, max, default) -- ranges are taken from the dataset when available
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
# keep canonical order, but add any extra value found in the dataset
OPTIONS = {}
for _c, (_opts, _) in CATEGORICAL.items():
    opts = list(_opts)
    if _c in df.columns:
        opts += [v for v in df[_c].dropna().astype(str).unique() if v not in opts]
    OPTIONS[_c] = opts

FEATURE_ORDER = list(NUMERIC) + list(CATEGORICAL)


def label(col: str) -> str:
    return col.replace("_", " ")


# ----------------------------------------------------------------------------
# Session state + presets
# ----------------------------------------------------------------------------
for c, (lo, hi, d) in NUMERIC.items():
    st.session_state.setdefault(f"in_{c}", min(max(d, lo), hi))
for c, (_, d) in CATEGORICAL.items():
    st.session_state.setdefault(f"in_{c}", d)

PRESETS = {
    "🌟 High achiever": {
        "Hours_Studied": 30, "Attendance": 96, "Sleep_Hours": 8, "Previous_Scores": 90,
        "Tutoring_Sessions": 3, "Physical_Activity": 4, "Parental_Involvement": "High",
        "Access_to_Resources": "High", "Extracurricular_Activities": "Yes",
        "Motivation_Level": "High", "Internet_Access": "Yes", "Teacher_Quality": "High",
        "Peer_Influence": "Positive", "Parental_Education_Level": "Postgraduate",
        "Distance_from_Home": "Near",
    },
    "⚖️ Average student": {
        "Hours_Studied": 20, "Attendance": 80, "Sleep_Hours": 7, "Previous_Scores": 75,
        "Tutoring_Sessions": 1, "Physical_Activity": 3, "Parental_Involvement": "Medium",
        "Access_to_Resources": "Medium", "Extracurricular_Activities": "Yes",
        "Motivation_Level": "Medium", "Internet_Access": "Yes", "Teacher_Quality": "Medium",
        "Peer_Influence": "Neutral", "Parental_Education_Level": "College",
        "Distance_from_Home": "Near",
    },
    "⚠️ Needs support": {
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
    # (label, text colour, badge background, arc colour)
    if score >= 90:
        return "Excellent", "#047857", "#d1fae5", "#10b981"
    if score >= 80:
        return "Very good", "#0f766e", "#ccfbf1", "#14b8a6"
    if score >= 70:
        return "Good", "#1d4ed8", "#dbeafe", "#6366f1"
    if score >= 60:
        return "Fair", "#b45309", "#fef3c7", "#f59e0b"
    return "Needs support", "#b91c1c", "#fee2e2", "#ef4444"


def gauge_svg(score: float, color: str) -> str:
    arc = np.pi * 80
    filled = arc * score / 100
    return (
        '<svg viewBox="0 0 200 112" width="100%">'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e5e7eb" '
        'stroke-width="17" stroke-linecap="round"/>'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="' + color + '" '
        'stroke-width="17" stroke-linecap="round" '
        f'stroke-dasharray="{filled:.1f} {arc:.1f}"/>'
        "</svg>"
    )


def top_levers(values: dict, base: float, n: int = 4):
    ideas = []

    def num(col, delta, text):
        lo, hi, _ = NUMERIC[col]
        new = min(values[col] + delta, hi)
        if new != values[col]:
            ideas.append((text, {col: new}))

    def cat(col, target, text):
        if values[col] != target and target in OPTIONS[col]:
            ideas.append((text, {col: target}))

    num("Hours_Studied", 5, "Study 5 more hours per week")
    num("Attendance", 10, "Raise attendance by 10 points")
    num("Tutoring_Sessions", 2, "Add 2 tutoring sessions")
    num("Physical_Activity", 1, "One more hour of physical activity")
    cat("Motivation_Level", "High", "High motivation level")
    cat("Parental_Involvement", "High", "High parental involvement")
    cat("Access_to_Resources", "High", "High access to resources")
    cat("Teacher_Quality", "High", "High teacher quality")
    cat("Internet_Access", "Yes", "Having internet access")
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
    return pd.DataFrame({"Predicted score": ys}, index=pd.Index(np.round(xs, 1), name=label(col)))


# ----------------------------------------------------------------------------
# Hero + stat strip
# ----------------------------------------------------------------------------
n_rows = f"{len(df):,}" if not df.empty else "6,607"
avg_txt = f"{AVG_SCORE:.1f}" if AVG_SCORE is not None else "—"
st.markdown(
    """
<div class="hero">
  <h1>🎓 StudentScore AI</h1>
  <p>Estimate a student's exam score from study habits, lifestyle and background — instantly, as you type.</p>
  <div class="chips">
    <span class="chip">Ridge Regression</span>
    <span class="chip">scikit-learn Pipeline</span>
    <span class="chip">Live predictions</span>
    <span class="chip">What-if analysis</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="stats">'
    '<div class="stat"><div class="k">Algorithm</div><div class="v">Ridge</div></div>'
    f'<div class="stat"><div class="k">Training records</div><div class="v">{n_rows}</div></div>'
    f'<div class="stat"><div class="k">Input features</div><div class="v">{len(FEATURE_ORDER)}</div></div>'
    f'<div class="stat"><div class="k">Avg. exam score</div><div class="v">{avg_txt}</div></div>'
    "</div>",
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📈 Insights", "ℹ️ About"])

# ----------------------------------------------------------------------------
# Predict tab
# ----------------------------------------------------------------------------
with tab_predict:
    st.markdown(
        '<div class="muted" style="margin:.4rem 0 .5rem">⚡ Quick start — load a sample profile, then tweak any input.</div>',
        unsafe_allow_html=True,
    )
    pcols = st.columns(len(PRESETS) + 1)
    for i, pname in enumerate(PRESETS):
        pcols[i].button(pname, on_click=apply_preset, args=(pname,), key=f"preset_{i}")
    pcols[-1].button("↺ Reset", on_click=reset_all, key="reset_btn")

    st.write("")
    left, right = st.columns([1.6, 1], gap="large")

    def slider(col):
        lo, hi, _ = NUMERIC[col]
        st.slider(label(col), lo, hi, key=f"in_{col}")

    def select(col):
        st.selectbox(label(col), OPTIONS[col], key=f"in_{col}")

    # ---- inputs (all visible, no hidden tabs)
    with left:
        with st.container(border=True):
            st.markdown('<div class="sec-title">📚 Academic</div><div class="sec-sub">Study habits and past performance</div>', unsafe_allow_html=True)
            a, b = st.columns(2, gap="large")
            with a:
                slider("Hours_Studied")
                slider("Previous_Scores")
                select("Motivation_Level")
            with b:
                slider("Attendance")
                slider("Tutoring_Sessions")
                select("Learning_Disabilities")

        with st.container(border=True):
            st.markdown('<div class="sec-title">🌿 Lifestyle</div><div class="sec-sub">Rest, activity and social environment</div>', unsafe_allow_html=True)
            a, b = st.columns(2, gap="large")
            with a:
                slider("Sleep_Hours")
                select("Extracurricular_Activities")
            with b:
                slider("Physical_Activity")
                select("Peer_Influence")

        with st.container(border=True):
            st.markdown('<div class="sec-title">🏠 Background</div><div class="sec-sub">Family, school and access</div>', unsafe_allow_html=True)
            a, b, c = st.columns(3, gap="medium")
            with a:
                select("Parental_Involvement")
                select("Family_Income")
                select("Gender")
            with b:
                select("Parental_Education_Level")
                select("Access_to_Resources")
                select("Distance_from_Home")
            with c:
                select("Teacher_Quality")
                select("School_Type")
                select("Internet_Access")

    # ---- live result
    values = current_values()
    try:
        raw = float(predict_batch(pd.DataFrame([values]))[0])
        score = clamp(raw)
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        err = exc

    with right:
        st.markdown('<div class="sticky-anchor"></div>', unsafe_allow_html=True)
        if not ok:
            st.error("Prediction failed. Check that the saved model matches the notebook features.")
            st.exception(err)
        else:
            gname, fg, bg, arc_color = grade(score)
            delta_html = ""
            if AVG_SCORE is not None:
                diff = score - AVG_SCORE
                sign = "+" if diff >= 0 else ""
                clr = "#059669" if diff >= 0 else "#dc2626"
                delta_html = (
                    f'<div class="delta"><b style="color:{clr}">{sign}{diff:.1f} pts</b> '
                    f"vs dataset average ({AVG_SCORE:.1f})</div>"
                )
            st.markdown(
                '<div class="card center">'
                '<div class="score-label">Predicted exam score</div>'
                '<div class="gauge-box">'
                + gauge_svg(score, arc_color)
                + f'<div class="gauge-num">{score:.1f}<span> / 100</span></div></div>'
                f'<div class="badge" style="color:{fg};background:{bg}">{gname}</div>'
                + delta_html
                + "</div>",
                unsafe_allow_html=True,
            )
            if raw > 100:
                st.caption("The raw model output exceeded 100 and was capped for display.")

            levers = top_levers(values, raw)
            if levers:
                items = "".join(
                    f'<div class="lever"><span>{t}</span><span class="gain">+{g:.1f}</span></div>'
                    for t, g in levers
                )
                st.markdown(
                    '<div class="card" style="margin-top:1rem">'
                    '<div class="section-title">🎯 Biggest model-estimated levers</div>'
                    + items
                    + '<div class="muted" style="margin-top:.7rem">Statistical associations learned by the model — not guaranteed causal effects.</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.write("")
            report = pd.DataFrame([{**values, "Predicted_Exam_Score": round(score, 2)}])
            st.download_button(
                "⬇️ Download this prediction (CSV)",
                report.to_csv(index=False).encode("utf-8"),
                file_name="studentscore_prediction.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------------
# Insights tab
# ----------------------------------------------------------------------------
with tab_insights:
    st.markdown("#### What-if explorer")
    st.caption("See how the predicted score changes as one input varies while everything else stays as set in the Predict tab.")
    try:
        vals = current_values()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hours studied → score**")
            st.line_chart(sweep(vals, "Hours_Studied"))
        with c2:
            st.markdown("**Attendance → score**")
            st.line_chart(sweep(vals, "Attendance"))
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Previous scores → score**")
            st.line_chart(sweep(vals, "Previous_Scores"))
        with c4:
            st.markdown("**Sleep hours → score**")
            st.line_chart(sweep(vals, "Sleep_Hours"))
    except Exception as exc:  # noqa: BLE001
        st.warning("What-if charts are unavailable for this model.")
        st.exception(exc)

    if HAS_DATA:
        st.divider()
        st.markdown("#### Dataset overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Students", f"{len(df):,}")
        m2.metric("Avg. score", f"{df['Exam_Score'].mean():.1f}")
        m3.metric("Min score", f"{df['Exam_Score'].min():.0f}")
        m4.metric("Max score", f"{df['Exam_Score'].max():.0f}")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Exam score distribution**")
            counts, edges = np.histogram(df["Exam_Score"], bins=15)
            hist = pd.DataFrame(
                {"Students": counts},
                index=[f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(counts))],
            )
            st.bar_chart(hist)
        with d2:
            st.markdown("**Correlation with exam score**")
            corr = (
                df.select_dtypes(include="number").corr()["Exam_Score"].drop("Exam_Score").sort_values()
            )
            st.bar_chart(corr.rename("Correlation"))

# ----------------------------------------------------------------------------
# About tab
# ----------------------------------------------------------------------------
with tab_about:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### How it works")
        st.markdown(
            "1. **Input** — academic, lifestyle and background features.\n"
            "2. **Preprocessing** — the saved scikit-learn pipeline encodes and scales the data.\n"
            "3. **Prediction** — a Ridge Regression model estimates the exam score.\n"
            "4. **Insights** — what-if sweeps re-run the model to show how each input moves the score."
        )
        st.markdown("#### Tech stack")
        st.markdown("`Python` · `scikit-learn` · `Ridge Regression` · `Pandas` · `Joblib` · `Streamlit`")
    with b:
        st.markdown("#### Responsible use")
        st.markdown(
            "- Predictions are **estimates**, not guaranteed results.\n"
            "- Accuracy depends on the training data's quality and representativeness.\n"
            "- Statistical relationships are **not** proof of causation.\n"
            "- Not intended as the sole basis for high-impact academic decisions."
        )
        st.link_button("View source on GitHub", GITHUB_URL)

st.markdown(
    f'<div class="footer">Built by <b>Abdelrahman Akl</b> · '
    f'<a href="{GITHUB_URL}" target="_blank">GitHub</a> · StudentScore AI · Ridge Regression</div>',
    unsafe_allow_html=True,
)
