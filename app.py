import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ------------------------------------------------------------
# PAGE CONFIGURATION + VISUAL THEME
# ------------------------------------------------------------
st.set_page_config(
    page_title="ChurnIQ | Telecom Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --navy:#101a2c; --panel:#17243a; --blue:#3978f6; --muted:#8190a8; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#0b1220 0%,#101b2e 55%,#0d1728 100%); color:#edf3ff; }
[data-testid="stHeader"] { background:rgba(0,0,0,0); }
[data-testid="stSidebar"] { background:#0b1424; border-right:1px solid #24344d; }
[data-testid="stSidebar"] * { color:#e7eefb; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:#f4f7ff !important; letter-spacing:-.5px; }
p, label, .stMarkdown { color:#d2dced; }
.hero { padding:26px 30px; border:1px solid #2b4162; border-radius:20px; background:linear-gradient(115deg,#172b49 0%,#15233a 60%,#1c3152 100%); margin-bottom:22px; }
.hero-kicker { color:#8fb5ff; font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.hero-title { font-family:'Manrope',sans-serif; font-size:32px; line-height:1.2; font-weight:800; color:#fff; margin:8px 0 7px; }
.hero-copy { color:#b6c6df; font-size:14px; margin:0; }
.section-label { color:#8fb5ff; font-size:11px; font-weight:700; letter-spacing:1.6px; text-transform:uppercase; margin:8px 0 5px; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#17263d,#142137); border:1px solid #2a3e5b; padding:17px 18px; border-radius:15px; min-height:112px; }
div[data-testid="stMetricLabel"] p { color:#9fb0c9 !important; font-size:12px !important; }
div[data-testid="stMetricValue"] { color:#f5f8ff !important; font-family:'Manrope',sans-serif; }
div[data-testid="stMetricDelta"] { color:#77d6ae !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#2a3e5b !important; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #2a3e5b; }
.stTabs [data-baseweb="tab"] { background:#142137; border-radius:9px 9px 0 0; padding:10px 18px; color:#b8c8df; }
.stTabs [aria-selected="true"] { background:#234477 !important; color:#fff !important; }
.stButton>button, .stFormSubmitButton>button { background:linear-gradient(90deg,#3978f6,#5595ff); color:white; border:0; border-radius:10px; padding:.62rem 1.1rem; font-weight:700; }
.stButton>button:hover, .stFormSubmitButton>button:hover { background:linear-gradient(90deg,#2866e5,#4385f4); color:white; border:0; }
.stSelectbox div[data-baseweb="select"]>div, .stTextInput input, .stNumberInput input, [data-testid="stFileUploader"] section { background:#142137; border-color:#304663; color:#edf3ff; }
[data-testid="stDataFrame"] { border:1px solid #2a3e5b; border-radius:12px; overflow:hidden; }
hr { border-color:#2a3e5b; }
.small-note { color:#91a2bc; font-size:12px; }
.status-pill { display:inline-block; padding:5px 10px; border-radius:30px; background:#173e3a; color:#8de0bd; font-size:11px; font-weight:700; }
.result-box { padding:22px; border-radius:16px; border:1px solid #35547c; background:#172b46; }
footer { visibility:hidden; }

/* ---------------------------------------------------------
   SUBTLE MOTION: gentle entrance, hover lift and transitions
   --------------------------------------------------------- */
@keyframes churnFadeUp {
  from { opacity: 0; transform: translateY(9px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes churnGlow {
  0%, 100% { border-color: #2b4162; }
  50%      { border-color: #3a5d8b; }
}
.hero {
  animation: churnFadeUp .65s ease-out both, churnGlow 7s ease-in-out 1s infinite;
  background-size: 180% 180%;
  transition: transform .25s ease, border-color .25s ease;
}
.hero:hover { transform: translateY(-2px); }
section.main > div { animation: churnFadeUp .55s ease-out both; }
div[data-testid="stMetric"] {
  animation: churnFadeUp .55s ease-out both;
  transition: transform .22s ease, border-color .22s ease, background .22s ease;
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: #527db5;
  background: linear-gradient(145deg,#1b2d49,#172942);
}
.result-box {
  transition: transform .22s ease, border-color .22s ease, background .22s ease;
  animation: churnFadeUp .55s ease-out both;
}
.result-box:hover { transform: translateY(-2px); border-color: #527db5; }
.stButton > button, .stFormSubmitButton > button {
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(57,120,246,.20);
  filter: brightness(1.06);
}
.stTabs [data-baseweb="tab"] { transition: background .2s ease, color .2s ease; }
[data-testid="stSidebar"] [role="radiogroup"] label { transition: background .18s ease, transform .18s ease; border-radius: 8px; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #172943; transform: translateX(2px); }
[data-testid="stProgressBar"] > div > div { transition: width .7s ease-in-out; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .01ms !important;
    scroll-behavior: auto !important;
  }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR: DATASET + CONFIGURATION + NAVIGATION
# ------------------------------------------------------------
st.sidebar.markdown("<div style='font-family:Manrope;font-size:23px;font-weight:800;color:#fff'>📡 Churn<span style='color:#70a5ff'>IQ</span></div><div style='color:#8fa2bf;font-size:11px;letter-spacing:1.5px;margin-bottom:22px'>TELECOM ANALYTICS PLATFORM</div>", unsafe_allow_html=True)
st.sidebar.markdown("### Dataset source")
uploaded_file = st.sidebar.file_uploader("Upload telecom CSV", type=["csv"])

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
        st.sidebar.success("Uploaded dataset loaded")
    else:
        df = pd.read_csv("Telco-Customer-Churn.csv")
        source_name = "IBM Telco Customer Churn"
        st.sidebar.info("Using default IBM dataset")
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("The uploaded dataset must contain customer records and at least two columns.")
    st.stop()

st.sidebar.markdown("### Churn configuration")
default_target = next((c for c in df.columns if c.lower() in ["churn", "class", "customer_status", "churn_status"]), df.columns[-1])
target_col = st.sidebar.selectbox("Target column", options=list(df.columns), index=list(df.columns).index(default_target))
target_values = df[target_col].dropna().unique().tolist()
if len(target_values) != 2:
    st.error("Please select a target column containing exactly two classes.")
    st.stop()

preferred_positive = next((v for v in target_values if str(v).strip().lower() in ["yes", "churn", "churned", "true", "1", "left"]), target_values[-1])
positive_index = target_values.index(preferred_positive)
positive_label = st.sidebar.selectbox("Value representing churn", options=target_values, index=positive_index)

st.sidebar.markdown("---")
page = st.sidebar.radio("NAVIGATION", ["Overview", "Explore Data", "Model & Evaluation", "Predict Customer"], label_visibility="visible")
st.sidebar.markdown("---")
st.sidebar.markdown(f"<div class='small-note'>Current dataset<br><b>{source_name}</b><br>{len(df):,} records · {len(df.columns)} columns</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TARGET + FEATURE PREPARATION
# ------------------------------------------------------------
df = df.dropna(subset=[target_col]).copy()
df["Churn"] = df[target_col].apply(lambda value: "Yes" if value == positive_label else "No")

feature_cols = [c for c in df.columns if c not in [target_col, "Churn"]]
for col in feature_cols.copy():
    if "date" in col.lower():
        parsed = pd.to_datetime(df[col], errors="coerce")
        df[col + "_year"] = parsed.dt.year
        df[col + "_month"] = parsed.dt.month
        df.drop(columns=[col], inplace=True)

feature_cols = [c for c in df.columns if c not in [target_col, "Churn"]]
for col in feature_cols:
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().mean() >= 0.8:
        df[col] = converted

id_cols = [c for c in feature_cols if c.lower().replace("_", "").replace(" ", "").replace("-", "").endswith(("id", "key", "ref", "number", "uuid"))]
feature_cols = [c for c in feature_cols if c not in id_cols]
feature_cols = [c for c in feature_cols if not df[c].isna().all()]
if not feature_cols:
    st.error("No usable customer features were found in this dataset.")
    st.stop()

X = df[feature_cols].copy()
y = (df["Churn"] == "Yes").astype(int)
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

if len(df) < 10:
    st.error("Dataset is too small. Please upload at least 10 records.")
    st.stop()
if y.nunique() != 2 or y.value_counts().min() < 2:
    st.error("The selected target must contain at least two records in each of its two classes.")
    st.stop()

# Build and train the same Logistic Regression pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
transformers = []
if numerical_cols:
    transformers.append(("num", num_pipeline, numerical_cols))
if categorical_cols:
    transformers.append(("cat", cat_pipeline, categorical_cols))
preprocessor = ColumnTransformer(transformers=transformers)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
try:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]
except Exception as exc:
    st.error(f"The model could not be trained on this dataset: {exc}")
    st.stop()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
churn_count = int((df["Churn"] == "Yes").sum())
churn_rate = churn_count / len(df) * 100

# ------------------------------------------------------------
# COMMON HEADER
# ------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div class="hero-kicker">CUSTOMER INTELLIGENCE · MACHINE LEARNING</div>
  <div class="hero-title">Telecom Customer Churn Prediction</div>
  <p class="hero-copy">A flexible data pipeline to explore customer behaviour, evaluate churn patterns and predict customer retention.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# PAGE 1: OVERVIEW
# ------------------------------------------------------------
if page == "Overview":
    st.markdown("<div class='section-label'>Executive summary</div>", unsafe_allow_html=True)
    st.header("Customer overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Churned customers", f"{churn_count:,}")
    c3.metric("Churn rate", f"{churn_rate:.1f}%")
    c4.metric("Model accuracy", f"{accuracy*100:.1f}%")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Customer churn distribution")
        distribution = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
        st.bar_chart(distribution, color="#4f8df7")
        st.caption("Number of customers classified as churn or non-churn.")
    with right:
        st.subheader("Dataset health")
        missing_total = int(df[feature_cols].isna().sum().sum())
        missing_pct = missing_total / max(1, df[feature_cols].size) * 100
        h1, h2 = st.columns(2)
        h1.metric("Missing cells", f"{missing_total:,}")
        h2.metric("Feature columns", len(feature_cols))
        st.progress(min(missing_pct / 100, 1.0))
        st.caption(f"{missing_pct:.2f}% of feature cells are missing before preprocessing.")
        st.markdown("<span class='status-pill'>● MODEL TRAINED</span> &nbsp; <span class='status-pill'>● DATASET READY</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Model snapshot")
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{precision*100:.1f}%")
    m2.metric("Recall", f"{recall*100:.1f}%")
    m3.metric("F1-score", f"{f1*100:.1f}%")
    st.info("Use the navigation menu to explore the dataset, inspect model performance or test an individual customer prediction.")

# ------------------------------------------------------------
# PAGE 2: EXPLORE DATA
# ------------------------------------------------------------
elif page == "Explore Data":
    st.markdown("<div class='section-label'>Dataset intelligence</div>", unsafe_allow_html=True)
    st.header("Explore your customer data")
    tab1, tab2, tab3 = st.tabs(["Dataset preview", "Feature analysis", "Data quality"])
    with tab1:
        st.caption(f"Showing a preview of {len(df):,} customer records from {source_name}.")
        st.dataframe(df.head(100), use_container_width=True, height=360)
        st.download_button("Download processed dataset", df.to_csv(index=False).encode("utf-8"), "churn_dataset_preview.csv", "text/csv")
    with tab2:
        st.subheader("Categorical feature analysis")
        if categorical_cols:
            selected_cat = st.selectbox("Choose a categorical feature", categorical_cols)
            cat_chart = pd.crosstab(df[selected_cat].fillna("Missing"), df["Churn"], normalize="index") * 100
            st.bar_chart(cat_chart, color=["#527fbd", "#e58b63"])
            st.caption("Bars show the percentage of churn and non-churn customers within each category.")
        else:
            st.info("No categorical features were detected in this dataset.")
        st.markdown("---")
        st.subheader("Numerical feature analysis")
        if numerical_cols:
            selected_num = st.selectbox("Choose a numerical feature", numerical_cols)
            chart_df = df[[selected_num, "Churn"]].copy()
            chart_df[selected_num] = pd.to_numeric(chart_df[selected_num], errors="coerce")
            chart_df = chart_df.dropna()
            if not chart_df.empty:
                if chart_df[selected_num].nunique() <= 10:
                    chart_df["Range"] = chart_df[selected_num].astype(str)
                else:
                    chart_df["Range"] = pd.qcut(chart_df[selected_num], q=5, duplicates="drop").astype(str)
                num_chart = pd.crosstab(chart_df["Range"], chart_df["Churn"], normalize="index") * 100
                st.bar_chart(num_chart, color=["#527fbd", "#e58b63"])
        else:
            st.info("No numerical features were detected in this dataset.")
    with tab3:
        st.subheader("Missing values by column")
        missing = df[feature_cols].isna().sum().sort_values(ascending=False)
        missing_df = missing[missing > 0].rename("Missing values").to_frame()
        if missing_df.empty:
            st.success("No missing values were found in the selected feature columns.")
        else:
            st.dataframe(missing_df, use_container_width=True)
            st.bar_chart(missing_df)
        st.write("**Detected numerical features**", numerical_cols)
        st.write("**Detected categorical features**", categorical_cols)
        if id_cols:
            st.write("**Excluded identifier columns**", id_cols)

# ------------------------------------------------------------
# PAGE 3: MODEL & EVALUATION
# ------------------------------------------------------------
elif page == "Model & Evaluation":
    st.markdown("<div class='section-label'>Machine learning workflow</div>", unsafe_allow_html=True)
    st.header("Model training & evaluation")
    st.markdown("The application uses an 80:20 stratified train-test split and a Logistic Regression pipeline.")
    steps = st.columns(4)
    for box, number, title, detail in zip(steps, ["01", "02", "03", "04"], ["Prepare", "Split", "Train", "Evaluate"], ["Impute and transform features", "80% train · 20% test", "Logistic Regression", "Measure test predictions"]):
        with box:
            st.markdown(f"<div class='result-box'><div class='hero-kicker'>{number}</div><h3 style='font-size:18px;margin:8px 0'>{title}</h3><div class='small-note'>{detail}</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Dataset split")
    a, b, c = st.columns(3)
    a.metric("Training records", f"{len(X_train):,}")
    b.metric("Testing records", f"{len(X_test):,}")
    c.metric("Input features", len(feature_cols))
    st.subheader("Performance metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{accuracy*100:.2f}%")
    k2.metric("Precision", f"{precision*100:.2f}%")
    k3.metric("Recall", f"{recall*100:.2f}%")
    k4.metric("F1-score", f"{f1*100:.2f}%")
    st.caption("Metrics are calculated on the held-out test set. Results depend on the selected dataset and target label.")
    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["Actual: No churn", "Actual: Churn"], columns=["Predicted: No churn", "Predicted: Churn"])
        st.dataframe(cm_df, use_container_width=True)
    with right:
        st.subheader("Classification report")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
    with st.expander("View training data sample"):
        st.dataframe(X_train.head(10), use_container_width=True)

# ------------------------------------------------------------
# PAGE 4: CUSTOMER PREDICTION
# ------------------------------------------------------------
else:
    st.markdown("<div class='section-label'>Individual customer assessment</div>", unsafe_allow_html=True)
    st.header("Predict customer churn")
    st.write("Enter customer details below. The form is generated automatically from the selected dataset's features.")
    with st.form("churn_prediction_form", border=True):
        input_data = {}
        for start in range(0, len(feature_cols), 2):
            cols = st.columns(2)
            for offset, col in enumerate(feature_cols[start:start+2]):
                with cols[offset]:
                    if col in numerical_cols:
                        default_value = pd.to_numeric(df[col], errors="coerce").median()
                        if pd.isna(default_value):
                            default_value = 0.0
                        input_data[col] = st.number_input(col.replace("_", " "), value=float(default_value))
                    elif col in categorical_cols:
                        options = df[col].dropna().unique().tolist()
                        if not options:
                            options = ["Unknown"]
                        input_data[col] = st.selectbox(col.replace("_", " "), options=options)
        predict_button = st.form_submit_button("✦  Predict churn probability", use_container_width=True)

    if predict_button:
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        st.markdown("---")
        st.subheader("Prediction result")
        if prediction == 1:
            st.markdown(f"<div class='result-box'><div class='hero-kicker'>CUSTOMER STATUS</div><h2 style='color:#ffb08e !important;margin:8px 0'>⚠ Likely to churn</h2><p class='hero-copy'>The model identifies this customer as being at risk of leaving.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box'><div class='hero-kicker'>CUSTOMER STATUS</div><h2 style='color:#8de0bd !important;margin:8px 0'>✓ Likely to stay</h2><p class='hero-copy'>The model identifies this customer as more likely to remain.</p></div>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Churn probability", f"{probability*100:.2f}%")
        with r2:
            st.write("**Estimated churn likelihood**")
            st.progress(probability)
            st.caption("This is a model estimate, not a guarantee of future customer behaviour.")
        with st.expander("Review entered customer details"):
            st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#71839e;font-size:11px'>CHURNIQ · TELECOM CUSTOMER CHURN ANALYSIS · BUILT WITH PYTHON, STREAMLIT & SCIKIT-LEARN</div>", unsafe_allow_html=True)
