
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    layout="wide"
)

# Application title
st.title("Telecom Customer Churn Prediction")
st.subheader("Data Pipeline for Customer Churn Analysis")

# Load dataset
# Load Dataset
st.sidebar.header("Dataset Selection")

uploaded_file = st.sidebar.file_uploader(
    "Upload a Telecom Churn CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded!")
else:
    df = pd.read_csv("Telco-Customer-Churn.csv")
    st.sidebar.info("Using default IBM dataset")

# Churn Target Selection
st.sidebar.header("Churn Target Configuration")

target_col = st.sidebar.selectbox(
    "Select the Churn Target Column",
    options=df.columns
)

target_values = df[target_col].dropna().unique().tolist()

if len(target_values) != 2:
    st.error("Please select a column containing exactly two churn classes.")
    st.stop()

positive_label = st.sidebar.selectbox(
    "Select the value representing Churn",
    options=target_values
)

# Standardize target values
df = df.dropna(subset=[target_col]).copy()

df["Churn"] = df[target_col].apply(
    lambda value: "Yes" if value == positive_label else "No"
)

st.sidebar.success("Churn target configured successfully!")
# Display dataset
st.header("Customer Dataset")
st.dataframe(df, use_container_width=True)

# Dynamic Dashboard

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Total Features", len(df.columns))

with col3:
    st.metric("Churned Customers", (df["Churn"] == "Yes").sum())

churn_rate = (df["Churn"] == "Yes").mean() * 100

st.metric("Customer Churn Rate", f"{churn_rate:.2f}%")

# Dataset Preview
st.header("Customer Dataset")
st.dataframe(df, use_container_width=True)

# Churn Distribution
st.header("Customer Churn Distribution")

st.bar_chart(df["Churn"].value_counts())

# Missing Values
st.header("Data Information")
st.write("Missing Values:")
st.write(df.isnull().sum())

# Identify available features
feature_cols = [
    col for col in df.columns
    if col not in [target_col, "Churn"]
]

# Convert numeric-looking columns
for col in feature_cols:
    converted = pd.to_numeric(df[col], errors="coerce")

    if converted.notna().mean() >= 0.8:
        df[col] = converted

# Exclude identifier columns
id_cols = [
    col for col in feature_cols
    if col.lower().replace("_", "").replace(" ", "")
    in ["id", "customerid", "subscriberid", "accountid"]
]

feature_cols = [
    col for col in feature_cols if col not in id_cols
]

categorical_cols = df[feature_cols].select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_cols = df[feature_cols].select_dtypes(
    include=["number"]
).columns.tolist()

# Categorical Feature Analysis
st.header("Categorical Feature Analysis")

for col in categorical_cols[:5]:
    st.subheader(f"Churn by {col}")

    chart_data = pd.crosstab(
        df[col].fillna("Missing"),
        df["Churn"],
        normalize="index"
    ) * 100

    st.bar_chart(chart_data)

# Numerical Feature Analysis
st.header("Numerical Feature Analysis")

for col in numerical_cols[:5]:
    st.subheader(f"Churn by {col}")

    chart_df = df[[col, "Churn"]].copy()
    chart_df[col] = pd.to_numeric(
        chart_df[col], errors="coerce"
    )

    chart_df = chart_df.dropna()

    if chart_df[col].nunique() > 1:
        chart_df["Range"] = pd.qcut(
            chart_df[col],
            q=5,
            duplicates="drop"
        ).astype(str)

        churn_analysis = pd.crosstab(
            chart_df["Range"],
            chart_df["Churn"],
            normalize="index"
        ) * 100

        st.bar_chart(churn_analysis)

st.success("Dataset analysis completed successfully!")


# Dynamic Data Preprocessing

st.header("Data Preprocessing")

# Separate features and target
X = df[feature_cols]
y = (df["Churn"] == "Yes").astype(int)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

st.write("Numerical Features:", numerical_cols)
st.write("Categorical Features:", categorical_cols)

st.success("Dynamic preprocessing configuration completed!")

# Train-Test Split
st.header("Train-Test Split")

from sklearn.model_selection import train_test_split

# Separate input features and target variable
X = df[feature_cols]
y = (df["Churn"] == "Yes").astype(int)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display split information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Samples", X_train.shape[0])

with col2:
    st.metric("Testing Samples", X_test.shape[0])

with col3:
    st.metric("Input Features", X_train.shape[1])

st.write("Training Data Preview")
st.dataframe(X_train.head(), use_container_width=True)

st.success("Train-test split completed successfully!")

# Machine Learning Model Training
st.header("Machine Learning Model Training")

from sklearn.linear_model import LogisticRegression

# Initialize the complete machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

st.success("Logistic Regression model trained successfully!")

st.write("The model has learned patterns from the training dataset.")

# Model Evaluation
st.header("Model Evaluation")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

# Classification Report
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No Churn", "Actual Churn"],
    columns=["Predicted No Churn", "Predicted Churn"]
)

st.dataframe(cm_df, use_container_width=True)

st.success("Model evaluation completed successfully!")

# Dynamic Customer Churn Prediction

st.header("Interactive Customer Churn Prediction")

st.write("Enter customer details to predict the likelihood of churn.")

# Generate input fields dynamically
with st.form("churn_prediction_form"):

    input_data = {}
    columns = st.columns(2)

    for index, col in enumerate(feature_cols):

        with columns[index % 2]:

            if col in numerical_cols:
                default_value = pd.to_numeric(
                    df[col], errors="coerce"
                ).median()

                if pd.isna(default_value):
                    default_value = 0.0

                input_data[col] = st.number_input(
                    col,
                    value=float(default_value)
                )

            elif col in categorical_cols:
                options = df[col].dropna().unique().tolist()

                if not options:
                    options = ["Unknown"]

                input_data[col] = st.selectbox(
                    col,
                    options=options
                )

    predict_button = st.form_submit_button("Predict Churn")

# Make prediction
if predict_button:

    input_df = pd.DataFrame([input_data], columns=feature_cols)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.metric("Churn Probability", f"{probability * 100:.2f}%")

    st.progress(float(probability))
