
import streamlit as st
import pandas as pd

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

# Dataset Validation
required_columns = [
    "customerID", "gender", "SeniorCitizen", "Partner",
    "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error("The uploaded dataset is not compatible with this application.")
    st.write("Missing required columns:", missing_columns)
    st.stop()
# Display dataset
st.header("Customer Dataset")
st.dataframe(df, use_container_width=True)

# Interactive Contract Type Filter
st.sidebar.header("Dashboard Filters")

selected_contract = st.sidebar.multiselect(
    "Select Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

filtered_df = df[df["Contract"].isin(selected_contract)]
# Filtered Churn Rate KPI
filtered_churn_rate = (filtered_df["Churn"] == "Yes").mean() * 100

st.sidebar.metric(
    "Filtered Churn Rate",
    f"{filtered_churn_rate:.2f}%"
)

# Dataset overview
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Total Features", len(df.columns))

with col3:
    st.metric("Churned Customers", (df["Churn"] == "Yes").sum())

    # Customer Churn Rate
churn_rate = (df["Churn"] == "Yes").mean() * 100

st.metric(
    "Customer Churn Rate",
    f"{churn_rate:.2f}%"
)

# Churn distribution
st.header("Customer Churn Distribution")

churn_counts = df["Churn"].value_counts()

st.bar_chart(churn_counts)

# Display dataset information
st.header("Data Information")

st.write("Missing Values:")
st.write(df.isnull().sum())

st.success("Customer churn dataset loaded successfully!")

# Churn Analysis by Contract Type
st.header("Churn Analysis by Contract Type")

contract_churn = pd.crosstab(
    filtered_df["Contract"],
    filtered_df["Churn"],
    normalize="index"
) * 100
st.bar_chart(contract_churn)

# Churn Analysis by Tenure
st.header("Churn Analysis by Customer Tenure")

tenure_churn = filtered_df.groupby("tenure")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.line_chart(tenure_churn)

st.caption(
    "This chart shows the percentage of customers who churn "
    "at different tenure durations."
)
# Churn Analysis by Internet Service
st.header("Churn Analysis by Internet Service")

internet_churn = filtered_df.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(internet_churn)

st.caption(
    "This chart compares the churn percentage of customers "
    "using DSL, Fiber Optic, and No Internet Service."
)

# Churn Analysis by Payment Method
st.header("Churn Analysis by Payment Method")

payment_churn = filtered_df.groupby("PaymentMethod")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(payment_churn)

st.caption(
    "This chart compares customer churn percentages across "
    "different payment methods."
)

# Churn Analysis by Monthly Charges
st.header("Churn Analysis by Monthly Charges")

# Divide monthly charges into 10 ranges
charge_bins = pd.cut(filtered_df["MonthlyCharges"], bins=10)

monthly_charges_churn = filtered_df.groupby(
    charge_bins,
    observed=False
)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)

# Convert interval labels into strings
monthly_charges_churn.index = monthly_charges_churn.index.astype(str)

# Display chart
st.bar_chart(monthly_charges_churn)

st.caption(
    "This chart shows churn percentages across different "
    "monthly charge ranges."
)
# Data Preprocessing
st.header("Data Preprocessing")

# Create a copy of the dataset
processed_df = df.copy()

# Remove customerID as it is not useful for prediction
processed_df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
processed_df["TotalCharges"] = pd.to_numeric(
    processed_df["TotalCharges"],
    errors="coerce"
)

# Handle missing values
processed_df["TotalCharges"].fillna(
    processed_df["TotalCharges"].median(),
    inplace=True
)

# Convert categorical variables into numerical values
processed_df = pd.get_dummies(
    processed_df,
    drop_first=True,
    dtype=int
)
# Handle any remaining missing values
processed_df = processed_df.fillna(processed_df.median())


# Display processed dataset
st.subheader("Processed Dataset")
st.dataframe(processed_df, use_container_width=True)

# Display processed dataset dimensions
col1, col2 = st.columns(2)

with col1:
    st.metric("Processed Rows", processed_df.shape[0])

with col2:
    st.metric("Processed Features", processed_df.shape[1])

st.success("Data preprocessing completed successfully!")

# Train-Test Split
st.header("Train-Test Split")

from sklearn.model_selection import train_test_split

# Separate input features and target variable
X = processed_df.drop("Churn_Yes", axis=1)
y = processed_df["Churn_Yes"]

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

# Initialize the model
model = LogisticRegression(max_iter=1000)

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

# Interactive Customer Churn Prediction
st.header("Interactive Customer Churn Prediction")

st.write("Enter customer details to predict the likelihood of churn.")

# Create input form
with st.form("churn_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=840.0
        )

    predict_button = st.form_submit_button("Predict Churn")

    # Process customer input and predict churn
if predict_button:

    input_data = {
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
        "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
        "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
        "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0
    }

    input_df = pd.DataFrame([input_data])

    # Align input columns with model training columns
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.metric("Churn Probability", f"{probability * 100:.2f}%")

    st.progress(float(probability))
