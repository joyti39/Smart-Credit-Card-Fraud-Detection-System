import pickle
import streamlit as st
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

# Background Image via CSS
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function
add_bg_from_local("img1.jpg")

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Smart Credit Card Fraud Detection System")
st.markdown("Upload your dataset & predict fraudulent transactions!")

uploaded_file = st.file_uploader("📂 Upload your `creditcard.csv` file", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.success(" Dataset Loaded Successfully!")

    with st.expander("Dataset Overview"):
        st.write("**Shape:**", data.shape)
        st.write("**Preview:**")
        st.write(data.head(20))

    # Split legit and fraud
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    # Under-sampling
    legit_sample = legit.sample(n=492, random_state=2)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)

    # Features & Target
    X = balanced_data.drop(columns='Class', axis=1)
    Y = balanced_data['Class']

    #Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
     X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Load model (pickle)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Predictions and accuracy
    y_train_pred = model.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, y_train_pred)
    y_test_pred = model.predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, y_test_pred)

    # Classification Report
    report_dict = classification_report(Y_test, y_test_pred, target_names=['Legit', 'Fraud'], output_dict=True)

    # Markdown report
    report_md = """
  Classification Report

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
"""
    for cls in ['Legit', 'Fraud']:
        precision = report_dict[cls]['precision']
        recall = report_dict[cls]['recall']
        f1_score = report_dict[cls]['f1-score']
        support = int(report_dict[cls]['support'])
        report_md += f"| {cls} | {precision:.2f} | {recall:.2f} | {f1_score:.2f} | {support} |\n"

    # Accuracy percentages
    training_accuracy_percent = round(training_data_accuracy * 100, 2)
    test_accuracy_percent = round(test_data_accuracy * 100, 2)

    # Show metrics
    st.markdown("## Model Evaluation")
    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{training_accuracy_percent}%")
    col2.metric("Test Accuracy", f"{test_accuracy_percent}%")

    # ROC Curve probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Plots
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Confusion Matrix")
        cm = confusion_matrix(Y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
            ax=ax,
            annot_kws={"size": 14}
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)

    with col2:
        st.write("### ROC Curve")
        fpr, tpr, _ = roc_curve(Y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(fpr, tpr, color='darkorange', label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax2.set_xlabel('False Positive Rate', fontsize=11)
        ax2.set_ylabel('True Positive Rate', fontsize=11)
        ax2.legend(loc="lower right", fontsize=10)
        ax2.tick_params(axis='both', labelsize=11)
        st.pyplot(fig2)

    with col3:
        st.write("### Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(Y_test, y_prob)
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.plot(recall, precision, marker='.')
        ax3.set_xlabel('Recall', fontsize=11)
        ax3.set_ylabel('Precision', fontsize=11)
        ax3.tick_params(axis='both', labelsize=11)
        st.pyplot(fig3)

    # Classification Report
    with st.expander("Classification Report"):
        st.markdown(report_md)

    with st.expander(" Detection Summary"):
        st.code(f"""
 Credit Card Fraud Detection Summary
- Model Used: Logistic Regression
- Accuracy Achieved: {test_accuracy_percent:.2f}%

- Techniques Applied:
  • Under-sampling for dataset balancing
  • Confusion Matrix for error visualization
  • ROC Curve & Precision-Recall Curve for classifier evaluation
        """)

    st.markdown("---")
    st.header(" Predict Fraud for a New Transaction")

    example_legit = legit[(legit['Time'] != 0) & (legit['Amount'] != 0)].iloc[0].drop('Class').to_dict()
    example_fraud = fraud[(fraud['Time'] != 0) & (fraud['Amount'] != 0)].iloc[0].drop('Class').to_dict()

    col1, col2 = st.columns(2)
    if col1.button(" Auto-fill Legit Transaction"):
        st.session_state.inputs = example_legit
    if col2.button("Auto-fill Fraud Transaction"):
        st.session_state.inputs = example_fraud

    input_data = {}
    cols = st.columns(4)
    for i, col in enumerate(X.columns):
        default = 0.0
        if "inputs" in st.session_state:
            default = float(st.session_state.inputs.get(col, 0.0))
        input_val = cols[i % 4].number_input(col, value=default, format="%.6f")
        input_data[col] = input_val

    if st.button(" Predict Transaction Type"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f" Fraud Detected with Probability: {prob*100:.2f}%")
        else:
            st.success(f"Legitimate Transaction with Probability: {(1-prob)*100:.2f}%")

else:
    st.info("Please upload the `creditcard.csv` file to begin.")
