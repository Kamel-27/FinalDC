import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set Streamlit page config
st.set_page_config(page_title="Diabetes Health Analysis", layout="wide")
st.title("🩺 Diabetes Health Indicators Analysis")

# GitHub dataset URL
github_url = "https://github.com/Kamel-27/FinalDC/blob/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

# Load models
try:
    lda = joblib.load("lda.pkl")
    scaler = joblib.load("scaler.pkl")
    svm = joblib.load("svm_model.pkl")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# Top features used in the model
top_features = ['HighBP', 'HighChol', 'BMI', 'HeartDiseaseorAttack', 'GenHlth',
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Income']

# Main function
def main():
    try:
        df = pd.read_csv(diabetes_binary_5050split_health_indicators_BRFSS2015.csv)
        data = df.copy()

        # Convert types
        cols_to_convert = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
                           "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                           "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth",
                           "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
        for col in cols_to_convert:
            if col in data.columns:
                data[col] = data[col].astype(int)

        # Sidebar navigation
        page = st.sidebar.radio("Select Section", [
            "📊 Dataset Overview", 
            "📉 LDA + SVM Evaluation", 
            "🧪 Predict from Input", 
            "📤 Upload & Predict CSV"
        ])

        if page == "📊 Dataset Overview":
            dataset_overview(data)

        elif page == "📉 LDA + SVM Evaluation":
            model_evaluation(data)

        elif page == "🧪 Predict from Input":
            input_prediction()

        elif page == "📤 Upload & Predict CSV":
            upload_and_predict()

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")


# 📊 Dataset overview
def dataset_overview(data):
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())
    st.write("**Shape:**", data.shape)
    st.write("**Missing Values:**")
    st.write(data.isnull().sum())
    st.write("**Unique Values Per Column:**")
    st.write(data.nunique())

    st.subheader("🧪 Target Variable Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Diabetes_binary', data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(20,10))
    sns.heatmap(data.corr(), cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)


# 📉 Evaluate model with LDA projection
def model_evaluation(data):
    st.subheader("📉 LDA + SVM Evaluation")
    X = data[top_features]
    Y = data['Diabetes_binary']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # LDA transform, scale, and predict
    X_test_lda = lda.transform(X_test)
    X_test_scaled = scaler.transform(X_test_lda)
    Y_pred = svm.predict(X_test_scaled)

    st.write("**✅ Accuracy:**", accuracy_score(Y_test, Y_pred))
    st.text("Classification Report:\n" + classification_report(Y_test, Y_pred))
    st.text("Confusion Matrix:\n" + str(confusion_matrix(Y_test, Y_pred)))

    st.subheader("🧬 LDA Projection of Test Data")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=Y_test, cmap='coolwarm', alpha=0.6)
    legend = ax.legend(*scatter.legend_elements(), title="Diabetes")
    ax.add_artist(legend)
    ax.set_xlabel("LDA Component 1")
    ax.set_ylabel("LDA Component 2")
    st.pyplot(fig)


# 🧪 Predict single input
def input_prediction():
    st.subheader("🧪 Predict Diabetes from User Input")
    with st.form("prediction_form"):
        HighBP = st.selectbox("HighBP", [0, 1])
        HighChol = st.selectbox("HighChol", [0, 1])
        BMI = st.number_input("BMI", value=25.0)
        HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack", [0, 1])
        GenHlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3)
        MentHlth = st.slider("Mental Health (Days)", 0, 30, 5)
        PhysHlth = st.slider("Physical Health (Days)", 0, 30, 5)
        DiffWalk = st.selectbox("Difficulty Walking", [0, 1])
        Age = st.slider("Age Category (0=18-24, 13=80+)", 0, 13, 9)
        Income = st.slider("Income Category (1=<$10K, 8=$75K+)", 1, 8, 4)

        submit = st.form_submit_button("Predict")

        if submit:
            new_record = pd.DataFrame([{
                'HighBP': HighBP, 'HighChol': HighChol, 'BMI': BMI,
                'HeartDiseaseorAttack': HeartDiseaseorAttack, 'GenHlth': GenHlth,
                'MentHlth': MentHlth, 'PhysHlth': PhysHlth, 'DiffWalk': DiffWalk,
                'Age': Age, 'Income': Income
            }])
            input_lda = lda.transform(new_record)
            input_scaled = scaler.transform(input_lda)
            prediction = svm.predict(input_scaled)
            if prediction[0] == 0:
                st.success("✅ Result: The patient is **not diabetic**.")
            else:
                st.warning("⚠️ Result: The patient is **pre-diabetic or diabetic**.")


# 📤 Upload data for batch prediction
def upload_and_predict():
    st.subheader("📤 Upload a Dataset for Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.write("✅ Uploaded Data Preview:")
            st.dataframe(uploaded_df.head())

            missing_cols = [col for col in top_features if col not in uploaded_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Make predictions
                uploaded_X = uploaded_df[top_features]
                uploaded_lda = lda.transform(uploaded_X)
                uploaded_scaled = scaler.transform(uploaded_lda)
                uploaded_preds = svm.predict(uploaded_scaled)

                uploaded_df['Predicted_Diabetes'] = uploaded_preds
                st.write("✅ Prediction Completed. Sample Output:")
                st.dataframe(uploaded_df.head())

                # Download link
                csv = uploaded_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name='predicted_diabetes.csv',
                    mime='text/csv'
                )
        except Exception as ex:
            st.error(f"❌ Error processing uploaded file: {ex}")


# Run the app
if __name__ == "__main__":
    main()
