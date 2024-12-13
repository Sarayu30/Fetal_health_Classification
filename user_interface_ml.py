import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Function to load the dataset (fixed path)
def load_dataset():
    file_path = r"C:\Users\sarayu krishna\Downloads\CTG.csv"  # Fixed dataset path
    df = pd.read_csv(file_path)
    df = df.drop(["FileName", "Date", "b", "e"], axis=1)  # Drop non-essential columns
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# Function for data preprocessing and training the model
def preprocess_and_train(df):
    df_copy = df.copy()
    X = df_copy[['CLASS', 'SUSP', 'FS', 'LD']]  # Use only important features
    y = df_copy['NSP']
    
    # Oversample using SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    # Train the Random Forest model
    rf_clf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=20, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    
    return rf_clf, scaler

# Function to make predictions based on user input
def make_prediction(class_value, susp_value, fs_value, ld_value):
    # Check if the model has been trained
    if 'rf_clf' not in st.session_state or 'scaler' not in st.session_state:
        st.error("Please train the model first.")
        return

    rf_clf = st.session_state.rf_clf
    scaler = st.session_state.scaler

    try:
        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[class_value, susp_value, fs_value, ld_value]], columns=['CLASS', 'SUSP', 'FS', 'LD'])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = rf_clf.predict(input_data_scaled)[0]

        # Display result
        if prediction == 1:
            result = "Normal"
        elif prediction == 2:
            result = "Suspect"
        else:
            result = "Pathologic"

        st.success(f"The predicted NSP is: {result}")
    
    except ValueError:
        st.error("Please enter valid numerical values.")

# Streamlit Interface

st.set_page_config(page_title="CTG Examination Prediction", layout="wide")
st.title("CTG Examination Prediction")

# Add light pink background and health theme styling
st.markdown("""
    <style>
    body {
        background-color: #FFEBEE; /* Light Pink background */
        color: #880E4F; /* Dark pink for text */
    }
    .stButton > button {
        background-color: #D81B60; /* Pink button color */
        color: white;
        padding: 15px 32px;
        text-align: center;
        font-size: 16px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #C2185B; /* Darker pink on hover */
    }
    h1, h2, h3 {
        color: #C2185B; /* Header color */
    }
    .stTextInput input, .stNumberInput input {
        background-color: #F8BBD0; /* Pinkish input fields */
        border: 2px solid #D81B60; /* Pink border */
        color: #880E4F; /* Text color */
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #C2185B; /* Focused input border color */
    }
    .stAlert {
        background-color: #F8BBD0; /* Light pink alert boxes */
        color: #880E4F; /* Dark pink text in alert boxes */
    }
    </style>
""", unsafe_allow_html=True)

# Instructions
st.markdown("""
    ## How it works:
    1. Train the model by clicking the **Train Model** button.
    2. After training, enter the values for **CLASS**, **SUSP**, **FS**, and **LD**.
    3. Click the **Make Prediction** button to get the result.
    4. The model will predict whether the examination result is **Normal**, **Suspect**, or **Pathologic**.
""")

# Train the model
if st.button("Train Model"):
    df = load_dataset()  # Load the dataset
    rf_clf, scaler = preprocess_and_train(df)
    
    # Save the trained model and scaler in the session state
    st.session_state.rf_clf = rf_clf
    st.session_state.scaler = scaler
    
    st.success("Model trained successfully! Ready to make predictions.")

# User inputs for prediction
class_value = st.number_input("Enter CLASS value:", min_value=0.0, max_value=100.0, step=0.1)
susp_value = st.number_input("Enter SUSP value:", min_value=0.0, max_value=100.0, step=0.1)
fs_value = st.number_input("Enter FS value:", min_value=0.0, max_value=100.0, step=0.1)
ld_value = st.number_input("Enter LD value:", min_value=0.0, max_value=100.0, step=0.1)

# Prediction button
if st.button("Make Prediction"):
    make_prediction(class_value, susp_value, fs_value, ld_value)

# Footer
st.markdown("Made with ❤️ using Streamlit")
