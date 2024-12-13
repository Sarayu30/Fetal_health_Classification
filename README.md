 

# CTG Examination Prediction

This project is a web-based application built using **Streamlit** to predict the condition of CTG (Cardiotocography) examinations, categorizing results into **Normal**, **Suspect**, or **Pathologic**. The application utilizes **Random Forest Classifier** and **SMOTE** for handling imbalanced datasets.12 Classification models with XAI (LIME  and SHAP) on the best model are implemented in 'ml-models.ipynb'.

---

## Features

- **Data Preprocessing**: The app handles missing values, duplicates, and irrelevant columns in the dataset.
- **Imbalanced Data Handling**: Uses **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Machine Learning**: Implements a **Random Forest Classifier** with scaled input features.
- **User-Friendly Interface**: Interactive Streamlit interface for model training and predictions.
- **Real-Time Predictions**: Predicts examination results based on user inputs.

---

## Installation and Setup

### Prerequisites
Ensure the following are installed on your system:
- Python 3.8 or above
- Required Python packages (see below)

### Install Dependencies
Run the following command to install all necessary packages:

```bash
pip install -r requirements.txt
```

#### Create `requirements.txt` with:
```txt
streamlit
pandas
scikit-learn
imblearn
```

---

## How to Run

1. Clone the repository or download the files:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. Ensure the dataset (`CTG.csv`) is in the correct path:
   ```
   C:\Users\sarayu krishna\Downloads\CTG.csv
   ```

3. Start the Streamlit app:
   ```bash
   streamlit run user_interface_ml.py
   ```

4. Open the link provided by Streamlit in your web browser.

---

## Usage

1. **Train the Model**:
   - Click the **Train Model** button to preprocess the dataset and train the Random Forest model.
2. **Input Values**:
   - Enter the values for **CLASS**, **SUSP**, **FS**, and **LD**.
3. **Make Prediction**:
   - Click the **Make Prediction** button to get the result (Normal, Suspect, or Pathologic).

---

## File Structure

- **`user_interface_ml.py`**: Main Streamlit application.
- **`CTG.csv`**: Dataset for training and prediction.
- **`ml-visualization.ipynb`**:  Exploratory Data Analysis and visualizations.
- **`ml-models.ipynb`**: Model training and evaluation notebook containing 12 different CLASSIFICATION models

---
Feel free to contribute or raise issues for improvement!
Made with ❤️ by Sarayu Krishna



 

Feel free to contribute or raise issues for improvement!  
**Made with ❤️ by Sarayu Krishna**

---

Let me know if you need help with further customization!
