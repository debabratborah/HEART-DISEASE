import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Data Preprocessing
x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

# Model Training
model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)

# Define a function to collect user input features
def user_input_features():
    age = st.number_input("Age", value=54, min_value=29, max_value=77)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=94, max_value=200, value=131)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=126, max_value=564, value=246)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=71, max_value=202, value=149)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=6.2, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Display the input form and make predictions
def main():
    st.title("Heart Disease Prediction")

    # Collect user input
    input_df = user_input_features()

    # Display user input
    st.subheader("User Input Features")
    st.write(input_df)

    # Make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display predictions
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("Presence of Heart Disease")
    else:
        st.write("No Heart Disease")

    st.subheader("Prediction Probability")
    st.write(prediction_proba)

if __name__ == "__main__":
    main()

 
 
