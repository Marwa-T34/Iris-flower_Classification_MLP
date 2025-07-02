import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib


st.set_page_config(page_title="🌼 Iris Flower Classifier", layout="centered")


model = load_model("iris_model.keras")
scaler = joblib.load("scaler.save")
class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


st.title("🌼 Iris Flower Classifier")
st.write("Enter flower measurements below to predict its species.")


with st.form("input_form"):
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")
    submitted = st.form_submit_button("Predict")

if submitted:
    input_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.success(f"🌸 Predicted: **{predicted_class}** ({confidence:.2f}% confidence)")
    st.write(" Prediction Vector:", prediction)
