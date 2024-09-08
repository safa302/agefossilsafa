import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model (Random Forest or Gradient Boosting)
rf_pickle_path = 'random_forest_model.pkl'
gb_pickle_path = 'gradient_boosting_model.pkl'

with open(rf_pickle_path, 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open(gb_pickle_path, 'rb') as gb_file:
    gb_model = pickle.load(gb_file)

# Function to make predictions
def make_prediction(model, input_data):
    # Convert input data to DataFrame for prediction
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app UI
st.title("Aplikasi Prediksi Usia Fosil")

# User input form
st.header("Masukkan Data untuk Prediksi Usia Fosil")
input_data = {}

# Fitur-fitur yang ada dalam dataset
input_data['Fossil_Length'] = st.number_input("Masukkan Panjang Fosil (cm)", min_value=0.0)
input_data['Fossil_Width'] = st.number_input("Masukkan Lebar Fosil (cm)", min_value=0.0)
input_data['Fossil_Height'] = st.number_input("Masukkan Tinggi Fosil (cm)", min_value=0.0)
input_data['Fossil_Weight'] = st.number_input("Masukkan Berat Fosil (kg)", min_value=0.0)
input_data['Location_Latitude'] = st.number_input("Masukkan Latitude Lokasi Fosil", min_value=-90.0, max_value=90.0)
input_data['Location_Longitude'] = st.number_input("Masukkan Longitude Lokasi Fosil", min_value=-180.0, max_value=180.0)
input_data['Soil_Type'] = st.selectbox("Jenis Tanah", ['Lempung', 'Pasir', 'Gersang', 'Subur'])

# Model selection
model_option = st.selectbox("Pilih Model:", ("Random Forest", "Gradient Boosting"))

# Make prediction button
if st.button("Prediksi"):
    if model_option == "Random Forest":
        prediction = make_prediction(rf_model, input_data)
    else:
        prediction = make_prediction(gb_model, input_data)
    
    st.success(f"Hasil Prediksi Usia Fosil: {prediction} tahun")
