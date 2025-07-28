
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and column names
lr_clf = pickle.load(open('model.pkl', 'rb'))
X_columns = pickle.load(open('columns.pkl', 'rb')) # Load as list

st.title("Prediksi Harga Mobil Bekas")

# Get list of car names for selectbox, excluding the first few non-name columns
car_names = [col for col in X_columns if col not in ['km_driven', 'transmission', 'car_age', 'seller_type.', 'owner.']]

# Buat input form
name = st.selectbox("Merk Mobil", car_names)
km_driven = st.number_input("Jarak Tempuh (km)", value=50000)
transmission_option = st.selectbox("Transmisi", ["Manual", "Automatic"])  # Use string options
transmission = 1 if transmission_option == "Manual" else 0
car_age = st.number_input("Usia Mobil (tahun)", value=5)

# Define seller type and owner options based on your encoding
seller_type_options = {0: 'Individual', 1: 'Dealer', 2: 'Trustmark Dealer'}
owner_options = {0: 'First Owner', 1: 'Second Owner', 2: 'Fourth & Above Owner', 3: 'Third Owner', 4: 'Test Drive Car'}

seller_type_display = st.selectbox("Tipe Penjual", list(seller_type_options.values()))
seller_type = [k for k, v in seller_type_options.items() if v == seller_type_display][0]


owner_display = st.selectbox("Jumlah Pemilik Sebelumnya", list(owner_options.values()))
owner = [k for k, v in owner_options.items() if v == owner_display][0]


# Fungsi prediksi (pindahkan dari cell lama)
def predict_price(name, km_driven, transmission, car_age, seller_type, owner):
    x = np.zeros(len(X_columns))

    try:
        name_index = X_columns.index(name)
        x[name_index] = 1
    except ValueError:
        # Handle cases where the selected name is not in the original columns (e.g., 'other')
        # Based on your notebook, 'other' is handled by not setting any specific name index to 1
        pass

    # Assign values to the known non-name features based on their expected index
    # Ensure these indices match the order in your X DataFrame when it was created
    # Assuming the order is: km_driven, transmission, car_age, seller_type., owner.
    x[0] = km_driven
    x[1] = transmission
    x[2] = car_age
    x[3] = seller_type
    x[4] = owner


    return lr_clf.predict([x])[0]

# Tombol Prediksi
if st.button("Prediksi"):
    prediction = predict_price(name, km_driven, transmission, car_age, seller_type, owner)
    st.success(f"Perkiraan Harga Mobil Bekas: Rp {prediction:,.0f}")
