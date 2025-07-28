
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan data
lr_clf = joblib.load('model.pkl')
X = joblib.load('X_columns.pkl')

def predict_price(name, km_driven, transmission, car_age, seller_type, owner):
    x = np.zeros(len(X))
    x[0] = km_driven
    x[1] = transmission
    x[2] = car_age
    x[3] = seller_type
    x[4] = owner
    if name in X:
        loc_index = X.index(name)
        x[loc_index] = 1
    return lr_clf.predict([x])[0]

st.title("Prediksi Harga Mobil Bekas")

name = st.selectbox("Merk Mobil", [col for col in X if col.startswith('name_')])
km_driven = st.number_input("Jarak Tempuh (km)", value=50000)
transmission = st.selectbox("Transmisi", ['Manual', 'Automatic'])
transmission = 0 if transmission == 'Manual' else 1
car_age = st.slider("Umur Mobil (tahun)", 0, 30, 5)
seller_type = st.selectbox("Tipe Penjual", ['Dealer', 'Individual', 'Trustmark Dealer'])
seller_type = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}[seller_type]
owner = st.selectbox("Jumlah Pemilik Sebelumnya", ['First', 'Second', 'Third', 'Fourth & Above', 'Test Drive Car'])
owner_map = {'First': 0, 'Second': 1, 'Third': 2, 'Fourth & Above': 3, 'Test Drive Car': 4}
owner = owner_map[owner]

if st.button("Prediksi Harga"):
    result = predict_price(name, km_driven, transmission, car_age, seller_type, owner)
    st.success(f"Estimasi Harga Mobil Bekas: ₹ {round(result, 2)}")
