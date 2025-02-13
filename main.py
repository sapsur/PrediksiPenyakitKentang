import streamlit as st
import pickle
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

# Load model dan label encoder
model_path = "E:\\Kuliyeah\\Citra\\model_knn.pkl"
with open(model_path, "rb") as model_file:
    clf, label_encoder = pickle.load(model_file)

# Parameter LBP
radius = 3
n_points = 8 * radius

st.title("Prediksi Penyakit Kentang dengan KNN")
st.write("Unggah gambar kentang untuk memprediksi penyakitnya.")

# Upload file gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Tampilkan gambar yang diunggah
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Konversi ke format RGB dan grayscale
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ekstraksi fitur warna (HSV)
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv_image, axis=(0, 1))
    h_std, s_std, v_std = np.std(hsv_image, axis=(0, 1))

    # Ekstraksi fitur tekstur menggunakan LBP
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)

    # Gabungkan semua fitur menjadi satu array
    feature_vector = [h_mean, s_mean, v_mean, h_std, s_std, v_std] + list(lbp_hist)
    input_array = np.array(feature_vector).reshape(1, -1)

    # Prediksi menggunakan model KNN
    prediction = clf.predict(input_array)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # Tampilkan hasil prediksi
    st.success(f"**Hasil Prediksi: {predicted_label}**")
