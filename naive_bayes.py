import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mysql.connector
import time
import os
from PIL import Image
from rembg import remove

# Fungsi untuk menyambung dan memasukkan data ke database
def insert_into_database(kelas_pred):
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'tomatin'
    }
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = "INSERT INTO tomat (id_kategori) VALUES (%s)"
        values = (kelas_pred,)
        cursor.execute(query, values)
        conn.commit()
        print("Data berhasil dimasukkan ke database!")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Fungsi untuk memeriksa keberadaan file
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File tidak ditemukan di: {file_path}")
        exit()

# Fungsi untuk membaca dataset dari Excel
def read_dataset(file_path):
    dataset = pd.read_excel(file_path)
    
    # Pastikan hanya mengambil kolom numerik untuk fitur
    fitur = dataset.iloc[:, 1:4].apply(pd.to_numeric, errors='coerce').values  # Kolom fitur warna (R, G, B)
    kelas = dataset.iloc[:, 4].values  # Kolom kelas kematangan

    # Menghapus nilai NaN yang mungkin muncul akibat konversi
    fitur = np.nan_to_num(fitur)
    
    return fitur, kelas

# Fungsi untuk menghapus latar belakang gambar dan menyimpannya
def remove_background(file_image, output_path):
    input_image = Image.open(file_image)
    output_image = remove(input_image)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_image.save(output_path)
    print(f"Gambar dengan latar belakang dihapus disimpan di {output_path}")
    return np.array(output_image)

# Fungsi untuk menghitung nilai RGB pada piksel tengah
def get_center_pixel_color(rgb_image):
    tinggi, lebar, _ = rgb_image.shape
    return rgb_image[tinggi // 2, lebar // 2]

# Fungsi untuk melakukan normalisasi dan imputasi fitur
def preprocess_features(fitur):
    scaler = StandardScaler()
    fitur = scaler.fit_transform(fitur)

    imputer = SimpleImputer(strategy='mean')
    fitur = imputer.fit_transform(fitur)
    return fitur, scaler, imputer

# Fungsi untuk inisialisasi dan pelatihan classifier
def train_classifier(fitur, kelas):
    classifier = GaussianNB()
    classifier.fit(fitur, kelas)
    return classifier

# Fungsi untuk prediksi dan memasukkan hasil ke database
def predict_and_store(classifier, tes_fitur):
    kelas_pred = classifier.predict(tes_fitur)
    print("Kelas Prediksi:", kelas_pred[0])  # Pastikan ini berisi nilai yang benar
    print("Tipe Data:", type(kelas_pred[0])) # Debugging tipe data
    insert_into_database(str(kelas_pred[0])) # Pastikan dikonversi ke string
    return kelas_pred[0]


# Main program
def main():
    # Path penyimpanan
    path_tomat = 'D:/New folder/pcv_sistem_cerdas/Tomat.In_PCV_SistemCerdas/TOMAT'
    file_excel = 'D:/New folder/pcv_sistem_cerdas/Tomat.In_PCV_SistemCerdas/dataset_tomat.in.xlsx'
    file_image = 'tomat60.jpg'
    output_path = 'D:/New folder/pcv_sistem_cerdas/Tomat.In_PCV_SistemCerdas/output_tomat.png'

    # Memastikan file Excel dan gambar ada
    file_path_excel = os.path.join(path_tomat, file_excel)
    check_file_exists(file_path_excel)

    file_name = os.path.join(path_tomat, file_image)
    check_file_exists(file_name)

    # Membaca dataset dan gambar
    fitur, kelas = read_dataset(file_path_excel)
    output_image_np = remove_background(file_name, output_path)

    # Konversi gambar ke RGB dan ambil piksel tengah
    rgb_image = cv2.cvtColor(output_image_np, cv2.COLOR_RGBA2RGB)
    piksel_tengah = get_center_pixel_color(rgb_image)

    # Normalisasi dan imputasi data pelatihan
    fitur, scaler, imputer = preprocess_features(fitur)

    # Membagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(fitur, kelas, test_size=0.3, random_state=42)

    # Inisialisasi dan latih classifier
    classifier = train_classifier(X_train, y_train)

    # Memprediksi pada data uji dan menghitung akurasi
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model: {accuracy * 100:.2f}%")

    # Imputasi dan normalisasi tes_fitur
    tes_fitur = imputer.transform([piksel_tengah])
    tes_fitur = scaler.transform(tes_fitur)

    # Loop untuk prediksi kelas kematangan
    while True:
        predict_and_store(classifier, tes_fitur)
        time.sleep(10)

if __name__ == "__main__":
    main()
