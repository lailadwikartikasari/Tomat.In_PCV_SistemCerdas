import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mysql.connector
import time
import os
from PIL import Image
from rembg import remove
import random

# Fungsi untuk menghubungkan ke database dan memasukkan data
def masukkan_ke_database(kelas_pred):
    konfigurasi_db = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'tomatin'
    }
    try:
        conn = mysql.connector.connect(**konfigurasi_db)
        cursor = conn.cursor()

        # Periksa apakah kelas_pred ada dalam tabel kategori
        cursor.execute("SELECT id_kategori FROM kategori WHERE id_kategori = %s", (kelas_pred,))
        if cursor.fetchone() is None:
            print(f"Error: id_kategori {kelas_pred} tidak ditemukan dalam tabel kategori.")
            return  # Keluar dari fungsi jika kategori tidak ditemukan

        # Jika id_kategori ditemukan, lanjutkan untuk memasukkan data
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
            
# Fungsi untuk memeriksa apakah file ada
def periksa_file(file_path):
    if not os.path.exists(file_path):
        print(f"File tidak ditemukan di: {file_path}")
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")

# Fungsi untuk membaca dataset dari Excel
def baca_dataset(file_path):
    try:
        dataset = pd.read_excel(file_path)

        # Pastikan hanya mengambil kolom numerik sebagai fitur
        fitur = dataset.iloc[:, 1:4].apply(pd.to_numeric, errors='coerce').values  # Mengasumsikan kolom 1 hingga 3 adalah R, G, B
        kelas = dataset.iloc[:, 4].values  # Mengasumsikan kolom 4 adalah label kelas

        # Menghapus nilai NaN yang mungkin ada dari konversi
        fitur = np.nan_to_num(fitur)

        return fitur, kelas
    except Exception as e:
        print(f"Error membaca dataset: {e}")
        raise

# Fungsi untuk menghapus latar belakang dari gambar
def hapus_latar_belakang(file_gambar, output_path):
    input_image = Image.open(file_gambar)
    output_image = remove(input_image)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_image.save(output_path)
    print(f"Gambar dengan latar belakang dihapus disimpan di {output_path}")
    return np.array(output_image)

# Fungsi untuk mendapatkan nilai RGB dari piksel tengah
def ambil_rgb_real(rgb_image):
    tinggi, lebar, _ = rgb_image.shape
    return rgb_image[tinggi // 2, lebar // 2]

# Fungsi untuk praproses fitur
def praproses_fitur(fitur):
    scaler = StandardScaler()
    fitur = scaler.fit_transform(fitur)

    imputer = SimpleImputer(strategy='mean')
    fitur = imputer.fit_transform(fitur)
    return fitur, scaler, imputer

# Fungsi untuk prediksi dan menyimpan hasil di database
def prediksi_dan_simpan(classifier, tes_fitur):
    kelas_pred = classifier.predict(tes_fitur)
    print("Kelas Prediksi:", kelas_pred[0])
    masukkan_ke_database(str(kelas_pred[0]))
    return kelas_pred[0]

# Program utama
def main():
    # Path
    path_tomat = 'D:/New folder/pcv_sistem_cerdas/Tomat.In_PCV_SistemCerdas/TOMAT'
    file_xlsx = 'D:/New folder/pcv_sistem_cerdas/Tomat.In_PCV_SistemCerdas/dataset_tomat.in.xlsx'
    file_image = 'tomat67.jpg'
    output_path = 'D:/New folder/naive bayes/output_tomat.png'

    # Periksa apakah file ada
    periksa_file(file_xlsx)

    file_name = os.path.join(path_tomat, file_image)
    periksa_file(file_name)

    # Membaca dataset dan gambar
    fitur, kelas = baca_dataset(file_xlsx)
    output_image_np = hapus_latar_belakang(file_name, output_path)

    # Konversi gambar ke RGB dan ambil piksel tengah
    rgb_image = cv2.cvtColor(output_image_np, cv2.COLOR_RGBA2RGB)
    piksel_tengah = ambil_rgb_real(rgb_image)

    # Normalisasi dan imputasi data pelatihan
    fitur, scaler, imputer = praproses_fitur(fitur)

    # Bagi dataset menjadi data pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(fitur, kelas, test_size=0.2, random_state=42)

    # Terapkan SMOTE untuk mengatasi ketidakseimbangan kelas
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Perulangan untuk memprediksi kelas kematangan dan memperbarui model
    while True:
        # Pilih subset data pelatihan secara acak untuk setiap iterasi
        subset_indices = random.sample(range(len(X_train_resampled)), int(len(X_train_resampled) * 0.8))
        X_subset = X_train_resampled[subset_indices]
        y_subset = y_train_resampled[subset_indices]

        # Latih model Naive Bayes pada subset
        model = GaussianNB()
        model.fit(X_subset, y_subset)

        # Evaluasi performa model pada data pengujian
        y_pred = model.predict(X_test)
        akurasi = accuracy_score(y_test, y_pred)
        print("Akurasi :", akurasi)
        # print(classification_report(y_test, y_pred, zero_division=0))

        # Imputasi dan normalisasi fitur pengujian
        tes_fitur = imputer.transform([piksel_tengah])
        tes_fitur = scaler.transform(tes_fitur)

        # Prediksi dan simpan hasil
        prediksi_dan_simpan(model, tes_fitur)

        # Tunggu sebelum mengulang dengan subset yang berbeda
        time.sleep(10)

if __name__ == "__main__":
    main()
