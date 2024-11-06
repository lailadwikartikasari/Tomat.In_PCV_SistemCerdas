import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.impute import SimpleImputer
import mysql.connector
import time
import os

def insert_into_database(kelas_pred):
    # Ganti dengan informasi koneksi ke database MySQL Anda
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'tomatin'
    }

    try:
        # Membuat koneksi ke database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Menyisipkan nilai kelas_pred ke dalam tabel tertentu
        query = "INSERT INTO kematangan_tomat (kematangan) VALUES (%s)"
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

# Path penyimpanan
path_tomat = 'D:/New folder/pcv_sistem_cerdas/TM_BARU'
file_excel = 'D:/New folder/pcv_sistem_cerdas/hasil_Matang.xlsx'  # Pastikan file ini ada di path yang benar
file_path_excel = os.path.join(path_tomat, file_excel)

# Cek apakah file Excel ada
if not os.path.exists(file_path_excel):
    print(f"File Excel tidak ditemukan di: {file_path_excel}")
    exit()

# Baca file Excel
dataset = pd.read_excel(file_path_excel)

fitur = dataset.iloc[:, 1:4].values
kelas = dataset.iloc[:, 4].values
tes_fitur = [[]]

# Gabungkan path ke file gambar
file_image = 'tomat1.jpg'  # Ganti dengan nama file gambar yang benar
file_name = os.path.join(path_tomat, file_image)

# Cek apakah file gambar ada
if not os.path.exists(file_name):
    print(f"File gambar tidak ditemukan di: {file_name}")
    exit()

# Baca gambar
src = cv2.imread(file_name, 1)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)

mask = cv2.dilate(mask.copy(), None, iterations=10)
mask = cv2.erode(mask.copy(), None, iterations=10)
b, g, r = cv2.split(src)
rgba = [b, g, r, mask]
dst = cv2.merge(rgba, 4)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
selected = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(selected)
cropped = dst[y:y+h, x:x+w]
mask = mask[y:y+h, x:x+w]

# HSV
hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
clt = KMeans(n_clusters=3)
labels = clt.fit_predict(image)
label_counts = Counter(labels)
dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

tes_fitur[0].append(dom_color[0])
tes_fitur[0].append(dom_color[1])
tes_fitur[0].append(dom_color[2])

# Normalisasi fitur
scaler = StandardScaler()
scaler.fit(fitur)

imputer = SimpleImputer(strategy='mean')

# Melakukan imputasi pada fitur-fitur yang memiliki nilai NaN
fitur = imputer.fit_transform(fitur)
tes_fitur = imputer.transform([tes_fitur[0]])  # Ubah ini untuk menggunakan imputasi yang benar

# Menggunakan Gaussian Naive Bayes sebagai classifier
classifier = GaussianNB()
classifier.fit(fitur, kelas)

while True:
    # Melakukan prediksi kelas
    kelas_pred = classifier.predict(tes_fitur)
    print("Kelas Prediksi:", kelas_pred[0])

    # Insert kematangan tomat ke dalam database
    insert_into_database(kelas_pred[0])

    time.sleep(10)