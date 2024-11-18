from PIL import Image
import cv2
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import mysql.connector
import os
import time
from rembg import remove

class pcv:
    @staticmethod
    def remove_background(file_gambar, output_path):
        if not os.path.exists(file_gambar):
            print(f"Error: File '{file_gambar}' tidak ditemukan.")
            return None

        input_image = Image.open(file_gambar)
        output_image = remove(input_image)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        output_image.save(output_path)
        print(f"Gambar dengan latar belakang dihapus disimpan di {output_path}")
        return np.array(output_image)
    
    @staticmethod
    def ambil_rata_rgb(gambar):
        gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
        tinggi, lebar, _ = gambar_rgb.shape
        piksel_tengah = gambar_rgb[tinggi // 2, lebar // 2]
        print("Nilai RGB real dari piksel tengah:", piksel_tengah)
        return piksel_tengah[:2]  # Return only the first two RGB values (R and G)

class sistemcerdas:
    def __init__(self, pcv):
        self.pcv = pcv
        self.nb_classifier = GaussianNB(var_smoothing=1e-9)
        self.scaler = StandardScaler()
        self.accuracy = 0
        self.db_connection = None

    def open_database(self):
        self.db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="tomatin"
        )
    
    def close_database(self):
        if self.db_connection:
            self.db_connection.close()

    def train(self, data_file):
        self.open_database()
        try:
            # Read the CSV file
            data = pd.read_csv(data_file)
        except ValueError as e:
            print(f"Error reading the CSV file: {e}")
            return

        # Ensure only numeric features are selected
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check if there are numeric columns to avoid index errors
        if len(numeric_columns) < 2:
            print("Error: Not enough numeric columns for training.")
            return
        
        # Pisahkan fitur dan label
        X = data[numeric_columns[:-1]].values  # All numeric columns except the last one
        y = data.iloc[:, -1].values  # Assuming the last column is the label

        # Normalisasi data
        X = self.scaler.fit_transform(X)

        # Split data menjadi training dan testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Latih model Naive Bayes
        self.nb_classifier.fit(X_train, y_train)

        # Evaluasi model
        y_pred = self.nb_classifier.predict(X_test)

        # Matriks Kebingungan
        class_matrix = confusion_matrix(y_test, y_pred)

        # Laporan Klasifikasi
        class_report = classification_report(y_test, y_pred)

        # Hitung Akurasi
        self.accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"Akurasi Model: {self.accuracy:.2f}%")
        print("Confusion Matrix:\n", class_matrix)
        print("Classification Report:\n", class_report)
            
    def simpan_database(self, category):
        category_mapping = {
            'WellDone': 1,
            'Madium': 2,
            'Raw': 3
        }

        if category in category_mapping:
            category_value = category_mapping[category]
        else:
            print(f"Kategori tidak valid: {category}")
            return

        try:
            cursor = self.db_connection.cursor()
            sql_insert = "INSERT INTO tomat (id_kategori) VALUES (%s)"
            values = (category_value,)
            cursor.execute(sql_insert, values)
            self.db_connection.commit()
            print(f"Data disimpan dengan kategori {category_value}")
        except mysql.connector.Error as e:
            print(f"Error while saving to database: {e}")
        finally:
            cursor.close()

if __name__ == "__main__":
    # Inisialisasi objek
    pcv_obj = pcv()
    sistem_obj = sistemcerdas(pcv_obj)

    data_file = "D:/New folder/baru/datasateBaru.csv"
    sistem_obj.train(data_file)

    # Path gambar yang ingin diproses
    sample_image_path = 'D:/New folder/baru/TOMAT/tomat30.jpg'

    # Ekstraksi fitur dan prediksi kategori
    while True:
        new_image_features = pcv_obj.ambil_rata_rgb(cv2.imread(sample_image_path))
        new_image_features = np.array(new_image_features).reshape(1, -1)  # Should have two features
        new_image_features = sistem_obj.scaler.transform(new_image_features)

        predicted_category = sistem_obj.nb_classifier.predict(new_image_features)[0]
        sistem_obj.simpan_database(predicted_category)

        print(f"Prediksi kategori: {predicted_category}")
        print(f"Akurasi data test: {sistem_obj.accuracy:.2f}%")

        # Tunggu 30 detik sebelum memproses gambar berikutnya
        time.sleep(30)