import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
from rembg import remove

class SistemCerdasPCV:
    @staticmethod
    def resize_image(image, lebar=100, tinggi=100):
        lebar = int(image.shape[1] * lebar / 100)
        tinggi = int(image.shape[0] * tinggi / 100)
        dimensi = (lebar, tinggi)
        return cv2.resize(image, dimensi)

    @staticmethod
    def adjust_brightness_contrast(image, brightness=30, contrast=30):
        return cv2.convertScaleAbs(image, alpha=1 + (contrast / 100), beta=brightness)

    @staticmethod
    def ambil_rgb_real(gambar):
        # Pastikan gambar dalam format RGB
        gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
        
        # Ambil nilai RGB dari piksel tengah gambar
        tinggi, lebar, _ = gambar_rgb.shape
        piksel_tengah = gambar_rgb[tinggi // 2, lebar // 2]
        
        print("Nilai RGB real dari piksel tengah:", piksel_tengah)
        return piksel_tengah

    @staticmethod
    def label_kematangan(rgb):
        R, G, B = rgb
        
        # Atur batasan untuk kematangan berdasarkan nilai RGB
        if R > 150 and G < 150 and B < 150:  # Merah pekat (R tinggi)
            return 'Matang'
        elif R > 120 and G > 120 and B < 130:  # Kuning ke oranye
            return 'Matang'
        else:  # Jika tidak memenuhi syarat untuk Merah Pekat, Matang, atau Setengah Matang
            return 'Matang'

    @staticmethod
    def simpan_hasil(gambar, path_output):
        gambar.save(path_output)

    @staticmethod
    def remove_background(input_path, output_path):
        # Membuka gambar dengan PIL
        input_image = Image.open(input_path)
        
        # Menghapus latar belakang
        output_image = remove(input_image)
        
        # Pastikan folder tujuan ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Menyimpan hasil dalam format PNG agar latar belakang tembus
        output_image.save(output_path)
        print(f"Gambar dengan latar belakang dihapus disimpan di {output_path}")

    @staticmethod
    def proses_gambar(folder_path):
        data = {'Nama Gambar': [], 'R': [], 'G': [], 'B': [], 'label': []}
        
        for namagambar in os.listdir(folder_path):
            path_file = os.path.join(folder_path, namagambar)
            if path_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_path = os.path.join("hasil_removeBG_Matang", f"hasil_{namagambar.split('.')[0]}.png")
                SistemCerdasPCV.remove_background(path_file, output_path)

                # Load gambar hasil tanpa latar belakang untuk analisis lanjutan
                gambar_asli = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
                if gambar_asli is not None:
                    rgb_real = SistemCerdasPCV.ambil_rgb_real(gambar_asli)
                    label = SistemCerdasPCV.label_kematangan(rgb_real)

                    data['Nama Gambar'].append(namagambar)
                    data['R'].append(rgb_real[0])
                    data['G'].append(rgb_real[1])
                    data['B'].append(rgb_real[2])
                    data['label'].append(label)

        df = pd.DataFrame(data)
        df.to_excel('hasil_Matang.xlsx', index=False)
        print("Data hasil pengolahan disimpan dalam 'hasil_Matang.xlsx'")

if __name__ == "__main__":
    folder_path = "D:/New folder/Tomat.in/PCV_SistemCerdas_Tomat.In/REVISI/TM_BARU"
    # os.makedirs("hasil_cropping_Setengah Matang", exist_ok=True)
    SistemCerdasPCV.proses_gambar(folder_path)
