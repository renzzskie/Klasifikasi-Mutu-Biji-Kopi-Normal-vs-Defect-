# Klasifikasi Mutu Biji Kopi (Normal vs Defect) Menggunakan Ekstraksi Fitur GLCM serta Perbandingan Metode KNN, SVM, dan Random Forest

## Nama Anggota

- RENDY WAHYU ISLAMI : F1D02410133
- MUHAMMAD AKBAR : F1D02410075
- WIMAR ARYASMARTA PRAKASA : F1D02410026
- M. ISMAIL CAHYADI : F1D02410120

## Project Overview

Project ini bertujuan untuk melakukan klasifikasi mutu biji kopi ke dalam dua kelas, yaitu normal (premium) dan defect, menggunakan pendekatan pengolahan citra digital secara manual. Seluruh tahapan preprocessing, mulai dari operasi spasial, deteksi tepi, hingga thresholding, diimplementasikan tanpa menggunakan fungsi bawaan OpenCV agar pemahaman terhadap konsep dasar pengolahan citra benar-benar teruji. Fokus utama project ini adalah pada ketepatan pemilihan tahapan preprocessing dan proses ekstraksi fitur, bukan pada nilai akurasi akhir model.

Sesuai arahan modul, dilakukan empat kali percobaan dengan kombinasi preprocessing yang berbeda untuk melihat bagaimana setiap kombinasi mempengaruhi hasil ekstraksi fitur dan performa klasifikasi:

- Percobaan 1: prepro1 (Median Filter + Normalisasi)
- Percobaan 2: prepro2 (Histogram Equalization + Sharpening)
- Percobaan 3: prepro3 (Smoothing Mean Filter + Edge Detection Sobel + Thresholding)
- Percobaan 4: prepro4 (Smoothing + Edge Detection + Thresholding + Normalisasi)

## Import Library

Library yang digunakan pada project ini meliputi:

```python
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

`cv2` hanya digunakan untuk membaca file gambar (`cv.imread`) dan konversi ruang warna untuk keperluan visualisasi, sedangkan seluruh proses preprocessing inti dibuat manual menggunakan numpy.

## Load Data

Dataset dibaca dari folder `dataset` yang berisi dua subfolder sesuai label, yaitu `defect` dan `premium`. Setiap gambar dibaca, dikonversi ke grayscale, lalu disimpan ke dalam list bersama label dan nama filenya.

```python
data = []
labels = []
file_name = []

for label in os.listdir("dataset"):
    folder_path = os.path.join("dataset", label)
    for fname in os.listdir(folder_path):
        img_path = os.path.join(folder_path, fname)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        data.append(img)
        labels.append(label)
        file_name.append(fname)

labels = np.array(labels)
print(f"Jumlah data: {len(data)}")
```

### Data Understanding

Pada tahap ini dilakukan eksplorasi untuk memahami karakteristik dataset, seperti jumlah data per kelas, kondisi pencahayaan, serta variasi ukuran gambar. Pemahaman ini menjadi dasar pemilihan teknik preprocessing yang tepat, misalnya kebutuhan resize karena ukuran gambar antar sampel tidak seragam, serta kebutuhan filter penghalus karena adanya noise pada gambar hasil pemotretan.

## Data Preparation

### Data Augmentation

Karena jumlah data per kelas berada pada rentang yang mencukupi, augmentasi data bersifat opsional dan hanya diterapkan jika diperlukan untuk menyeimbangkan jumlah sampel antar kelas.

### Preprocessing

Seluruh fungsi preprocessing dibuat secara manual menggunakan numpy, tanpa memanfaatkan fungsi siap pakai dari OpenCV.

**Kernel yang digunakan**

```python
kernelSharpening = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 8/9, 1/9],
    [1/9, 1/9, 1/9]
])
```

Kernel untuk mempertajam gambar. Nilai tengah 8/9 yang lebih besar dari sekitarnya membuat piksel pusat lebih dominan sehingga tepi dan detail gambar terlihat lebih tajam.

```python
sobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobelY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
```

Dua kernel Sobel untuk mendeteksi tepi. sobelX mendeteksi perubahan intensitas arah horizontal, sedangkan sobelY mendeteksi perubahan arah vertikal.

```python
prewittX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittY = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
```

Dua kernel Prewitt untuk mendeteksi tepi, mirip dengan Sobel namun tanpa pembobotan lebih besar pada baris atau kolom tengah, sehingga hasilnya sedikit lebih sensitif terhadap noise.

```python
robertsX = np.array([[1,0],[0,-1]])
robertsY = np.array([[0,1],[-1,0]])
```

Dua kernel Roberts berukuran dua kali dua untuk mendeteksi tepi dengan membandingkan piksel secara diagonal. Lebih cepat dihitung namun lebih rentan terhadap noise dibanding Sobel maupun Prewitt.

**Fungsi dasar**

`normalisasi(image)` melakukan peregangan kontras manual agar nilai piksel tersebar merata dari 0 sampai 255, dengan piksel paling gelap menjadi 0 dan paling terang menjadi 255.

`histogram_equalization(image)` menghitung histogram gambar secara manual, menghitung Cumulative Distribution Function (CDF), lalu menormalisasinya ke rentang 0-255 sebagai tabel pemetaan nilai piksel baru agar distribusi intensitas menjadi lebih merata.

**Fungsi spasial**

`convolution(img, kernel)` melakukan operasi konvolusi manual antara gambar dan kernel, dengan padding agar ukuran output sama dengan input.

`edge(img, kernelx, kernely)` mendeteksi tepi dengan menggabungkan hasil konvolusi kernel sumbu x dan y, lalu menjumlahkan nilai absolut keduanya untuk mendapatkan kekuatan tepi di setiap piksel.

`filter(img, size, mode)` memiliki tiga mode:

- `mean`: menghitung rata-rata piksel dalam area kernel untuk menghaluskan gambar.
- `median`: mengurutkan nilai piksel menggunakan bubble sort manual lalu mengambil nilai tengahnya, efektif menghilangkan noise tanpa mengaburkan tepi gambar.
- `modus`: menghitung nilai piksel yang paling sering muncul dalam area kernel menggunakan dictionary, cocok untuk menghilangkan noise berupa nilai piksel acak.

`thresholding(image, n)` mengubah gambar grayscale menjadi gambar biner berdasarkan nilai ambang n, memisahkan objek dari latar belakang berdasarkan tingkat kecerahan.

`resize(image, target_size)` mengubah ukuran gambar ke dimensi target secara manual menggunakan metode nearest neighbor, memetakan setiap piksel gambar baru ke posisi piksel terdekat di gambar asli.

**Pipeline preprocessing per eksperimen**

```python
# Pipeline preprocessing per eksperimen
def prepro1(image):
    img_median = filter(image, 3, 'median')
    img_norm = normalisasi(img_median)
    return img_norm

def prepro2(image):
    img_he = histogram_equalization(image)
    img_sharpened = convolution(img_he, kernelSharpening)
    return np.clip(img_sharpened, 0, 255).astype(np.uint8)

def prepro3(image):
    img_smooth = filter(image, 3, 'mean')
    img_edge = edge(img_smooth, sobelX, sobelY)
    img_tresh = thresholding(img_edge, 15)
    return img_tresh

def prepro4(image):
    img_smooth = filter(image, 3, 'mean')
    img_edge = edge(img_smooth, sobelX, sobelY)
    img_tresh = thresholding(img_edge, 15)
    img_norm = normalisasi(img_tresh)
    return img_norm
```

`prepro1` menghaluskan gambar dengan median filter untuk menghilangkan noise, lalu menormalisasi hasilnya.

`prepro2` meratakan distribusi intensitas dengan histogram equalization agar detail di area gelap maupun terang lebih terlihat, lalu mempertajam tepi melalui konvolusi dengan kernelSharpening.

`prepro3` menghaluskan gambar dengan mean filter sebelum deteksi tepi, menerapkan deteksi tepi Sobel, lalu mengubahnya menjadi gambar biner melalui thresholding agar hanya informasi tepi biji kopi yang ditampilkan.

`prepro4` merupakan pengembangan dari prepro3, di mana hasil thresholding dari deteksi tepi kemudian dinormalisasi ulang agar sebaran kontrasnya lebih optimal sebelum diekstrak oleh GLCM.

Pemanggilan pipeline pada dataset:

```python
dataPreprocessed = []
for i in range(len(data)):
    img_resized = resize(data[i], target_size=(256, 256))
    img_final = prepro4(img_resized) # Ubah prepro sesuai eksperimen (prepro1/2/3/4)
    dataPreprocessed.append(img_final)

dataPreprocessed = np.array(dataPreprocessed)
print(f"Selesai memproses {len(dataPreprocessed)} gambar.")
```

Setiap gambar diseragamkan ukurannya menjadi 256x256 piksel menggunakan fungsi resize manual, lalu diproses melalui pipeline eksperimen yang dipilih sebelum disimpan ke dalam array numpy untuk tahap ekstraksi fitur.

## Feature Extraction

Ekstraksi fitur dilakukan menggunakan Gray Level Co-occurrence Matrix (GLCM) pada sudut 0, 45, 90, dan 135 derajat secara simetris, dengan uji coba pada distance 1 sampai 5. Fitur yang dihitung meliputi Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Entropy, dan ASM.

```python
def glcm(image, derajat):
    ...
```

## Feature Selection

Seleksi fitur dilakukan menggunakan correlation matrix terhadap fitur hasil ekstraksi GLCM, untuk mengidentifikasi dan menghilangkan fitur yang saling berkorelasi tinggi sehingga mengurangi redundansi.

```python
correlation = hasilEkstrak.drop(columns=['Label','Filename']).corr()
```

## Splitting Data

Data dibagi menjadi data training dan data testing dengan perbandingan 80:20.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
```

## Normalization

Normalisasi data fitur dilakukan menggunakan teknik standarization (z-score) agar setiap fitur memiliki skala yang setara sebelum masuk ke tahap modeling.

```python
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_train.mean()) / X_train.std()
```

## Modeling

Tiga model klasifikasi dilatih menggunakan fitur GLCM yang telah diseleksi dan dinormalisasi, yaitu Random Forest, SVM, dan KNN.

```python
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

def generateClassificationReport(y_true, y_pred):
    print(classification_report(y_true, y_pred))
```

Random Forest membangun kumpulan pohon keputusan yang secara kolektif menentukan kelas mutu biji kopi. SVM mencari hyperplane dengan margin terbesar untuk memisahkan kelas defect dan premium. KNN tidak membangun model, melainkan menyimpan seluruh data latih dan mencari k tetangga terdekat di ruang fitur saat melakukan prediksi.

Setiap model dievaluasi dua kali, pertama pada training set untuk melihat seberapa baik model mempelajari data latih, kemudian pada testing set untuk mengukur kemampuan generalisasinya terhadap data baru yang belum pernah dilihat sebelumnya.

## Evaluation

Berikut adalah rekam jejak akurasi pengujian (Testing Accuracy) dari keempat skenario preprocessing yang dilakukan:

| Skenario Preprocessing                 | Akurasi KNN | Akurasi SVM | Akurasi Random Forest |
| :------------------------------------- | :---------: | :---------: | :-------------------: |
| **P1** (Median + Norm)                 |    62.5%    |    65.0%    |         55.0%         |
| **P2** (HE + Sharpening)               |    52.5%    |    72.5%    |         67.5%         |
| **P3** (Smooth + Edge + Thresh)        |   63.33%    |    75.0%    |        71.67%         |
| **P4** (Smooth + Edge + Thresh + Norm) |  **82.5%**  |  **80.0%**  |       **75.0%**       |

Dari tabel di atas, **Percobaan 4 (P4)** menghasilkan performa tertinggi untuk semua model. Berikut adalah rincian performa model (Training vs Testing) khusus pada dataset hasil **Percobaan 4**:

| Model         | Akurasi Training | Akurasi Testing |
| ------------- | ---------------- | --------------- |
| Random Forest | 0.95             | 0.75            |
| SVM           | 0.8125           | 0.80            |
| KNN           | 0.7875           | 0.825           |

**Random Forest** menunjukkan akurasi training yang sangat tinggi (0.95) namun turun cukup signifikan pada testing (0.75). Selisih yang besar ini mengindikasikan model mengalami overfitting, yaitu model terlalu menghafal pola spesifik dari data latih sehingga kurang optimal saat menghadapi data baru. Pada confusion matrix testing, kesalahan klasifikasi masih cukup banyak terjadi pada kelas defect.

**SVM** menunjukkan akurasi training (0.8125) dan testing (0.80) yang relatif konsisten dengan selisih kecil. Hal ini menunjukkan SVM memiliki kemampuan generalisasi yang baik, tidak overfitting maupun underfitting, karena performanya stabil antara data latih dan data baru.

**KNN** justru menunjukkan akurasi testing (0.825) yang lebih tinggi dibanding training (0.7875). Pola ini wajar karena KNN menggunakan nilai k lebih dari satu, sehingga prediksi pada data latih juga mempertimbangkan tetangga dari kelas lain alih-alih hanya menghafal dirinya sendiri. Hasil ini menunjukkan KNN tidak mengalami overfitting dan mampu menggeneralisasi dengan baik pada data uji.

## Kesimpulan Akhir

1. **Pengaruh Preprocessing:** Penambahan tahapan deteksi tepi (Edge Detection) yang diakhiri dengan Normalisasi (Skenario P4) terbukti sangat efektif dalam menonjolkan fitur tekstur cacat (defect) pada biji kopi. Normalisasi di tahap akhir membantu GLCM membaca matriks ketetanggaan piksel dengan rentang nilai yang lebih konsisten, sehingga mendongkrak akurasi secara signifikan dibandingkan P1, P2, dan P3.
2. **Performa Model ML:** Dari ketiga model yang diuji, **SVM dan KNN** menunjukkan kemampuan generalisasi yang lebih baik dibanding Random Forest. Random Forest cenderung mengalami _overfitting_ pada kombinasi fitur tekstur GLCM ini, sementara KNN berhasil mencapai akurasi _testing_ tertinggi sebesar 82.5%.
