# IF5152 Computer Vision – Aplikasi Analisis Pipeline CV  


## Deskripsi   
Pipeline terdiri dari empat modul utama:

1.  **Image Filtering** – Gaussian & Median Filtering  
2.  **Edge Detection** – Sobel & Canny  
3.  **Feature Points Detection** – Harris, FAST, ORB, dan SIFT  
4.  **Geometry Transformation** – Homography (Projective Transform)

---

## 1. Petunjuk Instalasi  

Aplikasi dikembangkan menggunakan **Python 3.10+** dan disarankan dijalankan di dalam **virtual environment**.

### 1️⃣ Clone atau Unduh Proyek  
Unduh dan ekstrak folder proyek ke komputer
```
git clone https://github.com/kaylanamira/IF5152-Computer-Vision.git
```

### 2️⃣ Buat Virtual Environment  
Buka terminal di folder root proyek:
```bash
python -m venv venv
```

### 3️⃣ Aktifkan Virtual Environment  
**Windows (CMD/PowerShell):**
```bash
.env\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

Pastikan terminal menampilkan `(venv)` di awal prompt.

### 4️⃣ Instal Dependencies  
```bash
pip install scikit-image matplotlib numpy pandas opencv-contrib-python
```

> **Catatan:**  
> `opencv-contrib-python` diperlukan untuk detektor **SIFT** dan **FAST (cv2)**.  
> `scikit-image` digunakan untuk Harris, ORB, dan Canny.

---

##  2. Cara Menjalankan Aplikasi  

Setiap modul bersifat **independen** dan dapat dijalankan terpisah.  
> ⚠️ Jalankan semua perintah dari **folder root proyek**, bukan dari dalam subfolder.

### Modul 1 – Image Filtering  
```bash
python -m 01_filtering.filtering
```

### Modul 2 – Edge Detection  
```bash
python -m 02_edge.edge_detection
```

### Modul 3 – Feature Points Detection  
```bash
python -m 03_featurepoints.feature_points
```

### Modul 4 – Geometry Transformation  
```bash
python -m 04_geometry.geometry
```

Setiap modul secara otomatis akan:
- Memproses gambar standar (`skimage.data`) dan gambar pribadi (`inputs/personal/`)
- Menyimpan **gambar output**, **plot perbandingan**, dan **file `.csv`** berisi parameter hasil analisis

---

## 3. Fitur Unik & Pilihan Desain  

### Arsitektur Modular  
Semua parameter disimpan dalam dictionary konfigurasi (`*_CONFIG`) sehingga mudah dimodifikasi tanpa mengubah kode utama.

### utils.py Terpusat  
Fungsi-fungsi umum (load, save, plotting, marking, logging) dipusatkan di `utils.py` untuk menjaga kebersihan arsitektur.

### Integrasi Multi-Library  
- **OpenCV (`cv2`)** → digunakan untuk **SIFT** & **FAST** (karena versi `skimage` tidak stabil).  
- **scikit-image (`skimage`)** → digunakan untuk Harris, ORB, Gaussian, Canny, Sobel, dan Transformasi Geometrik.

### Penanganan Tipe  
- Harris / ORB → menerima `float32 [0,1]`  
- SIFT / FAST (cv2) → menerima `uint8 [0,255]`  
Konversi otomatis di `utils.py` memastikan kompatibilitas antarlibrary.

## Kontributor  
**Kayla Namira Mariadi**  
NIM 13522050 – Informatika ITB  
