# Nama: Kayla Namira Mariadi
# NIM: 13522050
# Deskripsi: Modul helper untuk memuat gambar, menyimpan output, dan plotting

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import data, img_as_float, io
from skimage.color import gray2rgb, rgb2gray
from skimage.util import img_as_ubyte
import io as skio

def to_gray(img):
    """Konversi gambar ke grayscale jika berwarna."""
    if img.ndim == 3:
        return rgb2gray(img)
    return img

def load_images():
    """
    Memuat dan memproses semua gambar standar dan tambahan.
    Mengembalikan dictionary berisi gambar-gambar.
    """
    print("Memuat gambar...")

    personal_image_path = "personal.jpeg" 
    personal_image_key = "personal"
    imgs = {
        "cameraman": to_gray(data.camera()),
        "coins": to_gray(data.coins()),
        "checkerboard": to_gray(data.checkerboard()),
        "astronaut": img_as_float(data.astronaut()), 
        "chelsea": img_as_float(data.chelsea())      
    }
    try:
        try:
            root_dir = Path(__file__).resolve().parent
            personal_image_path = root_dir / personal_image_path
        except NameError:
            personal_image_path = Path(personal_image_path)
            
        img_pribadi = io.imread(str(personal_image_path))
        img_pribadi_float = img_as_float(img_pribadi)
        imgs[personal_image_key] = img_pribadi_float
        
    except FileNotFoundError:
        print(f"Gambar pribadi '{personal_image_path}' tidak ditemukan.")
    except Exception as e:
        print(f"ERROR saat memuat gambar pribadi: {e}")

    print("Selesai memuat gambar.")
    return imgs

def save_img(filepath, img):
    """
    Menyimpan gambar ke file, membuat direktori jika perlu.
    Menangani konversi tipe data (float -> ubyte) dan grayscale -> RGB.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if img.dtype in (np.float32, np.float64):
        img = img_as_ubyte(img)
        
    # Konversi grayscale ke RGB untuk konsistensi (opsional, tapi aman)
    if img.ndim == 2:
        img = gray2rgb(img)
            
    io.imsave(str(filepath), img)

def save_params_to_csv(filepath, params_list):
    """
    Menyimpan daftar dictionary parameter ke file CSV.
    """
    filepath = Path(filepath)
    df = pd.DataFrame(params_list)
    df.to_csv(filepath, index=False)
    print(f"\nParameter disimpan di: {filepath.name}")

def plot_comparison(images_dict, title, filepath):
    """
    Membuat plot perbandingan hasil image dalam 2 BARIS dan menyimpannya ke file.
    """
    n = len(images_dict)
    
    n_rows = 2
    n_cols = (n + 1) // 2 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), layout='tight')
    
    axes_flat = axes.ravel()
        
    for (ax, (label, img)) in zip(axes_flat, images_dict.items()):
        ax.imshow(img, cmap='gray') 
        ax.set_title(label)
        ax.axis('off')
    
    n_total_plots = n_rows * n_cols
    for i in range(n, n_total_plots):
        axes_flat[i].axis('off')
        
    fig.suptitle(title, fontsize=16)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print(f"Comparison plot disimpan di: {filepath.name}")
    plt.close()

def create_marked_image(image, coords, color='r', marker_size=10):
    """
    Menggambar keypoints (koordinat) pada gambar dan mengembalikannya 
    sebagai numpy array.
    """
    
    is_gray = False
    if image.ndim == 2:
        image = gray2rgb(image) # Ubah ke RGB agar bisa ditandai warna
        is_gray = True

    # Buat plot di memory
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Tampilkan gambar
    if is_gray:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    
    # Gambar koordinat. Plot butuh (x, y) padahal coords (row, col)
    # Jadi kita balik: coords[:, 1] adalah x, coords[:, 0] adalah y
    ax.scatter(coords[:, 1], coords[:, 0], c=color, s=marker_size, 
               marker='.', edgecolor='none', alpha=0.7)
    ax.axis('off')
    
    # Simpan ke buffer memory
    buf = skio.BytesIO()
    # bbox_inches='tight' dan pad_inches=0 untuk hapus border putih
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Baca kembali sebagai image array
    marked_img = io.imread(buf)
    plt.close(fig) # Tutup plot agar tidak tampil di notebook
    
    return marked_img