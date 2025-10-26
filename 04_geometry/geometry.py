# Nama: Kayla Namira Mariadi
# NIM: 13522050
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pathlib import Path
from skimage import transform, img_as_float
import utils

def save_matrix_to_txt(filepath, matrix, header=""):
    """
    Menyimpan matriks NumPy ke file .txt dengan format yang rapi.
    """
    filepath = Path(filepath)
    with open(filepath, 'w') as f:
        if header:
            f.write(header + "\n\n")
        # Simpan matriks dengan format float 4 angka di belakang koma
        np.savetxt(f, matrix, fmt='%.4f', delimiter='\t')
    print(f"Matrix parameter disimpan di: {filepath.name}")

def plot_transform_overlay(img_orig, img_warped, src_pts, dst_pts, title, filepath):
    """
    Membuat plot overlay side-by-side untuk transformasi geometri.
    Menampilkan gambar asli + titik src, dan gambar warped + titik dst.
    Poin (pts) diasumsikan dalam format (x, y) / (col, row).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='tight')
    
    # --- Plot Gambar Asli ---
    axes[0].imshow(img_orig, cmap='gray')
    # plt.scatter butuh x dan y secara terpisah
    axes[0].scatter(src_pts[:, 0], src_pts[:, 1], c='r', s=40, marker='x', label='Source Pts')
    axes[0].set_title("Original Image + Source Points")
    axes[0].axis('off')
    axes[0].legend()

    # --- Plot Gambar Hasil Transformasi ---
    axes[1].imshow(img_warped, cmap='gray')
    axes[1].scatter(dst_pts[:, 0], dst_pts[:, 1], c='g', s=40, marker='+', label='Dest Pts')
    axes[1].set_title("Warped Image + Destination Points")
    axes[1].axis('off')
    axes[1].legend()
    
    # --- Simpan ---
    fig.suptitle(title, fontsize=16)
    filepath = Path(filepath)
    plt.savefig(filepath)
    plt.close()
    print(f"Overlay plot disimpan di: {filepath.name}")


def run_all_transforms(base_output_dir):
    """
    Fungsi utama untuk menjalankan simulasi transformasi.
    """
    print("--- 4. Menjalankan Modul Geometry ---")
    imgs = utils.load_images()
    
    # Kita hanya butuh checkerboard untuk modul ini
    img_checker = imgs.get('checkerboard')
    if img_checker is None:
        print("ERROR: 'checkerboard' tidak ditemukan oleh utils.load_images().")
        return
        
    # Pastikan grayscale dan float
    img_checker = utils.to_gray(img_as_float(img_checker))
    h, w = img_checker.shape
    
    # 1. Tentukan Titik Sumber (Source)
    # Titik sumber adalah 4 sudut gambar (format: x, y / col, row)
    src_pts_4 = np.array([
        [0, 0],     # Top-left
        [w - 1, 0],   # Top-right
        [w - 1, h - 1], # Bottom-right
        [0, h - 1]    # Bottom-left
    ])
    # Affine transform hanya butuh 3 titik
    src_pts_3 = src_pts_4[:3] 

    TRANSFORM_CONFIG = [
        {
            "name": "projective_transform",
            "transform_type": transform.ProjectiveTransform,
            "src_points": src_pts_4,
            "dst_points": np.array([
                [w*0.1, h*0.2],  # Top-left (ditarik ke dalam)
                [w*0.9, h*0.1],  # Top-right (ditarik ke dalam)
                [w*0.8, h*0.9],  # Bottom-right (ditarik ke dalam)
                [w*0.2, h*0.8]   # Bottom-left (ditarik ke dalam)
            ]),
            "notes": "Simulasi Homography (Projective Transform) - Tampilan perspektif"
        },
        {
            "name": "affine_shear_transform",
            "transform_type": transform.AffineTransform,
            "src_points": src_pts_3, # Affine hanya butuh 3 titik
            "dst_points": np.array([
                [0, 0],              # Top-left (tetap)
                [w - 1, 0],          # Top-right (tetap)
                [w - 150, h - 1]     # Bottom-right (ditarik ke kiri -> 'shear')
            ]),
            "notes": "Simulasi Affine Transform - Efek 'Shear'"
        }
    ]

    all_params_log = [] 

    # 3. Loop dan jalankan setiap transformasi
    for config in TRANSFORM_CONFIG:
        print(f"  Menerapkan: {config['name']}...")
        
        # Inisialisasi transformasi
        t = config["transform_type"]()
        src_pts = config["src_points"]
        dst_pts = config["dst_points"]

        # 4. Estimasi Matriks
        if not t.estimate(src_pts, dst_pts):
            print(f"  ERROR: Estimasi matriks gagal untuk {config['name']}")
            continue
            
        matrix = t.params # Ini adalah matriks 3x3 yang kita cari

        # 5. Terapkan Transformasi (Warp)
        # Kita gunakan t.inverse untuk memetakan piksel output kembali ke input
        warped_img = transform.warp(img_checker, t.inverse, output_shape=(h, w))
        
        # 6. Buat Overlay Plot
        # Untuk plotting, kita butuh 4 titik tujuan
        if config["transform_type"] == transform.AffineTransform:
            # Hitung titik ke-4 secara manual untuk plot overlay affine
            plot_dst_pts = t(src_pts_4)
        else:
            plot_dst_pts = dst_pts
            
        plot_path = base_output_dir / f"overlay_{config['name']}.png"
        plot_transform_overlay(
            img_checker, warped_img, src_pts_4, plot_dst_pts, 
            config['name'], plot_path
        )

        # 7. Simpan Matriks Parameter
        matrix_path = base_output_dir / f"matrix_{config['name']}.txt"
        header = f"Matriks Estimasi ({config['name']})\n" + \
                 f"Tipe: {config['transform_type'].__name__}"
        save_matrix_to_txt(matrix_path, matrix, header=header)

        # 8. Siapkan Log untuk Worksheet
        log_entry = {
            "transform_name": config['name'],
            "type": config['transform_type'].__name__,
            "src_points_used": str(src_pts.tolist()),
            "dst_points_used": str(dst_pts.tolist()),
            "notes": config['notes']
        }
        all_params_log.append(log_entry)

    # 9. Simpan Worksheet
    csv_path = base_output_dir / "geometry_worksheet.csv"
    utils.save_params_to_csv(csv_path, all_params_log)
    
    print("\n--- Modul Geometry Selesai ---")


def main():
    output_dir = Path(__file__).resolve().parent
    run_all_transforms(output_dir)

if __name__ == "__main__":
    main()