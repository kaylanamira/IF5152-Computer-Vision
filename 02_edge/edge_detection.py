# Nama: Kayla Namira Mariadi
# NIM: 13522050
# Deskripsi: Runner untuk modul edge detection (Sobel & Canny).

from pathlib import Path
from skimage import filters, feature

import utils


EDGE_CONFIG = [
    {
        "name": "sobel",
        "function": filters.sobel,
        "params": {}, 
        "notes": "Sobel"
    },
    {
        "name": "canny_sigma1_low_thresh",
        "function": feature.canny,
        "params": {"sigma": 1.0, "low_threshold": 0.05, "high_threshold": 0.15},
        "notes": "Eksperimen Threshold: Sigma=1.0, Thresholds rendah"
    },
    {
        "name": "canny_sigma1_high_thresh",
        "function": feature.canny,
        "params": {"sigma": 1.0, "low_threshold": 0.1, "high_threshold": 0.3},
        "notes": "Eksperimen Threshold: Sigma=1.0, Thresholds tinggi"
    },
    {
        "name": "canny_sigma3_low_thresh",
        "function": feature.canny,
        "params": {"sigma": 3.0, "low_threshold": 0.05, "high_threshold": 0.15},
        "notes": "Eksperimen Threshold: Sigma=3.0, Thresholds rendah"
    },
    {
        "name": "canny_sigma3_high_thresh",
        "function": feature.canny,
        "params": {"sigma": 3.0, "low_threshold": 0.1, "high_threshold": 0.3},
        "notes": "Eksperimen Threshold: Sigma=3.0, Thresholds tinggi"
    }
]


def process_one_image(img_name, img, base_output_dir):
    """
    Memproses satu gambar dengan semua metode edge detection di config.
    """
    print(f"\nMemproses image: {img_name}")
    
    # Ubah gambar menjadi gambar grayscale
    img_gray = utils.to_gray(img)
    
    img_output_dir = base_output_dir / img_name
    
    # Siapkan dict untuk plot perbandingan
    comparison_plots = {"Original (Grayscale)": img_gray}

    utils.save_img(img_output_dir / f"{img_name}_original_gray.png", img_gray)

    image_params_log = []

    for config in EDGE_CONFIG:
        print(f"  Menerapkan: {config['name']}...")
        filter_func = config["function"]
        params = config["params"].copy()
        
        # Terapkan filter
        # Canny mengembalikan boolean, Sobel mengembalikan float
        # Kita konversi ke float (0-1) agar konsisten saat disimpan
        edge_img = filter_func(img_gray, **params).astype(float)
        
        output_filename = img_output_dir / f"{img_name}_{config['name']}.png"
        utils.save_img(output_filename, edge_img)

        comparison_plots[config['name']] = edge_img

        log_entry = {"image": img_name, "method": config['name'], "notes": config['notes']}
        log_entry.update(config["params"])
        image_params_log.append(log_entry)
    
    # Simpan plot perbandingan untuk gambar ini
    plot_filepath = base_output_dir / f"comparison_{img_name}.png"
    utils.plot_comparison(
        comparison_plots, 
        f"Edge Detection Comparison for {img_name}", 
        plot_filepath
    )

    return image_params_log


def run_all_edges(base_output_dir):
    """
    Fungsi utama yang me-loop semua gambar dan memanggil prosesor.
    """
    print("--- 2. Menjalankan Modul Edge Detection ---")
    imgs = utils.load_images()
    all_params_log = []

    for img_name, img in imgs.items():
        logs_from_image = process_one_image(img_name, img, base_output_dir)
        all_params_log.extend(logs_from_image)

    csv_path = base_output_dir / "edge_parameters.csv"
    utils.save_params_to_csv(csv_path, all_params_log)
    
    print("\n--- Modul Edge Detection Selesai ---")


def main():
    output_dir = Path(__file__).resolve().parent
    run_all_edges(output_dir)

if __name__ == "__main__":
    main()