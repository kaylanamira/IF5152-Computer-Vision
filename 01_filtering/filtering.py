# Nama: Kayla Namira Mariadi
# NIM: 13522050
# Deskripsi: Runner untuk modul image filtering. 

from pathlib import Path
from skimage import filters
from skimage.morphology import disk 
import utils

FILTER_CONFIG = [
    {
        "name": "gaussian_sigma1",       
        "function": filters.gaussian,    
        "base_params": {"sigma": 1.0},       
        "requires_gray": False, 
        "color_params": {"channel_axis": -1}, 
        "notes": "Standard smoothing"    
    },
    {
        "name": "gaussian_sigma3",
        "function": filters.gaussian,
        "base_params": {"sigma": 3.0},
        "requires_gray": False, 
        "color_params": {"channel_axis": -1}, 
        "notes": "Strong smoothing"
    },
    {
        "name": "median_disk3",
        "function": filters.median,
        "base_params": {"footprint": disk(3)},   
        "requires_gray": True, 
        "color_params": {}, 
        "notes": "Salt-pepper removal, k=3"
    },
    {
        "name": "median_disk5",
        "function": filters.median,
        "base_params": {"footprint": disk(5)},
        "requires_gray": True, 
        "color_params": {}, 
        "notes": "Strong Salt-pepper removal, k=5"
    },
    {
        "name": "sobel",
        "function": filters.sobel,
        "base_params": {},
        "requires_gray": True, 
        "color_params": {}, 
        "notes": "Gradient magnitude (edge)"
    }
]

def process_one_image(img_name, img, base_output_dir):
    """
    Memproses satu gambar dengan semua filter di config.
    """
    print(f"\nMemproses image: {img_name}")
    
    img_output_dir = base_output_dir / img_name
    is_color = img.ndim == 3
    
    comparison_plots = {"Original": utils.to_gray(img)}

    # 1. Simpan gambar original
    utils.save_img(img_output_dir / f"{img_name}_original.png", img)

    image_params_log = []

    # 2. Loop melalui semua konfigurasi filter
    for config in FILTER_CONFIG:
        filter_func = config["function"]
        # Ambil parameter dasar
        params = config["base_params"].copy()
        img_to_filter = img

        if is_color:
            if config["requires_gray"]:
                img_to_filter = utils.to_gray(img)
            else:
                # Jika tidak butuh gray, tambahkan params khusus warna
                params.update(config["color_params"])

        # Terapkan filter
        filtered_img = filter_func(img_to_filter, **params)
        
        # 3. Simpan hasil filter individual
        output_filename = img_output_dir / f"{img_name}_{config['name']}.png"
        utils.save_img(output_filename, filtered_img)

        # 4. Tambahkan ke dict plot perbandingan
        comparison_plots[config['name']] = utils.to_gray(filtered_img)

        # 5. Parameter untuk CSV
        log_entry = {"image": img_name, "filter_name": config['name'], "notes": config['notes']}
        log_entry.update(config["base_params"]) 
        image_params_log.append(log_entry)
    
    # 6. Simpan plot perbandingan untuk gambar ini
    plot_filepath = base_output_dir / f"comparison_{img_name}.png"
    utils.plot_comparison(comparison_plots, f"Filter Comparison for {img_name}", plot_filepath)

    return image_params_log


def run_all_filters(base_output_dir):
    """
    Fungsi untuk memproses tiap gambar.
    """
    print("--- 1. Menjalankan Modul Filtering---")
    imgs = utils.load_images()
    all_params_log = [] # List untuk menyimpan semua parameter di csv  

    for img_name, img in imgs.items():
        logs_from_image = process_one_image(img_name, img, base_output_dir)
        all_params_log.extend(logs_from_image) 

    csv_path = base_output_dir / "filter_parameters.csv"
    utils.save_params_to_csv(csv_path, all_params_log)
    
    print("\n--- Modul Filtering Selesai ---")


def main():
    output_dir = Path(__file__).resolve().parent
    run_all_filters(output_dir)

if __name__ == "__main__":
    main()