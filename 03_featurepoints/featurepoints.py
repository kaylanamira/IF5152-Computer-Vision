# Nama: Kayla Namira Mariadi
# NIM: 13522050
# Deskripsi: Runner untuk modul feature point detection (Harris, FAST, SIFT, & ORB).

import numpy as np
from pathlib import Path
from skimage import feature, img_as_float
import cv2

import utils  


def _detect_harris(img_gray, **kwargs):
    """ Helper untuk Harris (corner_harris + corner_peaks) """
    k = kwargs.get('k', 0.05)
    min_distance = kwargs.get('min_distance', 5)
    threshold_rel = kwargs.get('threshold_rel', 0.01)

    response_img = feature.corner_harris(img_gray, k=k)
    coords = feature.corner_peaks(response_img, 
                                  min_distance=min_distance, 
                                  threshold_rel=threshold_rel)
    responses_at_coords = response_img[coords[:, 0], coords[:, 1]]
    
    return coords, responses_at_coords

def _detect_orb(img_gray, **kwargs):
    # Ambil n_keypoints dari params
    n_keypoints = kwargs.get('n_keypoints', 200)
    
    detector = feature.ORB(n_keypoints=n_keypoints)
    detector.detect(img_gray)
    
    coords = detector.keypoints
    responses = detector.responses
    
    return coords, responses

def detect_sift(image, n_keypoints=500):
    """
    Mendeteksi fitur SIFT menggunakan OpenCV (cv2) dan mengembalikan koordinatnya.
    Mengembalikan None jika library tidak ter-install.
    """
    sift = cv2.SIFT_create(nfeatures=n_keypoints)

    if image.dtype in (np.float32, np.float64):
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    keypoints = sift.detect(img_uint8, None)

    if not keypoints:
        return np.empty((0, 2)), np.array([]) 

    coords_xy = np.array([kp.pt for kp in keypoints])

    # coords_xy = np.array([kp.pt for kp in keypoints])
    coords_rc = coords_xy[:, ::-1]
    responses = np.array([kp.response for kp in keypoints])
    
    return coords_rc, responses

def _detect_fast(img_gray, **kwargs):
    # n = kwargs.get('n', 9)
    # threshold = kwargs.get('threshold', 0.12)
    # resp = feature.corner_fast(img_gray, n=n, threshold=threshold)
    # coords = np.column_stack(np.nonzero(resp))
    # responses = np.ones((coords.shape[0],), dtype=np.float32)
    # return coords[:, ::-1], responses
    if cv2 is None: return None

    threshold = kwargs.get('threshold', 10)

    # Buat detektor FAST dengan Non-Maximum Suppression AKTIF (default)
    fast_detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
    
    img_uint8 = (img_gray * 255).astype(np.uint8)
    keypoints = fast_detector.detect(img_uint8, None)

    if not keypoints:
        return np.empty((0, 2)), np.array([]) 

    coords_xy = np.array([kp.pt for kp in keypoints])
    coords_rc = coords_xy[:, ::-1]
    responses = np.array([kp.response for kp in keypoints])
    
    return coords_rc, responses


FEATURE_CONFIG = [
    {
        "name": "harris_mindist5_k0.05",
        "function": _detect_harris,
        "params": {"min_distance": 5, "k": 0.05, "threshold_rel": 0.01},
        "notes": "Harris: Jarak minimum 5px"
    },
    {
        "name": "harris_mindist20_k0.05",
        "function": _detect_harris,
        "params": {"min_distance": 20, "k": 0.05, "threshold_rel": 0.01},
        "notes": "Harris: Jarak minimum 20px"
    },
    {
        "name": "fast_n9_thresh10",
        "function": _detect_fast,
        "params": {"n": 9, "threshold": 10},
        "notes": "FAST-9: Threshold standar"
    },
    {
        "name": "fast_n9_thresh30",
        "function": _detect_fast,
        "params": {"n": 9, "threshold": 30},
        "notes": "FAST-9: Threshold tinggi"
    },
    {
        "name": "orb_200_features",
        "function": _detect_orb, 
        "params": {"n_keypoints": 200},
        "notes": "ORB: Target 200 keypoints"
    },
    {
        "name": "orb_500_features",
        "function": _detect_orb, 
        "params": {"n_keypoints": 500},
        "notes": "ORB: Target 500 keypoints"
    },
    {
        "name": "sift_500_features",
        "function": detect_sift,
        "params": {"n_keypoints": 500},
        "notes": "SIFT: Target 500 keypoints"
    }
]


def process_one_image(img_name, img, base_output_dir):
    """
    Memproses satu gambar dengan semua metode feature detection di config.
    """
    print(f"\nMemproses image: {img_name}")
    
    img_gray = utils.to_gray(img_as_float(img))
    
    img_output_dir = base_output_dir / img_name
    
    comparison_plots = {"Original (Grayscale)": img_gray}
    utils.save_img(img_output_dir / f"{img_name}_original_gray.png", img_gray)
    image_params_log = []

    for config in FEATURE_CONFIG:
        print(f"  Menerapkan: {config['name']}...")
        func = config["function"]
        params = config["params"].copy()
        
        # Harris dan ORB mengembalikan (coords, responses)
        if func in (_detect_harris, _detect_orb, _detect_fast):
            coords, responses = func(img_gray, **params)
            num_features = len(coords)
            mean_response = responses.mean() if responses.size > 0 else 0
        
        # elif func == feature.corner_fast:
        #     coords = func(img_gray, **params)
        #     num_features = len(coords)
        #     mean_response = 'N/A' 

        # elif func == _detect_fast:
        #     img_gray_u8 = (img_gray * 255).astype(np.uint8)
        #     coords = func(img_gray_u8, **params)
        #     num_features = len(coords)
        #     mean_response = 'N/A'
        
        elif func == detect_sift: 
            result = func(img_gray, **params)
            
            if result is None:
                print("  SKIPPING SIFT (OpenCV tidak terinstall)")
                continue 
            
            coords, responses = result
            num_features = len(coords)
            mean_response = responses.mean() if responses.size > 0 else 0
        else:
            print(f"  SKIPPING: Fungsi {func} tidak dikenali")
            continue
        
        marked_image = utils.create_marked_image(img, coords)
        
        output_filename = img_output_dir / f"{img_name}_{config['name']}.png"
        utils.save_img(output_filename, marked_image)

        gray_marked_image = utils.create_marked_image(img_gray, coords)
        comparison_plots[config['name']] = gray_marked_image

        log_entry = {
            "image": img_name, 
            "method": config['name'], 
            "num_features": num_features,
            "mean_response": f"{mean_response:.4f}" if isinstance(mean_response, (int, float)) else "N/A",
            "notes": config['notes']
        }
        log_entry.update(config["params"])
        image_params_log.append(log_entry)
    
    plot_filepath = base_output_dir / f"comparison_{img_name}.png"
    utils.plot_comparison(
        comparison_plots, 
        f"Feature Detection Comparison for {img_name}", 
        plot_filepath
    )

    return image_params_log


def run_all_features(base_output_dir):
    """
    Fungsi utama yang me-loop semua gambar dan memanggil prosesor.
    """
    print("--- 3. Menjalankan Modul Feature Detection ---")
    imgs = utils.load_images()
    all_params_log = []

    # imgs.pop('checkerboard', None) 

    for img_name, img in imgs.items():
        logs_from_image = process_one_image(img_name, img, base_output_dir)
        all_params_log.extend(logs_from_image)

    csv_path = base_output_dir / "feature_parameters.csv"
    utils.save_params_to_csv(csv_path, all_params_log)
    
    print("\n--- Modul Feature Detection Selesai ---")


def main():
    output_dir = Path(__file__).resolve().parent
    run_all_features(output_dir)

if __name__ == "__main__":
    main()