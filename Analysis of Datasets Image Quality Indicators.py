import os
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# 图像分析函数
def analyze_images(image_paths):
    resolutions = []
    brightness = []
    contrast = []
    sharpness = []
    file_sizes = []

    for img_path in tqdm(image_paths, desc="Analyzing"):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                arr = np.array(img).astype(np.float32)

                # 分析项
                resolutions.append(w * h)
                brightness.append(np.mean(arr))
                contrast.append(np.std(arr))
                sharpness.append(np.std(np.gradient(arr)[0]))  # 粗略sharpness
                file_sizes.append(os.path.getsize(img_path) / 1024)
        except Exception as e:
            print(f"[SKIP] Cannot process {img_path}: {e}")
            continue

    return {
        'resolution': (np.mean(resolutions), np.std(resolutions)),
        'brightness': (np.mean(brightness), np.std(brightness)),
        'contrast': (np.mean(contrast), np.std(contrast)),
        'sharpness': (np.mean(sharpness), np.std(sharpness)),
        'filesize_kb': (np.mean(file_sizes), np.std(file_sizes))
    }

# 从 pairs.txt 中读取图像路径集合
def load_image_paths_from_pairs(pairs_txt, base_dir):
    used = set()
    with open(pairs_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                used.add(os.path.join(base_dir, parts[0]))
                used.add(os.path.join(base_dir, parts[1]))
    return list(used)

# 数据集配置
DATASETS = {
    "Digiface1M": {
        "img_dir": "./datasets/digiface1m_raw/",
        "pairs_txt": "./datasets/digiface1m_raw/image_pairs.txt"
    },
    "SFHQ": {
        "img_dir": "./datasets/sfhq_subset/",
        "pairs_txt": "./datasets/sfhq_subset/image_pairs.txt"
    },
    "CelebA": {
        "img_dir": "./datasets/celeba/img_align_celeba/",
        "pairs_txt": "./datasets/celeba/img_align_celeba/image_pairs.txt"
    }
}

# 主程序
if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        print(f"\n=== Analyzing dataset: {name} ===")
        if not os.path.exists(cfg["pairs_txt"]):
            print(f"[SKIP] pairs.txt not found: {cfg['pairs_txt']}")
            continue

        image_paths = load_image_paths_from_pairs(cfg["pairs_txt"], cfg["img_dir"])
        if not image_paths:
            print("❌ No valid image paths extracted.")
            continue

        stats = analyze_images(image_paths)  # 分析所有图像
        print(f"✅ Analyzed {min(100, len(image_paths))} images from {name}")
        for key, (mean, std) in stats.items():
            print(f"- {key}: mean = {mean:.2f}, std = {std:.2f}")
