import os
from PIL import Image, ImageEnhance
from tqdm import tqdm

# ============ 配置区域 ============
TARGET_SIZE = (112, 112)
TARGET_BRIGHTNESS = 82.42 / 127.5  # 归一化亮度
DATASETS = {
    "CelebA": {
        "img_dir": "./datasets/celeba/img_align_celeba/",
        "pairs_txt": "./datasets/celeba/img_align_celeba/image_pairs.txt",
        "output_dir": "./datasets/celeba/celeba_degraded/"
    },
    "SFHQ": {
        "img_dir": "./datasets/sfhq_subset/",
        "pairs_txt": "./datasets/sfhq_subset/image_pairs.txt",
        "output_dir": "./datasets/sfhq_degraded/"
    }
}

# ============ 辅助函数 ============

def get_brightness(pil_img):
    grayscale = pil_img.convert('L')
    hist = grayscale.histogram()
    pixels = sum(hist)
    brightness = sum(i * hist[i] for i in range(256)) / pixels
    return brightness / 255.0

def degrade_image(img_path, out_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        current_brightness = get_brightness(img)
        factor = TARGET_BRIGHTNESS / current_brightness if current_brightness > 0 else 1.0
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")

def load_pairs(pairs_txt_path):
    used = set()
    with open(pairs_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                used.add(parts[0])
                used.add(parts[1])
    return list(used)

def process_dataset(name, img_dir, pairs_txt, output_dir):
    print(f"\n=== Processing dataset: {name} ===")
    image_list = load_pairs(pairs_txt)
    for rel_path in tqdm(image_list, desc=name):
        in_path = os.path.join(img_dir, rel_path)
        out_path = os.path.join(output_dir, rel_path)
        if not os.path.exists(in_path):
            print(f"[SKIP] Not found: {in_path}")
            continue
        degrade_image(in_path, out_path)
    print(f"✅ Finished processing {name}. Output saved to: {output_dir}")

# ============ 主程序 ============
if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        process_dataset(name, cfg["img_dir"], cfg["pairs_txt"], cfg["output_dir"])
