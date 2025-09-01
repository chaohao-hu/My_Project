import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

# ===== 数据集路径（兼容 image_pairs.txt 用法）=====
DATASETS = {
    'Digiface1M': {
        'img_dir': './datasets/digiface1m_raw/',
        'pairs_txt': './datasets/digiface1m_raw/image_pairs.txt'
    },
    'SFHQ': {
        'img_dir': './datasets/sfhq_subset/',
        'pairs_txt': './datasets/sfhq_subset/image_pairs.txt'
    },
    'CelebA': {
        'img_dir': './datasets/celeba/img_align_celeba/',
        'pairs_txt': './datasets/celeba/img_align_celeba/image_pairs.txt'
    }
}
OUTPUT_SUFFIX = '_illum_blur_noise'

def load_used_images(pairs_txt):
    used = set()
    with open(pairs_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                used.add(parts[0])
                used.add(parts[1])
    return used

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def random_brightness_contrast(img, max_bright=0.4, max_contrast=0.4):
    alpha = 1.0 + random.uniform(-max_contrast, max_contrast)
    beta = int(255 * random.uniform(-max_bright, max_bright))
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def add_noise(img, noise_level=0.09):
    noise = np.random.randn(*img.shape) * (noise_level * 255)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def random_blur(img):
    kernel_size = random.choice([5, 7, 9])
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def random_jpeg_artifact(img):
    # 模拟JPEG压缩伪影
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 65)]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)
    return img

def full_illum_blur_noise(img):
    # 1. 随机gamma（光照）
    gamma = random.uniform(0.55, 1.85)
    img = adjust_gamma(img, gamma)
    # 2. 随机亮度/对比度
    img = random_brightness_contrast(img, max_bright=0.38, max_contrast=0.36)
    # 3. 随机模糊
    if random.random() < 0.8:
        img = random_blur(img)
    # 4. 随机噪声
    img = add_noise(img, noise_level=random.uniform(0.06, 0.14))
    # 5. 随机jpeg压缩
    if random.random() < 0.7:
        img = random_jpeg_artifact(img)
    return img

def process_folder(name, img_dir, pairs_txt):
    out_folder = img_dir.rstrip('/\\') + OUTPUT_SUFFIX
    used_images = load_used_images(pairs_txt)
    for rel_img_path in tqdm(used_images, desc=f"{name} Illum+Blur+Noise"):
        in_path = os.path.join(img_dir, rel_img_path)
        out_path = os.path.join(out_folder, rel_img_path)
        if not os.path.exists(in_path):
            print(f"[SKIP] Not found: {in_path}")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            img = cv2.imread(in_path)
            if img is None:
                print(f"[ERROR] Fail to read {in_path}")
                continue
            img = full_illum_blur_noise(img)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_pil.save(out_path)
        except Exception as e:
            print(f"[ERROR] {rel_img_path}: {e}")

if __name__ == "__main__":
    for name, ds_info in DATASETS.items():
        process_folder(name, ds_info['img_dir'], ds_info['pairs_txt'])
    print("\n🎉 Illumination+Blur+Noise 批量扰动全部完成！")
