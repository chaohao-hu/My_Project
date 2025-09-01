import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ========= 数据集路径配置 =========
DATASETS = {
    "DigiFace1M": {
        "img_dir": "./datasets/digiface1m_raw/",
        "pairs_txt": "./datasets/digiface1m_raw/image_pairs.txt",
        "acc": 0.7145  # Facenet makeup (71.45)
    },
    "SFHQ": {
        "img_dir": "./datasets/sfhq_subset/",
        "pairs_txt": "./datasets/sfhq_subset/image_pairs.txt",
        "acc": 0.999  # Facenet makeup (99.9)
    },
    "CelebA": {
        "img_dir": "./datasets/celeba/img_align_celebA/",
        "pairs_txt": "./datasets/celeba/img_align_celebA/image_pairs.txt",
        "acc": 1.0  # Facenet makeup (100)
    }
}

# ========= 加载 CLIP 模型 =========
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ========= 从配对文件中读取图像路径 =========
def read_pairs_images(pairs_txt, img_dir):
    image_paths = set()
    with open(pairs_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_paths.add(os.path.join(img_dir, parts[0]))
                image_paths.add(os.path.join(img_dir, parts[1]))
    return list(image_paths)

# ========= 提取所有图像的风格特征 =========
def extract_clip_features(image_paths):
    features = []
    for path in tqdm(image_paths[:800], desc="Extracting CLIP features"):  # 限制前200张加速
        if not os.path.exists(path):
            continue
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image)
                feat /= feat.norm(dim=-1, keepdim=True)
                features.append(feat.squeeze(0).cpu())
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")
    if features:
        return torch.stack(features)
    else:
        return None

# ========= 主流程 =========
def analyze():
    dataset_vectors = {}
    for name, info in DATASETS.items():
        print(f"\n📦 处理数据集: {name}")
        image_paths = read_pairs_images(info["pairs_txt"], info["img_dir"])
        feats = extract_clip_features(image_paths)
        if feats is not None:
            dataset_vectors[name] = feats.mean(dim=0)
        else:
            print(f"⚠️ No features extracted for {name}")

    # ========= 计算 Cosine 距离（1 - cos 相似度） =========
    ref_center = (dataset_vectors["CelebA"] + dataset_vectors["SFHQ"]) / 2
    ref_center /= ref_center.norm()  # 归一化

    distances = {}
    for name in DATASETS:
        vec = dataset_vectors[name]
        vec /= vec.norm()
        cosine_sim = F.cosine_similarity(vec.unsqueeze(0), ref_center.unsqueeze(0)).item()
        cosine_dist = 1 - cosine_sim  # 越大越不相似，最大为 2（方向完全相反）
        distances[name] = cosine_dist

    # ========= 输出表格 =========
    result_df = pd.DataFrame({
        "Dataset": list(DATASETS.keys()),
        "Accuracy": [DATASETS[k]["acc"] for k in DATASETS],
        "Cosine_Distance": [distances[k] for k in DATASETS]
    })

    print("\n📊 余弦风格距离与准确率对比：\n")
    print(result_df)

    # ========= 可视化 =========
    plt.figure(figsize=(8, 5))
    plt.scatter(result_df["Cosine_Distance"], result_df["Accuracy"], s=100, c='blue')
    for i in range(len(result_df)):
        plt.text(result_df["Cosine_Distance"][i]+0.005, result_df["Accuracy"][i]-0.01, result_df["Dataset"][i])
    plt.xlabel("Cosine Distance to Normal Center (0 = identical, 2 = opposite)")
    plt.ylabel("Recognition Accuracy")
    plt.title("🎯 Cosine Distance vs Recognition Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cosine_distance_vs_accuracy.png")
    plt.show()

# ========= 启动 =========
if __name__ == "__main__":
    analyze()
