import os
import random

# 设置解压后的 Digiface 路径
digiface_path = r"./datasets/digiface1m_raw"  # 按实际修改
output_file = os.path.join(digiface_path, "image_pairs.txt")

# 获取所有身份文件夹（数字命名）
identities = [name for name in os.listdir(digiface_path) if os.path.isdir(os.path.join(digiface_path, name))]
identities = sorted(identities)
print(f"Found {len(identities)} identities.")

pairs = []

# ============ SAME PERSON pairs ============
for identity in random.sample(identities, 1000):  # 抽1000个身份
    id_path = os.path.join(digiface_path, identity)
    images = sorted(os.listdir(id_path))
    if len(images) < 2:
        continue
    img1, img2 = random.sample(images, 2)
    pairs.append(f"{identity}/{img1} {identity}/{img2} 1")

# ============ DIFFERENT PERSON pairs ============
for _ in range(1000):
    id1, id2 = random.sample(identities, 2)
    img1_list = sorted(os.listdir(os.path.join(digiface_path, id1)))
    img2_list = sorted(os.listdir(os.path.join(digiface_path, id2)))
    if not img1_list or not img2_list:
        continue
    img1 = random.choice(img1_list)
    img2 = random.choice(img2_list)
    pairs.append(f"{id1}/{img1} {id2}/{img2} 0")

# 保存为文件
with open(output_file, "w") as f:
    f.write("\n".join(pairs))

print(f"✅ image_pairs.txt 生成完成，共 {len(pairs)} 对比图像对")
