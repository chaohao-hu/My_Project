import os
import random
import pandas as pd

# ========== 配置区域 ==========
celeba_dir     = './datasets/celeba'
img_dir        = os.path.join(celeba_dir, 'img_align_celeba')
partition_file = os.path.join(celeba_dir, 'list_eval_partition.csv')
output_file    = os.path.join(celeba_dir, 'image_pairs.txt')

total_pairs = 500  # 总对数
# ==============================

# 1. 读取 CSV（逗号分隔，带表头）
df = pd.read_csv(partition_file)

# 确认列名
if 'partition' not in df.columns or 'image_id' not in df.columns:
    # 有些版本表头可能不一样，尝试以下
    df = pd.read_csv(partition_file, header=None, names=['image_id', 'partition'])

# 2. 提取测试集图片名（partition==2）
test_imgs = df[df['partition'] == 2]['image_id'].astype(str).tolist()

# 3. 过滤本地不存在的文件
test_imgs = [img for img in test_imgs if os.path.isfile(os.path.join(img_dir, img))]

if not test_imgs:
    raise RuntimeError(f"No images found in test set at {img_dir}")

# 4. 生成正负样本对
half = total_pairs // 2
pairs = []

# 正样本：图像自己 vs 自己
for _ in range(half):
    img = random.choice(test_imgs)
    pairs.append((img, img, 1))

# 负样本：两张不同图
for _ in range(half):
    img1, img2 = random.sample(test_imgs, 2)
    pairs.append((img1, img2, 0))

random.shuffle(pairs)

# 5. 写入文件
with open(output_file, 'w') as f:
    for img1, img2, label in pairs:
        f.write(f"img_align_celeba/{img1} img_align_celeba/{img2} {label}\n")

print(f"✅ 配对完成：共 {len(pairs)} 对 → {output_file}")
