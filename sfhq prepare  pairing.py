import os, shutil, random

src_dir = './datasets/sfhq_full/images'
dst_dir = './datasets/sfhq_subset'
os.makedirs(dst_dir, exist_ok=True)

sample_size = 2000

# 遍历所有子目录获取图像路径
all_imgs = []
for root, _, files in os.walk(src_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_imgs.append(os.path.join(root, file))

print(f"📂 共找到图像文件 {len(all_imgs)} 张")
if len(all_imgs) == 0:
    raise RuntimeError("❌ 没找到图像文件，请确认 images 文件夹结构！")

# 随机采样图像
sample_size = min(sample_size, len(all_imgs))
sample_imgs = sorted(random.sample(all_imgs, sample_size))

# 拷贝图像到子集文件夹
for path in sample_imgs:
    filename = os.path.basename(path)
    shutil.copy(path, os.path.join(dst_dir, filename))

print(f"✅ 抽取完成，共 {sample_size} 张图像复制到 sfhq_subset")

# 构造图像对（5000 same + 5000 different）
pairs = []

for img in sample_imgs[:min(1000, len(sample_imgs))]:
    filename = os.path.basename(img)
    pairs.append(f"{filename} {filename} 1")

for _ in range(min(1000, len(sample_imgs) // 2)):
    img1, img2 = random.sample(sample_imgs, 2)
    pairs.append(f"{os.path.basename(img1)} {os.path.basename(img2)} 0")

with open(os.path.join(dst_dir, 'image_pairs.txt'), 'w') as f:
    f.write('\n'.join(pairs))

print(f"✅ image_pairs.txt 已生成，共 {len(pairs)} 条图像对")
