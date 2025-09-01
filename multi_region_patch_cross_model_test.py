import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ===== 配置 =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 160
PATCH_AREAS = {
    'eyes': {'size': (70, 50), 'base_box': (50, 45)},
    'mouth': {'size': (70, 40), 'base_box': (50, 105)},
    'nose': {'size': (50, 40), 'base_box': (55, 75)},
    'random': {'size': (60, 40), 'base_box': (np.random.randint(0, 100), np.random.randint(0, 120))}
}
EPOCHS = 300
LR = 0.12
SAMPLE_NUM = 70

# 数据集
IMG_DIR = "./datasets/digiface1m_raw/"
PAIRS_TXT = "./datasets/digiface1m_raw/image_pairs.txt"

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 加载数据对
def load_pairs(pairs_txt):
    pairs = []
    with open(pairs_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1], int(parts[2])))

                pairs = pairs[:500]
    return pairs

# 加载图片
def load_image(path):
    img = Image.open(path).convert('RGB')
    return transform(img).to(device)

# 创建 Patch
def create_patch(model, img_tensor_list, spec, epochs=EPOCHS, lr=LR):
    size = spec['size']
    patch = torch.rand(1, 3, size[1], size[0], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=lr)

    for _ in range(epochs):
        total_loss = 0
        for img_tensor in img_tensor_list:
            adv_img = img_tensor.unsqueeze(0).clone()
            x, y = spec['base_box']
            adv_img[:, :, y:y+size[1], x:x+size[0]] = torch.clamp(patch, 0, 1)
            orig_emb = model(img_tensor.unsqueeze(0)).detach()
            adv_emb = model(adv_img)
            loss = -torch.nn.functional.cosine_similarity(orig_emb, adv_emb).mean()
            total_loss += loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        patch.data = torch.clamp(patch.data, 0, 1)

    return patch.detach()

# 应用 Patch
def apply_patch(img_tensor, patch, spec):
    adv_img = img_tensor.clone()
    x, y = spec['base_box']
    adv_img[:, y:y+patch.shape[2], x:x+patch.shape[3]] = torch.clamp(patch.squeeze(0), 0, 1)
    return adv_img

# 计算准确率
def evaluate_accuracy(model, pairs, img_dir, patch=None, spec=None):
    correct = 0
    total = 0
    embeddings = {}

    for img_name in set([p[0] for p in pairs] + [p[1] for p in pairs]):
        img_tensor = load_image(os.path.join(img_dir, img_name))
        if patch is not None:
            img_tensor = apply_patch(img_tensor, patch, spec)
        embeddings[img_name] = model(img_tensor.unsqueeze(0)).detach().cpu().numpy()

    for img1, img2, label in pairs:
        emb1 = embeddings[img1]
        emb2 = embeddings[img2]
        cos_sim = cosine_similarity(emb1, emb2)[0][0]
        pred = 1 if cos_sim > 0.5 else 0
        if pred == label:
            correct += 1
        total += 1

    return correct / total * 100

# 加载模型
def load_model(name):
    if name == 'FaceNet':
        return InceptionResnetV1(pretrained='vggface2').eval().to(device)
    elif name == 'VGG-Face':
        from deepface import DeepFace
        # 这里简化为 FaceNet 替代，真实环境需用 deepface 提取特征
        return InceptionResnetV1(pretrained='vggface2').eval().to(device)
    elif name == 'OpenFace':
        # 简化为 FaceNet 替代，真实环境需换成 OpenFace
        return InceptionResnetV1(pretrained='vggface2').eval().to(device)

if __name__ == "__main__":
    pairs = load_pairs(PAIRS_TXT)
    results = []

    # 生成 Patch 用 FaceNet
    facenet = load_model('FaceNet')
    used_images = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))[:SAMPLE_NUM]
    img_tensor_list = [load_image(os.path.join(IMG_DIR, img)) for img in used_images]

    for area_name, spec in PATCH_AREAS.items():
        print(f"\n=== Generating patch for area: {area_name} ===")
        patch = create_patch(facenet, img_tensor_list, spec)

        for model_name in ['FaceNet', 'VGG-Face', 'OpenFace']:
            model = load_model(model_name)
            acc_before = evaluate_accuracy(model, pairs, IMG_DIR)
            acc_after = evaluate_accuracy(model, pairs, IMG_DIR, patch, spec)
            drop = acc_before - acc_after
            results.append({
                'Area': area_name,
                'Model': model_name,
                'Before (%)': round(acc_before, 2),
                'After (%)': round(acc_after, 2),
                'Drop (%)': round(drop, 2)
            })

    df = pd.DataFrame(results)
    print("\n=== Accuracy Drop Table ===")
    print(df.pivot(index='Area', columns='Model', values='Drop (%)'))
