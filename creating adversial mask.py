import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

# ===== 配置区 =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 160
EPOCHS = 600
SAMPLE_NUM = 170
MODEL_NAME = 'Facenet'
OUT_ROOT = "./datasets"

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

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 多 patch 配置（可自定义大小和位置）
PATCH_SPECS = [
    {'size': (70, 50),  'base_box': (50, 45)},   # 眼睛区域
    {'size': (70, 40),  'base_box': (50, 105)}   # 嘴巴区域
]
RANDOM_SHIFT = 8  # patch位置抖动幅度（像素）

def load_facenet():
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return model

def load_used_images(pairs_txt):
    used = set()
    with open(pairs_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                used.add(parts[0])
                used.add(parts[1])
    return used

# -------- 多patch联合优化 --------
def create_multi_adv_patch(model, img_tensor_list, patch_specs, epochs=300, lr=0.12):
    patches = []
    for spec in patch_specs:
        size = spec['size']
        patch = torch.rand(1, 3, size[1], size[0], device=device, requires_grad=True)
        patches.append(patch)
    optimizer = torch.optim.Adam(patches, lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for img_tensor in img_tensor_list:
            adv_img = img_tensor.unsqueeze(0).clone()
            # 每个 patch 随机偏移位置
            for patch, spec in zip(patches, patch_specs):
                base_x, base_y = spec['base_box']
                shift_x = np.random.randint(-RANDOM_SHIFT, RANDOM_SHIFT+1)
                shift_y = np.random.randint(-RANDOM_SHIFT, RANDOM_SHIFT+1)
                x = np.clip(base_x + shift_x, 0, IMG_SIZE - patch.shape[3])
                y = np.clip(base_y + shift_y, 0, IMG_SIZE - patch.shape[2])
                adv_img[:, :, y:y+patch.shape[2], x:x+patch.shape[3]] = torch.clamp(patch, 0, 1)
            orig_emb = model(img_tensor.unsqueeze(0)).detach()
            adv_emb = model(adv_img)
            loss = -torch.nn.functional.cosine_similarity(orig_emb, adv_emb).mean()
            total_loss += loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        for patch in patches:
            patch.data = torch.clamp(patch.data, 0, 1)
    # detach后返回
    return [p.detach() for p in patches]

# -------- 应用多个patch --------
def apply_multi_patch(img_path, patches, patch_specs):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    adv_img = img_tensor.clone()
    for patch, spec in zip(patches, patch_specs):
        x, y = spec['base_box']
        adv_img[:, :, y:y+patch.shape[2], x:x+patch.shape[3]] = torch.clamp(patch, 0, 1)
    adv_img = adv_img.squeeze(0).detach().cpu()
    adv_img = (adv_img * 0.5 + 0.5) * 255
    adv_img = adv_img.permute(1,2,0).numpy().astype(np.uint8)
    adv_pil = Image.fromarray(adv_img)
    return adv_pil

if __name__ == "__main__":
    print(f"\n======= Processing Model: {MODEL_NAME} (Multi-patch) =======")
    model = load_facenet()

    # 1. 采样图片用于patch优化
    img_dir = DATASETS['Digiface1M']['img_dir']
    pairs_txt = DATASETS['Digiface1M']['pairs_txt']
    used_images = list(load_used_images(pairs_txt))
    np.random.shuffle(used_images)
    img_tensor_list = []
    cnt = 0
    for rel_img_path in used_images:
        in_path = os.path.join(img_dir, rel_img_path)
        if os.path.exists(in_path):
            img = Image.open(in_path).convert('RGB')
            img_tensor = transform(img)
            img_tensor_list.append(img_tensor.to(device))
            cnt += 1
            if cnt >= SAMPLE_NUM:
                break

    # 2. 优化多个Patch
    print("⏳ 正在优化多个对抗Patch...")
    adv_patches = create_multi_adv_patch(model, img_tensor_list, PATCH_SPECS, epochs=EPOCHS)
    print("✅ 多patch优化完成，开始批量处理！")

    # 3. 批量应用Patch
    for ds_name, ds_info in DATASETS.items():
        print(f"\n>>> Dataset: {ds_name}")
        img_dir = ds_info["img_dir"]
        pairs_txt = ds_info["pairs_txt"]
        out_dir = os.path.join(OUT_ROOT, f"{os.path.basename(img_dir.rstrip('/'))}_adv_{MODEL_NAME}")
        for rel_img_path in tqdm(load_used_images(pairs_txt), desc=f"{ds_name} ({MODEL_NAME})"):
            in_path = os.path.join(img_dir, rel_img_path)
            out_path = os.path.join(out_dir, rel_img_path)
            if not os.path.exists(in_path):
                print(f"[SKIP] Not found: {in_path}")
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                adv_img = apply_multi_patch(in_path, adv_patches, PATCH_SPECS)
                adv_img.save(out_path)
            except Exception as e:
                print(f"[ERROR] {rel_img_path}: {e}")
        print(f"✅ Done: {out_dir}")

    print("\n🎉 ALL multi-patch adversarial images generated!")
