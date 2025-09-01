import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from tqdm import tqdm
import glob
import random

# === 配置区 ===
DATASETS = {
    'Digiface1M': './datasets/digiface1m_raw',         # 这里是解压后最顶层
    'SFHQ'      : './datasets/sfhq_subset',            # 你已经整理好的子集
    'CelebA'    : './datasets/celeba/img_align_celeba' # 原始对齐后那层
}
OVERLAY_DIR   = './datasets/overlays'
OUTPUT_SUFFIX = '_makeup'

# === Face Mesh 初始化 ===
mpfm = mp.solutions.face_mesh
face_mesh = mpfm.FaceMesh(static_image_mode=True,
                         max_num_faces=1,
                         refine_landmarks=True,
                         min_detection_confidence=0.5)

# === 关键点索引 ===
LEFT_EYE_IDX   = [33, 133]
RIGHT_EYE_IDX  = [362, 263]
LIPS_IDX       = list(range(61, 88))  # 上下唇区域

# === 加载多贴图素材 ===
def load_all(prefix):
    return [Image.open(p).convert("RGBA")
            for p in glob.glob(os.path.join(OVERLAY_DIR, f"{prefix}*.png"))]

glasses_list   = load_all('glasses')
lipstick_list  = load_all('lipstick')
eyeshadow_list = load_all('eyeshadow')

def apply_overlay(base, overlay, center, size, offset=(0,0)):
    """以 center 为中心贴 overlay，size=(w,h)，offset 微调"""
    w,h = size
    overlay = overlay.resize((w,h), Image.LANCZOS)
    x = center[0] - w//2 + offset[0]
    y = center[1] - h//2 + offset[1]
    layer = Image.new('RGBA', base.size, (0,0,0,0))
    layer.paste(overlay, (x,y), overlay)
    return Image.alpha_composite(base.convert("RGBA"), layer)

def process_folder(name, folder):
    out_folder = folder + OUTPUT_SUFFIX
    os.makedirs(out_folder, exist_ok=True)

    # 使用 os.walk 递归所有子目录，兼容 Digiface1M 的分子文件夹结构
    files = []
    for root, dirs, fnames in os.walk(folder):
        for f in fnames:
            if f.lower().endswith(('.jpg','jpeg','png')):
                files.append(os.path.join(root, f))

    if not files:
        print(f"⚠️ 目录 {folder} 下未找到任何图片！")
        return

    for img_path in tqdm(files, desc=f"Processing {name}"):
        # 读取并检测
        bgr = cv2.imread(img_path)
        if bgr is None: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        pil = Image.fromarray(rgb)

        # 如果检测到脸
        if res.multi_face_landmarks:
            pts = [(int(p.x*rgb.shape[1]), int(p.y*rgb.shape[0]))
                   for p in res.multi_face_landmarks[0].landmark]

            # 眼睛中心和大小
            le, re = pts[LEFT_EYE_IDX[0]], pts[RIGHT_EYE_IDX[1]]
            eye_ctr = ((le[0]+re[0])//2, (le[1]+re[1])//2)
            eye_w = int(np.hypot(re[0]-le[0], re[1]-le[1]) * 2.0)
            eye_h = int(eye_w * (glasses_list[0].height / glasses_list[0].width) * 0.5)

            # 随机贴眼镜
            if glasses_list and random.random()<0.7:
                pil = apply_overlay(pil, random.choice(glasses_list),
                                    eye_ctr, (eye_w, eye_h))

            # 随机贴眼影
            if eyeshadow_list and random.random()<0.5:
                w_sh = int(eye_w*1.2)
                h_sh = int(w_sh * (eyeshadow_list[0].height / eyeshadow_list[0].width) * 0.6)
                pil = apply_overlay(pil, random.choice(eyeshadow_list),
                                    eye_ctr, (w_sh, h_sh), offset=(0, int(h_sh*0.1)))

            # 随机贴口红
            lip_pts = [pts[i] for i in LIPS_IDX]
            xs, ys = zip(*lip_pts)
            lip_ctr = (int(np.mean(xs)), int(np.mean(ys)))
            lip_w = int((max(xs)-min(xs)) * 1.2)
            lip_h = int(lip_w * (lipstick_list[0].height / lipstick_list[0].width))
            if lipstick_list and random.random()<0.5:
                pil = apply_overlay(pil, random.choice(lipstick_list),
                                    lip_ctr, (lip_w, lip_h), offset=(0, int(lip_h*0.1)))

        # 构造保存路径：保留原目录层级
        rel = os.path.relpath(img_path, folder)
        save_path = os.path.join(out_folder, rel)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pil.convert("RGB").save(save_path)

if __name__ == "__main__":
    for name, path in DATASETS.items():
        process_folder(name, path)
    print("🎉 全部处理完毕，见 *_makeup 文件夹")
