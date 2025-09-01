import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from tqdm import tqdm
import glob
import random

# ======== 基础参数区 ========
DATASETS = {
    'Digiface1M': './datasets/digiface1m_raw',
    'SFHQ': './datasets/sfhq_subset',
    'CelebA': './datasets/celeba/img_align_celeba'
}
OVERLAY_DIR = './datasets/overlays'
OUTPUT_SUFFIX = '_makeup_occlude'

# ======== Makeup 素材加载 ========
def load_all(prefix):
    return [Image.open(p).convert("RGBA")
            for p in glob.glob(os.path.join(OVERLAY_DIR, f"{prefix}*.png"))]
glasses_list   = load_all('glasses')
lipstick_list  = load_all('lipstick')
eyeshadow_list = load_all('eyeshadow')

# ======== mediapipe 初始化 ========
mpfm = mp.solutions.face_mesh
face_mesh = mpfm.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

LEFT_EYE_IDX  = [33, 133]
RIGHT_EYE_IDX = [362, 263]
LIPS_IDX      = list(range(61, 88))  # 上下唇
NOSE_IDX      = list(range(168, 178))
FOREHEAD_OFFSET = (-10, -60)  # 可以微调

def apply_overlay(base, overlay, center, size, offset=(0,0)):
    w, h = size
    overlay = overlay.resize((w, h), Image.LANCZOS)
    x = center[0] - w // 2 + offset[0]
    y = center[1] - h // 2 + offset[1]
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    layer.paste(overlay, (x, y), overlay)
    return Image.alpha_composite(base.convert("RGBA"), layer)

def apply_occlusion(img_pil, landmarks, region='mouth'):
    draw = img_pil.copy()
    draw_np = np.array(draw)

    if region == 'mouth':
        pts = np.array([(int(p.x * img_pil.width), int(p.y * img_pil.height))
                        for i, p in enumerate(landmarks) if i in LIPS_IDX])
    elif region == 'nose':
        pts = np.array([(int(p.x * img_pil.width), int(p.y * img_pil.height))
                        for i, p in enumerate(landmarks) if i in NOSE_IDX])
    elif region == 'forehead':
        nose = np.mean([(p.x, p.y) for i, p in enumerate(landmarks) if i in NOSE_IDX], axis=0)
        x = int(nose[0] * img_pil.width + FOREHEAD_OFFSET[0])
        y = int(nose[1] * img_pil.height + FOREHEAD_OFFSET[1])
        w, h = 40, 30
        cv2.rectangle(draw_np, (x, y), (x + w, y + h), (128, 128, 128), -1)
        return Image.fromarray(draw_np)
    else:
        pts = []

    if len(pts) > 0:
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(draw_np, (x, y), (x + w, y + h), (128, 128, 128), -1)
    return Image.fromarray(draw_np)

def load_used_images(dataset_path):
    txt_path = os.path.join(dataset_path, 'image_pairs.txt')
    used = set()
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                used.add(parts[0])
                used.add(parts[1])
    return used

def process_folder(name, folder):
    out_folder = folder + OUTPUT_SUFFIX
    os.makedirs(out_folder, exist_ok=True)

    used_images = load_used_images(folder)

    for img_name in tqdm(used_images, desc=f"Processing {name}"):
        img_path = os.path.join(folder, img_name)
        if not os.path.exists(img_path):
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        res = face_mesh.process(rgb)

        # ==== Step1: Makeup ====
        if res.multi_face_landmarks:
            pts = [(int(p.x*rgb.shape[1]), int(p.y*rgb.shape[0]))
                   for p in res.multi_face_landmarks[0].landmark]

            # 眼睛中心和大小
            le, re = pts[LEFT_EYE_IDX[0]], pts[RIGHT_EYE_IDX[1]]
            eye_ctr = ((le[0]+re[0])//2, (le[1]+re[1])//2)
            eye_w = int(np.hypot(re[0]-le[0], re[1]-le[1]) * 2.0)
            eye_h = int(eye_w * (glasses_list[0].height / glasses_list[0].width) * 0.5)
            # 眼镜
            if glasses_list and random.random() < 0.7:
                pil = apply_overlay(pil, random.choice(glasses_list), eye_ctr, (eye_w, eye_h))
            # 眼影
            if eyeshadow_list and random.random() < 0.5:
                w_sh = int(eye_w * 1.2)
                h_sh = int(w_sh * (eyeshadow_list[0].height / eyeshadow_list[0].width) * 0.6)
                pil = apply_overlay(pil, random.choice(eyeshadow_list), eye_ctr, (w_sh, h_sh), offset=(0, int(h_sh*0.1)))
            # 口红
            lip_pts = [pts[i] for i in LIPS_IDX]
            xs, ys = zip(*lip_pts)
            lip_ctr = (int(np.mean(xs)), int(np.mean(ys)))
            lip_w = int((max(xs)-min(xs)) * 1.2)
            lip_h = int(lip_w * (lipstick_list[0].height / lipstick_list[0].width))
            if lipstick_list and random.random() < 0.5:
                pil = apply_overlay(pil, random.choice(lipstick_list), lip_ctr, (lip_w, lip_h), offset=(0, int(lip_h*0.1)))

            # ==== Step2: Occlusion ====
            landmarks = res.multi_face_landmarks[0].landmark
            # 可更换 region='nose' or 'forehead'
            pil = apply_occlusion(pil, landmarks, region='mouth')
        # 无脸则直接保存
        save_path = os.path.join(out_folder, img_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pil.convert("RGB").save(save_path)

if __name__ == "__main__":
    for name, path in DATASETS.items():
        process_folder(name, path)
    print("🎉 Makeup+遮挡全部处理完毕，见 *_makeup_occlude 文件夹")
