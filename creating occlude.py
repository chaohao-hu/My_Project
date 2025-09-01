import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from tqdm import tqdm

DATASETS = {
    'Digiface1M': './datasets/digiface1m_raw',
    'SFHQ': './datasets/sfhq_subset',
    'CelebA': './datasets/celeba/img_align_celeba'
}
OUTPUT_SUFFIX = '_occlude'

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5)

MOUTH_IDX = list(range(61, 88))
NOSE_IDX = list(range(168, 178))
FOREHEAD_OFFSET = (-10, -60)

def apply_occlusion(img_pil, landmarks, region='mouth'):
    draw = img_pil.copy()
    draw_np = np.array(draw)

    if region == 'mouth':
        pts = np.array([(int(p.x * img_pil.width), int(p.y * img_pil.height))
                        for i, p in enumerate(landmarks) if i in MOUTH_IDX])
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

        if res.multi_face_landmarks:
            landmarks = res.multi_face_landmarks[0].landmark
            out_img = apply_occlusion(pil, landmarks, region='mouth')  # 可换成 'nose' 或 'forehead'
        else:
            out_img = pil

        save_path = os.path.join(out_folder, img_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_img.convert("RGB").save(save_path)

if __name__ == "__main__":
    for name, path in DATASETS.items():
        process_folder(name, path)
    print("🎯 遮挡处理完成，仅对 image_pairs.txt 中图片生效！")
