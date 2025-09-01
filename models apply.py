from deepface import DeepFace
import os
import numpy as np

# —— STEP 0: 模型 & 数据集 配置 ——
models = ['Facenet', 'OpenFace', 'VGG-Face']
#'OpenFace', 'VGG-Face'
datasets = {
    'Digiface1M': './datasets/digiface1m_raw/',
    # 'Digiface1M_makeup': './datasets/digiface1m_raw_makeup/',
    # 'Digiface1M_occlude': './datasets/digiface1m_raw_occlude/',
    # 'Digiface1M_adv_Facenet': './datasets/digiface1m_raw_adv_Facenet/',
    # 'Digiface1M_makeup_occlude' : './datasets/digiface1m_raw_makeup_occlude/',
    # 'Digiface1M_illum_blur_noise' : './datasets/digiface1m_raw_illum_blur_noise/',

    # 'SFHQ'       : './datasets/sfhq_subset/',
    # 'SFHQ_makeup'      : './datasets/sfhq_subset_makeup/',
    # 'SFHQ_occlude'      : './datasets/sfhq_subset_occlude/',
    # 'SFHQ_adv_Facenet'      : './datasets/sfhq_subset_adv_Facenet/',
    # 'SFHQ_makeup_occlude'       : './datasets/sfhq_subset_makeup_occlude/',
    # 'SFHQ_illum_blur_noise'       : './datasets/sfhq_subset_illum_blur_noise/',
    'SFHQ_degraded'      :'./datasets/sfhq_degraded/',

    # 'CelebA'     : './datasets/celeba/img_align_celeba/',
    # 'CelebA_makeup'    : './datasets/celeba/img_align_celeba_makeup/',
    # 'CelebA_occlude'    : './datasets/celeba/img_align_celeba_occlude/',
    # 'CelebA_adv_Facenet'    : './datasets/celeba/img_align_celeba_adv_Facenet/',
    # 'CelebA_makeup_occlude'     : './datasets/celeba/img_align_celeba_makeup_occlude/',
    # 'CelebA_illum_blur_noise'     : './datasets/celeba/img_align_celeba_illum_blur_noise/',
    'CelebA_degraded'    :'./datasets/celeba/celeba_degraded/'
}

# 最终结果容器
results = {}

# —— STEP 1: 对每个模型 & 数据集 循环评估 ——
for model in models:
    print(f"\n=== Evaluating model: {model} ===")
    results[model] = {}

    for ds_name, ds_root in datasets.items():
        print(f"\n>> Dataset: {ds_name}")

        # 1.1 载入 image_pairs.txt
        pairs_path = os.path.join(ds_root, 'image_pairs.txt')
        with open(pairs_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # 1.2 计算所有距离和标签
        distances = []
        labels    = []
        for line in lines:
            img1, img2, lbl = line.split()
            lbl = int(lbl)

            p1 = os.path.join(ds_root, img1)
            p2 = os.path.join(ds_root, img2)

            try:
                out = DeepFace.verify(
                    p1, p2,
                    model_name=model,
                    distance_metric='cosine',
                    enforce_detection=False
                )
                distances.append(out['distance'])
                labels.append(lbl)
            except Exception as e:
                # 某些样本可能检测失败，跳过
                continue

        distances = np.array(distances)
        labels    = np.array(labels)
        print(f"  Valid pairs: {len(distances)}")

        # 1.3 阈值搜索：找到最好区分同/异人的阈值
        best_acc    = 0.0
        best_thresh = 0.0
        for thresh in np.arange(0.20, 0.81, 0.01):
            preds = (distances < thresh).astype(int)
            acc   = (preds == labels).mean()
            if acc > best_acc:
                best_acc    = acc
                best_thresh = thresh

        # 1.4 记录并打印该数据集的最优结果
        results[model][ds_name] = {
            'threshold': best_thresh,
            'accuracy' : best_acc * 100
        }
        print(f"  → Best threshold: {best_thresh:.2f}, Accuracy: {best_acc*100:.2f}%")

# —— STEP 2: 输出所有模型 & 数据集 的汇总 ——
print("\n=== Final Summary ===")
for m, ds_info in results.items():
    for ds_name, info in ds_info.items():
        print(f"{m} @ {ds_name}: thresh={info['threshold']:.2f}, acc={info['accuracy']:.2f}%")
