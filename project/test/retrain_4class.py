"""
Скрипт для переобучения 4-классного модели классификатора кожи.
Классы: real_clean, real_noisy, silicone_clean, silicone_noisy

Использование:
  python retrain_4class.py                    # обучить на текущих данных
  python retrain_4class.py --data /other/path # обучить на другой папке
  python retrain_4class.py --test-only        # только протестировать

Структура папок:
  data/
    real_clean/       ← настоящая кожа, хорошее фото
    real_noisy/       ← настоящая кожа, шумное фото
    silicone_clean/   ← силикон, хорошее фото
    silicone_noisy/   ← силикон, шумное фото
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DEEPUTIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPUTIN_ROOT))
sys.path.insert(0, str(DEEPUTIN_ROOT / "s2_metrics" / "modules"))

from texture_extractor import TextureExtractor

MODEL_PATH = DEEPUTIN_ROOT / "s2_metrics" / "modules" / "texture" / "skin_classifier_model_4class.pkl"
DEFAULT_DATASET = DEEPUTIN_ROOT / "test_dataset_4class"

# 4 classes: real_clean=0, real_noisy=1, silicone_clean=2, silicone_noisy=3
CLASS_NAMES = ["real_clean", "real_noisy", "silicone_clean", "silicone_noisy"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

TOP20 = [
    "glcm_dissimilarity_d5_a0", "glcm_homogeneity_d5_a0", "glcm_dissimilarity_d3_a0",
    "homo_local_var_w15_cv", "contrast_weber_mean", "homo_local_var_w31_cv",
    "color_b_mean", "glcm_homogeneity_d3_a0", "glcm_dissimilarity_d3_a135",
    "glcm_dissimilarity_d2_a0", "lbp_uniform_r5_std", "glcm_dissimilarity_d5_avg",
    "glcm_dissimilarity_d3_avg", "morph_tophat_r4_std", "glcm_dissimilarity_d5_a135",
    "glcm_dissimilarity_d2_range", "grad_sobel_mag_skewness", "residual_bio_iqr",
    "morph_tophat_r8_std", "glcm_dissimilarity_d5_a45",
]

POSE_KEYS = ["yaw", "pitch", "roll"]
TOP23 = TOP20 + POSE_KEYS


def extract_features(img_path: Path, extractor: TextureExtractor) -> dict[str, float] | None:
    """Извлекает текстурные фичи из одного изображения."""
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return None

    class Ctx:
        image_rgb = rgb
        face_bbox = [0, 0, img.shape[1], img.shape[0]]
        face_mask_path = img_path

    metrics = extractor.extract(Ctx(), exclude_sensitive=False)

    for feat in TOP20:
        val = metrics.get(feat)
        if val is None or not np.isfinite(float(val)):
            return None
    return metrics


def load_poses(dataset: Path) -> dict[str, dict]:
    poses_path = dataset / "poses.json"
    if not poses_path.exists():
        return {}
    with open(poses_path) as f:
        data = json.load(f)
    return {p["photo_id"]: p for p in data}


def save_features_cache(dataset: Path, X: np.ndarray, y: np.ndarray, names: list, feature_names: list):
    cache_path = dataset / "features_cache_4class.npz"
    np.savez_compressed(cache_path, X=X, y=y, names=np.array(names, dtype=object),
                        feature_names=np.array(feature_names, dtype=object))
    print(f"  Кэш сохранён: {cache_path} ({X.shape[0]} образцов, {X.shape[1]} фич)")


def load_features_cache(dataset: Path):
    cache_path = dataset / "features_cache_4class.npz"
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    return data["X"], data["y"], list(data["names"]), list(data["feature_names"])


def main():
    parser = argparse.ArgumentParser(description="Retrain 4-class skin classifier")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cv", type=str, default="5fold", choices=["loo", "5fold"])
    args = parser.parse_args()

    dataset = Path(args.data)

    extractor = TextureExtractor()
    extractor._skin_scorer = None

    poses_map = load_poses(dataset)
    use_poses = len(poses_map) > 0
    feature_names = TOP23 if use_poses else TOP20
    print(f"  Pose features: {'включены' if use_poses else 'нет poses.json'}")

    cached = None if args.no_cache else load_features_cache(dataset)
    if cached is not None:
        X, y, file_names, cached_feature_names = cached
        if list(cached_feature_names) != list(feature_names):
            print(f"  Кэш устарел, извлекаем заново")
            cached = None

    if cached is not None:
        X, y, file_names, _ = cached
        skipped = []
    else:
        X_rows, y_labels, file_names, skipped = [], [], [], []

        for label_name in CLASS_NAMES:
            folder = dataset / label_name
            if not folder.exists():
                print(f"  Папка не найдена: {folder}")
                continue
            masks = sorted(folder.glob("*.png"))
            print(f"\n  {label_name}: {len(masks)} файлов")
            for m in masks:
                metrics = extract_features(m, extractor)
                if metrics is None:
                    skipped.append(f"{label_name}/{m.name}")
                    continue
                row = [float(metrics[f]) for f in TOP20]
                if use_poses:
                    pose_data = poses_map.get(m.name, {})
                    for pk in POSE_KEYS:
                        row.append(float(pose_data.get(pk, 0.0)))
                X_rows.append(row)
                y_labels.append(CLASS_TO_IDX[label_name])
                file_names.append(f"{label_name}/{m.name}")

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_labels, dtype=np.int32)
        save_features_cache(dataset, X, y, file_names, feature_names)

    print(f"\n{'='*60}")
    print(f"Итого: {X.shape[0]} образцов, {X.shape[1]} фич")
    print(f"Классы: {dict(Counter(y.tolist()))}")
    if skipped:
        print(f"Пропущено: {len(skipped)}")

    if args.test_only:
        from texture.classifier import TextureSkinClassifier
        classifier = TextureSkinClassifier()
        if classifier._pipeline is None:
            print("\nМодель не найдена!")
            return
        correct = 0
        for i in range(len(X)):
            metrics_dict = dict(zip(TOP20, X[i][:20]))
            pose_dict = dict(zip(POSE_KEYS, X[i][20:])) if use_poses else None
            result = classifier.classify(metrics_dict, pose=pose_dict)
            expected = CLASS_NAMES[y[i]]
            if result.get("texture_skin_hint") == expected.split("_")[0]:
                correct += 1
        print(f"\nТочность: {correct}/{len(X)} = {correct/len(X)*100:.1f}%")
        return

    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ (4 класса)")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict

    pipeline = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42, multi_class="multinomial")),
    ])
    clf_name = "LogisticRegression (multinomial)"

    print(f"  Classifier: {clf_name}")

    if args.cv == "5fold":
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_name = "5-Fold"
    else:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
        cv_name = "LOO-CV"

    y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    total_correct = int(np.sum(y_pred == y))
    print(f"\n{cv_name} Accuracy: {total_correct}/{len(y)} = {total_correct/len(y)*100:.1f}%")

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y == cls_idx
        if mask.sum() > 0:
            acc = int(np.sum(y_pred[mask] == cls_idx))
            print(f"  {cls_name}: {acc}/{mask.sum()} = {acc/mask.sum()*100:.1f}%")

    errors = np.where(y_pred != y)[0]
    if len(errors) > 0:
        print(f"\nОшибки ({len(errors)}):")
        for idx in errors:
            expected = CLASS_NAMES[y[idx]]
            got = CLASS_NAMES[y_pred[idx]]
            print(f"  {file_names[idx]}: ожидалось={expected}, получено={got}")

    pipeline.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "feature_names": feature_names,
            "class_names": CLASS_NAMES,
        }, f)
    print(f"\nМодель сохранена: {MODEL_PATH}")

    lr = pipeline.named_steps["clf"]
    if hasattr(lr, "coef_"):
        print(f"\nКоэффициенты (4 класса):")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            print(f"\n  {cls_name}:")
            for name, coef in sorted(zip(feature_names, lr.coef_[cls_idx]), key=lambda x: abs(x[1]), reverse=True)[:10]:
                print(f"    {name:40s}: {coef:+.4f}")


if __name__ == "__main__":
    main()
