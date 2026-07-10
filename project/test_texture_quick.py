"""Тест texture_extractor на 3 папках: real, real2, silicone."""
import sys
from pathlib import Path
from collections import defaultdict

# Добавляем корень deeputin в путь
DEEPUTIN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEEPUTIN_ROOT))

# Импортируем напрямую из модуля
sys.path.insert(0, str(DEEPUTIN_ROOT / "s2_metrics" / "modules"))
from texture_extractor import TextureExtractor

DATASET_ROOT = Path(__file__).resolve().parent.parent / "imgtest" / "2"
FOLDERS = ["real", "silicone"]  # real2 - JPG без масок, пропускаем


def test_folder(folder_name: str, extractor: TextureExtractor, max_photos: int = 10):
    """Тест одной папки."""
    folder = DATASET_ROOT / folder_name
    if not folder.exists():
        print(f"  Папка не найдена: {folder}")
        return {}

    face_masks = sorted(folder.glob("*_face_mask.png"))
    if not face_masks:
        print(f"  Нет *_face_mask.png в {folder_name}")
        return {}

    face_masks = face_masks[:max_photos]
    print(f"  {folder_name}: {len(face_masks)} фото")

    results = {
        "total": len(face_masks),
        "metrics_count": defaultdict(list),
        "skin_prob": [],
        "skin_confidence": [],
        "texture_gray_mean": [],
        "texture_glcm_contrast": [],
        "glcm_dissimilarity_d5_a0": [],
    }

    for i, path in enumerate(face_masks):
        try:
            import cv2
            import numpy as np

            # Загружаем face_mask.png с alpha каналом
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"    Ошибка загрузки: {path.name}")
                continue

            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3]
                skin_mask = (alpha > 30).astype(np.uint8)
                rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            else:
                print(f"    Нет alpha канала: {path.name}")
                continue

            # Создаём контекст
            class MockCtx:
                image_rgb = rgb
                face_bbox = [0, 0, img.shape[1], img.shape[0]]
                face_mask_path = path

            ctx = MockCtx()

            # Извлекаем метрики
            metrics = extractor.extract(ctx, exclude_sensitive=False)

            # Собираем результаты
            results["skin_prob"].append(metrics.get("silicone_prob", 0.0))
            results["skin_confidence"].append(metrics.get("skin_confidence", 0.0))
            results["texture_gray_mean"].append(metrics.get("texture_gray_mean", 0.0))
            results["texture_glcm_contrast"].append(metrics.get("texture_glcm_contrast", 0.0))
            results["glcm_dissimilarity_d5_a0"].append(metrics.get("glcm_dissimilarity_d5_a0", 0.0))

            for k in metrics:
                results["metrics_count"][k].append(1)

            if i == 0:
                print(f"    Пример: {path.name}")
                silicone_prob = metrics.get('silicone_prob', 0.0)
                skin_confidence = metrics.get('skin_confidence', 0.0)
                texture_gray_mean = metrics.get('texture_gray_mean', 0.0)
                glcm_d5_a0 = metrics.get('glcm_dissimilarity_d5_a0', 0.0)
                print(f"      silicone_prob={silicone_prob:.3f}")
                print(f"      skin_confidence={skin_confidence:.3f}")
                print(f"      texture_gray_mean={texture_gray_mean:.1f}")
                print(f"      glcm_d5_a0={glcm_d5_a0:.3f}")

        except Exception as e:
            print(f"    Ошибка на {path.name}: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    print("=" * 70)
    print("Тест TextureExtractor на 3 папках (real, silicone)")
    print("=" * 70)

    extractor = TextureExtractor()
    print(f"Backend skin_authenticity доступен: {extractor._skin_scorer is not None}")

    all_results = {}
    for folder in FOLDERS:
        print(f"\n[{folder}]")
        all_results[folder] = test_folder(folder, extractor, max_photos=10)

    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ")
    print("=" * 70)

    for folder, res in all_results.items():
        if not res:
            continue
        print(f"\n{folder}:")
        print(f"  Всего фото: {res['total']}")
        print(f"  Уникальных метрик: {len(res['metrics_count'])}")

        if res["skin_prob"]:
            avg_prob = sum(res["skin_prob"]) / len(res["skin_prob"])
            avg_conf = sum(res["skin_confidence"]) / len(res["skin_confidence"])
            print(f"  Средний silicone_prob: {avg_prob:.3f}")
            print(f"  Средний skin_confidence: {avg_conf:.3f}")

        if res["glcm_dissimilarity_d5_a0"]:
            avg_glcm = sum(res["glcm_dissimilarity_d5_a0"]) / len(res["glcm_dissimilarity_d5_a0"])
            print(f"  Средний glcm_d5_a0: {avg_glcm:.3f}")

    # Сравнение
    if "real" in all_results and "silicone" in all_results:
        real = all_results["real"]
        silicone = all_results["silicone"]
        if real.get("skin_prob") and silicone.get("skin_prob"):
            real_avg = sum(real["skin_prob"]) / len(real["skin_prob"])
            silicone_avg = sum(silicone["skin_prob"]) / len(silicone["skin_prob"])
            print(f"\n--- СРАВНЕНИЕ ---")
            print(f"Real silicone_prob: {real_avg:.3f}")
            print(f"Silicone silicone_prob: {silicone_avg:.3f}")
            print(f"Разница: {silicone_avg - real_avg:.3f}")
            if silicone_avg > real_avg:
                print("✓ Классификатор правильно отличает silicone от real")
            else:
                print("✗ Классификатор НЕ правильно отличает silicone от real")

    # Показываем примеры метрик из backend
    print("\n--- ПРИМЕРЫ МЕТРИК ИЗ BACKEND ---")
    for folder, res in all_results.items():
        if not res or not res.get("skin_prob"):
            continue
        print(f"\n{folder} (первые 3 фото):")
        for i in range(min(3, len(res["skin_prob"]))):
            print(f"  Фото {i+1}: silicone_prob={res['skin_prob'][i]:.3f}, confidence={res['skin_confidence'][i]:.3f}")


if __name__ == "__main__":
    main()
