"""Тест texture_extractor на 4 папках с face_crop.jpg."""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2_metrics.modules.texture_extractor import TextureExtractor

DATASET_ROOT = Path("/Volumes/SDCARD/анозер текст")
FOLDERS = ["live_clean", "live_noise", "mask_clean", "put_context"]


def load_image(path: Path):
    """Загрузка изображения."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def test_folder(folder_name: str, extractor: TextureExtractor):
    """Тест одной папки."""
    folder = DATASET_ROOT / folder_name
    if not folder.exists():
        print(f"  Папка не найдена: {folder}")
        return {}

    face_crops = sorted(folder.rglob("face_crop.jpg"))
    if not face_crops:
        print(f"  Нет face_crop.jpg в {folder_name}")
        return {}

    print(f"  {folder_name}: {len(face_crops)} фото")

    results = {
        "total": len(face_crops),
        "with_quality": 0,
        "sensitive_excluded": 0,
        "metrics_count": defaultdict(list),
        "quality_stats": defaultdict(list),
    }

    for i, path in enumerate(face_crops[:10]):  # Тест на первых 10
        img = load_image(path)
        if img is None:
            continue

        # Создаём простой контекст
        class MockCtx:
            image_rgb = img
            face_bbox = [0, 0, img.shape[1], img.shape[0]]

        ctx = MockCtx()

        # Извлекаем метрики с фильтрацией
        extractor._quality_sensitive_excluded = False
        metrics = extractor.extract(ctx, exclude_sensitive=True)

        # Считаем
        results["with_quality"] += 1
        if extractor._quality_sensitive_excluded:
            results["sensitive_excluded"] += 1

        for k in metrics:
            results["metrics_count"][k].append(1)

        # Качество
        for k in ["noise_level", "sharpness_score", "jpeg_blockiness", "overall_quality"]:
            if k in metrics:
                results["quality_stats"][k].append(metrics[k])

        if i == 0:
            print(f"    Пример: {path.name} → {len(metrics)} метрик")
            print(f"      Качество: noise={metrics.get('noise_level', 'N/A'):.2f}, "
                  f"sharpness={metrics.get('sharpness_score', 'N/A'):.2f}, "
                  f"overall={metrics.get('overall_quality', 'N/A'):.2f}")

    return results


def main():
    print("=" * 60)
    print("Тест TextureExtractor на 4 папках")
    print("=" * 60)

    extractor = TextureExtractor()

    all_results = {}
    for folder in FOLDERS:
        print(f"\n[{folder}]")
        all_results[folder] = test_folder(folder, extractor)

    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)

    for folder, res in all_results.items():
        if not res:
            continue
        print(f"\n{folder}:")
        print(f"  Всего фото: {res['total']}")
        print(f"  Обработано: {res['with_quality']}")
        print(f"  Чувствительные исключены: {res['sensitive_excluded']}")
        print(f"  Уникальных метрик: {len(res['metrics_count'])}")

        if res['quality_stats']:
            print(f"  Среднее качество:")
            for k, v in res['quality_stats'].items():
                print(f"    {k}: {sum(v)/len(v):.3f}")


if __name__ == "__main__":
    main()
