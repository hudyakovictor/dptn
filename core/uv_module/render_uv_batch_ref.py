#!/usr/bin/env python3
"""
Тестовые рендеры UV-текстуры и масок по списку фото (полные пути).

Пример (нужен Python с torch, как в проекте):

  cd /path/to/dutin
  ./image_enhancer/venv/bin/python uv_module/render_uv_batch_ref.py

Пути к входам — ``INPUT_IMAGES``; выход: ``app/masktest/uv_test_renders_fastest/``.

Файлы на кадр:
  - ``*_uv.png`` — UV-текстура (BGR);
  - ``*_mask_bake_visible.png`` — маска bake (есть сэмпл с фото);
  - ``*_confidence.png`` — карта confidence;
  - ``*_mask_analytic.png`` — bake ∧ порог по контрастной confidence ∧ uv_is_original;
  - ``*_confidence_stretched.png`` — контрастная карта для порога (перцентили 5/95 по видимой маске, степень 1.25);
  - ``*_forehead_wrinkle_crop.png`` — обрезка UV-текстуры лба (~650×210 в коорд. 1024), BGR с фото;
  - ``*_forehead_wrinkle_mask.png`` — та же сетка: ROI ∧ bake ∧ analytic (для метрик морщин).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_FASTEST = _REPO / "app" / "fastest"
INPUT_IMAGES = [
    str(_FASTEST / "Снимок экрана 2026-04-09 в 21.13.12.png"),
    str(_FASTEST / "Снимок экрана 2026-04-09 в 21.12.36.png"),
    str(_FASTEST / "Снимок экрана 2026-04-09 в 21.12.30.png"),
    str(_FASTEST / "Снимок экрана 2026-04-09 в 21.10.19.png"),
]

OUT_DIR = _REPO / "app" / "masktest" / "uv_test_renders_fastest"


def main() -> int:
    from app.pipeline.hduv_texture import build_recon_dict_from_result, make_hduv_generator
    from app.pipeline.reconstruction import ReconstructionAdapter
    from uv_module.analysis import build_uv_analysis_bundle
    from uv_module.forehead_wrinkle_roi import forehead_wrinkle_crop_and_mask

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recon = ReconstructionAdapter(device="cpu")
    gen = make_hduv_generator(uv_size=1024, super_sample=1)

    for idx, raw in enumerate(INPUT_IMAGES):
        p = Path(raw)
        if not p.is_file():
            print(f"[skip] нет файла: {p}", file=sys.stderr)
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"[skip] не читается: {p}", file=sys.stderr)
            continue
        stem = f"{idx:02d}_{p.stem}"[:120]
        print(f"… {p.name}")

        res = recon.reconstruct(p)
        if res.uv_coords is None:
            print(f"[skip] нет uv_coords: {p.name}", file=sys.stderr)
            continue

        recon_dict = build_recon_dict_from_result(res)
        uv_tex, uv_mask, uv_conf, aux = gen.generate(img, recon_dict)

        cv2.imwrite(str(OUT_DIR / f"{stem}_uv.png"), uv_tex)
        cv2.imwrite(
            str(OUT_DIR / f"{stem}_mask_bake_visible.png"),
            (np.asarray(uv_mask, dtype=np.uint8) * 255),
        )
        cv2.imwrite(
            str(OUT_DIR / f"{stem}_confidence.png"),
            np.clip(uv_conf * 255.0, 0, 255).astype(np.uint8),
        )

        bundle = build_uv_analysis_bundle(
            uv_tex,
            uv_mask,
            uv_conf,
            aux=aux,
            min_confidence=0.42,
            require_original=True,
            min_coverage_ratio=0.03,
            confidence_contrast="percentile",
            contrast_low_percentile=5.0,
            contrast_high_percentile=95.0,
            contrast_power=1.25,
        )
        cv2.imwrite(
            str(OUT_DIR / f"{stem}_mask_analytic.png"),
            (bundle.analytic_mask.astype(np.uint8) * 255),
        )
        cs = bundle.aux.get("confidence_stretched")
        if cs is not None:
            cv2.imwrite(
                str(OUT_DIR / f"{stem}_confidence_stretched.png"),
                np.clip(np.asarray(cs, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8),
            )

        f_crop, f_mask, _roi = forehead_wrinkle_crop_and_mask(
            uv_tex,
            uv_mask,
            bundle.analytic_mask,
        )
        cv2.imwrite(str(OUT_DIR / f"{stem}_forehead_wrinkle_crop.png"), f_crop)
        cv2.imwrite(str(OUT_DIR / f"{stem}_forehead_wrinkle_mask.png"), f_mask)

    print(f"OK -> {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
