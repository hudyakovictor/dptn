#!/usr/bin/env python3
"""
Один кадр: фото → UV-текстура (3DDFA через app.pipeline + бейк uv_module).

Запуск из корня репозитория:

    cd /path/to/dutin
    python -m uv_module.render_uv -i photo.jpg
    python -m uv_module.render_uv -i photo.jpg -o custom/face_uv.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_UV_OUT_DIR = _REPO_ROOT / "app" / "masktest" / "test_results"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Рендер UV-текстуры с одного фото")
    p.add_argument("--input", "-i", type=Path, required=True, help="Входное изображение")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=f"Куда сохранить UV (BGR JPEG). По умолчанию: {_DEFAULT_UV_OUT_DIR}/<stem>_uv.jpg",
    )
    p.add_argument("--uv-size", type=int, default=2048, help="Размер стороны UV")
    p.add_argument("--super-sample", type=int, default=1, help="Суперсэмплинг бейкера")
    args = p.parse_args()

    image_path = args.input.resolve()
    if not image_path.is_file():
        print(f"Нет файла: {image_path}", file=sys.stderr)
        return 2

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Не удалось прочитать изображение: {image_path}", file=sys.stderr)
        return 2

    out_path = args.output
    if out_path is None:
        _DEFAULT_UV_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _DEFAULT_UV_OUT_DIR / f"{image_path.stem}_uv.jpg"
    else:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    from app.pipeline.hduv_texture import generate_hduv_from_reconstruction
    from app.pipeline.reconstruction import ReconstructionAdapter
    from uv_module.hd_uv_generator import HDUVConfig, HDUVTextureGenerator

    recon = ReconstructionAdapter(device="cpu")
    result = recon.reconstruct(image_path)

    generator = HDUVTextureGenerator(
        HDUVConfig(uv_size=args.uv_size, super_sample=args.super_sample)
    )
    uv_tex, _mask, _conf, _aux = generate_hduv_from_reconstruction(
        image_bgr,
        result,
        generator=generator,
    )

    cv2.imwrite(str(out_path), uv_tex)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
