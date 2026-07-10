"""Реконструкция 3D лица через 3DDFA_v3."""
from __future__ import annotations

import gc
import hashlib
import os
import pickle
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from deeputin.shared.utils import classify_pose_bucket

from .types import ReconstructionResult


def get_3ddfa_root() -> Path:
    """Resolve 3DDFA-V3 root path from env or relative to this file."""
    env_path = os.environ.get("DUTIN_3DDFA_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    # Fallback: relative to this file (project/core/3ddfa_v3)
    return Path(__file__).resolve().parents[3] / "core" / "3ddfa_v3"


CORE_3DDFA_ROOT = get_3ddfa_root()


def log_progress(message: str) -> None:
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# Add core paths for uv_module and other dependencies
CORE_ROOT = Path(__file__).resolve().parents[3] / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

if str(CORE_3DDFA_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_3DDFA_ROOT))

try:
    from demo import resolve_detector_device, resolve_torch_device  # type: ignore
    from face_box import face_box  # type: ignore
    from model.recon import face_model  # type: ignore
    _HAS_3DDFA = True
    _IMPORT_ERROR = None
except ImportError as e:
    _HAS_3DDFA = False
    _IMPORT_ERROR = e
    def resolve_detector_device(device: str) -> str:
        return device
    def resolve_torch_device(device: str) -> str:
        return device
    def face_box(*args, **kwargs):
        raise RuntimeError(f"3DDFA_v3 not available: {_IMPORT_ERROR}")
    def face_model(*args, **kwargs):
        raise RuntimeError(f"3DDFA_v3 not available: {_IMPORT_ERROR}")


def compute_image_hash(image_path: Path | str) -> str:
    """
    [SYS-03] MD5-хэширование содержимого файла.
    Защищает кэш от ложной инвалидации при копировании папок (изменение mtime).
    """
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


class ReconstructionAdapter:
    def __init__(self, device: str = "auto", detector_device: str = "auto", backbone: str = "resnet50", _max_cache_size: int = 10):
        self.device = resolve_torch_device(device)
        self.detector_device = resolve_detector_device(detector_device)
        self.backbone = backbone
        self.runtime_device = self._resolve_runtime_device(self.device)
        self._face_model_assets = self._load_face_model_assets()
        self._cache: dict[str, ReconstructionResult] = {}
        self._cache_keys_queue: list[str] = []
        self._model = None
        self._detector = None
        self._max_cache_size = _max_cache_size
        self._last_reconstruction_trust_issue: str | None = None
        self._init_models()

    def _evict_cache_if_needed(self):
        """
        [SYS-02] Защита от OOM (Out Of Memory).
        Удаляем старые тензоры и принудительно очищаем VRAM.
        """
        while len(self._cache) >= self._max_cache_size:
            oldest_key = self._cache_keys_queue.pop(0)
            removed_item = self._cache.pop(oldest_key, None)

            # Явно удаляем тяжелые тензоры
            if removed_item:
                del removed_item

            # Принудительный вызов сборщика мусора Python
            gc.collect()

            # Очистка кэша видеопамяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            log_progress(f"Cache evicted: {oldest_key}. VRAM cleared.")

    def _load_face_model_assets(self) -> dict:
        assets_path = CORE_3DDFA_ROOT / "assets" / "face_model.npy"
        return np.load(assets_path, allow_pickle=True).item()

    def _resolve_runtime_device(self, preferred: str) -> str:
        if sys.platform == "darwin":
            log_progress("Mac detected: forcing CPU for 3DDFA_v3 renderer stability")
            return "cpu"
        if preferred == "cpu":
            return "cpu"
        try:
            import nvdiffrast.torch  # type: ignore
            return preferred
        except ModuleNotFoundError:
            return "cpu"

    def _init_models(self) -> None:
        log_progress(
            f"Initializing 3DDFA_v3 model backbone={self.backbone} "
            f"device={self.runtime_device} detector_device={self.detector_device}"
        )
        args = Namespace(
            device=self.runtime_device,
            detector_device=self.detector_device,
            iscrop=True,
            detector="retinaface",
            ldm68=False,
            ldm106=True,
            ldm106_2d=False,
            ldm134=False,
            seg=True,
            seg_visible=True,
            useTex=False,
            extractTex=False,
            extractTexNew=False,
            uv_res=1024,
            detail_strength=0.75,
            backbone=self.backbone,
        )
        cwd = Path.cwd()
        try:
            os.chdir(CORE_3DDFA_ROOT)
            self._model = face_model(args)
            self._detector = face_box(args).detector
        finally:
            os.chdir(cwd)
        log_progress("3DDFA_v3 model initialized")

    def reconstruct(
        self,
        image_path: Path,
        neutral_expression: bool = False,
        identity_only: bool = False,
    ) -> ReconstructionResult:
        if self._model is None or self._detector is None:
            raise RuntimeError("3DDFA_v3 model is not initialized")

        log_progress(f"Reconstructing {image_path.name}: loading image")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image {image_path}: {exc}") from exc

        # [FIX 37]: Pixel-based image MD5 hashing to prevent cache collision
        pixel_array = np.asarray(image)
        file_hash = hashlib.md5(pixel_array.tobytes()).hexdigest()
        cache_key = f"{image_path.name}_{file_hash}_{neutral_expression}_{identity_only}_{self.backbone}"

        if cache_key in self._cache:
            log_progress(f"Cache HIT for {image_path.name}")
            return self._cache[cache_key]

        log_progress(f"Cache MISS for {image_path.name}. Extracting 3D...")
        self._evict_cache_if_needed()
        self._last_reconstruction_trust_issue = None

        # [FIX 38]: Robust try...except block that guarantees CPU/GPU VRAM cleanup on forward-pass failure
        try:
            log_progress(f"Reconstructing {image_path.name}: running face detector")
            try:
                trans_params, image_tensor = self._detector(image)
            except Exception as exc:
                raise RuntimeError(f"Face detector failed on {image_path.name}: {exc}") from exc

            if image_tensor is None:
                raise RuntimeError(f"No face detected in {image_path.name}")
            original_tensor = (
                torch.tensor(pixel_array / 255.0, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.runtime_device)
            )

            self._model.input_img = image_tensor.to(self.runtime_device)
            self._model.orig_img_tensor = original_tensor
            self._model.trans_params = trans_params
            log_progress(f"Reconstructing {image_path.name}: running 3D forward pass")
            cached_alpha: dict[str, torch.Tensor] = {}

            def _capture_recon_alpha(_module, _inputs, output) -> None:
                cached_alpha["alpha"] = output

            hook = self._model.net_recon.register_forward_hook(_capture_recon_alpha)
            try:
                result = self._model.forward()
            finally:
                hook.remove()

            log_progress(f"Reconstructing {image_path.name}: extracting geometry payload")
            alpha = cached_alpha.get("alpha")
            if alpha is None:
                alpha = self._model.net_recon(self._model.input_img)
            alpha_dict = self._model.split_alpha(alpha)

            exp_tensor = alpha_dict["exp"]
            if identity_only:
                exp_tensor = torch.zeros_like(exp_tensor)
            elif neutral_expression:
                # [FIX-A1] Сглаживание без потери анатомической асимметрии
                exp_tensor = exp_tensor * 0.1

            base_shape = self._model.compute_shape(alpha_dict["id"], exp_tensor)
            rotation = self._model.compute_rotation(alpha_dict["angle"])
            transformed_shape = self._model.transform(base_shape, rotation, alpha_dict["trans"])
            vertices_camera = self._model.to_camera(transformed_shape.clone())
            vertices_image = self._model.to_image(vertices_camera.clone())
            normals_world = self._model.compute_norm(base_shape).detach().cpu().numpy()[0]
            normals_camera = (self._model.compute_norm(base_shape) @ rotation).detach().cpu().numpy()[0]
            vertices_world_np = transformed_shape.detach().cpu().numpy()[0]
            v_sz = np.ptp(vertices_world_np, axis=0)

            visible_idx_renderer = self._derive_visible_idx_renderer(result, normals_camera)
            angles_deg = np.rad2deg(alpha_dict["angle"].detach().cpu().numpy()[0])
            log_progress(
                f"Reconstructing {image_path.name}: done (visible={int(np.count_nonzero(visible_idx_renderer))}) "
                f"Dims(X={v_sz[0]:.3f}, Y={v_sz[1]:.3f}, Z={v_sz[2]:.3f})"
            )

            trans_p = alpha_dict["trans"].detach().cpu().numpy()[0]

            exp_np = exp_tensor.detach().cpu().numpy()[0]

            # 3DDFA-V3 returns angles as [pitch, yaw, roll] in radians
            raw_yaw = float(angles_deg[1])
            _pose_bucket = classify_pose_bucket(
                raw_yaw,
                float(angles_deg[0]),
                float(angles_deg[2]),
            )

            reconstruction = ReconstructionResult(
                image_path=image_path,
                vertices_world=vertices_world_np,
                vertices_camera=vertices_camera.detach().cpu().numpy()[0],
                vertices_image=vertices_image.detach().cpu().numpy()[0],
                triangles=self._model.tri.detach().cpu().numpy(),
                point_buffer=self._model.point_buf.detach().cpu().numpy(),
                annotation_groups=[
                    np.asarray(group, dtype=np.int64) for group in self._face_model_assets["annotation"]
                ],
                visible_idx_renderer=visible_idx_renderer,
                normals_world=normals_world,
                normals_camera=normals_camera,
                rotation_matrix=rotation.detach().cpu().numpy()[0],
                translation=trans_p,
                angles_deg=angles_deg,
                trans_params=None if trans_params is None else np.asarray(trans_params),
                landmarks_106=None if "ldm106" not in result else np.asarray(result["ldm106"])[0],
                uv_coords=self._model.uv_coords.detach().cpu().numpy() if hasattr(self._model, 'uv_coords') else None,
                pose_bucket=_pose_bucket,
                payload={
                    "raw_result": result,
                    "alpha_angles_deg": angles_deg,
                    "exp_params": exp_np,
                    "expression": {
                        "mouth_open_intensity": float(abs(exp_np[0])),
                        "jaw_open_intensity": float(abs(exp_np[0])),
                        "smile_intensity": float(max(abs(exp_np[1]), abs(exp_np[2]))),
                    },
                    "id_params": alpha_dict["id"].detach().cpu().numpy()[0],
                    "runtime_device": self.runtime_device,
                    "detector_device": self.detector_device,
                    "backbone": self.backbone,
                    "seg_visible": result.get("seg_visible"),
                    # [FIX-A2] Trust issue с маской видимости для downstream обработки
                    "trust_issue": getattr(self, '_last_reconstruction_trust_issue', None),
                },
            )
            self._cache[cache_key] = reconstruction
            self._cache_keys_queue.append(cache_key)
            return reconstruction

        except Exception as e:
            # Clear CUDA and MPS cache immediately on failure!
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            raise e


    def _derive_visible_idx_renderer(self, result: dict, normals_camera: np.ndarray) -> np.ndarray:
        """
        [PIPE-01] Маска видимости (Visibility Mask).
        Согласно ТЗ pipeline/task.txt:
        1. Для каждой вершины рассчитываем угол между нормалью и вектором на камеру.
        2. Если угол > 82 градусов — вершина отбрасывается.
        """
        # В камерных координатах 3DDFA_v3 камера находится в (0,0,0) и смотрит вдоль Z
        # Вектор на камеру для каждой вершины v — это -v, но в ортографической проекции
        # которую использует 3DDFA, вектор на камеру — это [0, 0, -1] (на нас).
        # Однако, 3DDFA_v3 может использовать перспективную или сложную проекцию.
        # В большинстве случаев для маскировки "затылка" достаточно проверить Z-компоненту нормали.

        # Вектор "на камеру" в пространстве камеры для 3DDFA обычно [0, 0, 1] или [0, 0, -1]
        # Мы используем нормали в пространстве камеры.
        # Вершины, смотрящие на нас, имеют положительную Z компоненту нормали (в системе 3DDFA).
        
        # Вектор "на камеру" в пространстве камеры для 3DDFA:
        # 3DDFA_v3 использует конвенцию где камера смотрит вдоль +Z,
        # а нормали лица направлены в сторону камеры (normal_z > 0 = лицом к камере).
        # Поэтому cos_theta = normals_camera[:, 2] (без отрицания).
        cos_theta = normals_camera[:, 2]
        
        # Ограничиваем значения для arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles_rad = np.arccos(cos_theta)
        angles_deg = np.rad2deg(angles_rad)
        
        # Угол > 82° -> невидима
        mask_82 = angles_deg <= 82.0
        
        # Маска рендерера дополняет угловой фильтр; при отсутствии — только mask_82.
        visible = mask_82.copy()
        raw_visible = result.get("visible_idx")

        trust_issue = None
        if raw_visible is not None:
            raw_visible = np.asarray(raw_visible)
            renderer_mask = np.zeros(normals_camera.shape[0], dtype=bool)
            if raw_visible.ndim == 1 and raw_visible.size > 0 and raw_visible.dtype != np.bool_:
                raw_visible = raw_visible.astype(np.int64)
                raw_visible = raw_visible[(raw_visible >= 0) & (raw_visible < renderer_mask.shape[0])]
                renderer_mask[raw_visible] = True
            elif raw_visible.shape[0] == visible.shape[0]:
                renderer_mask = raw_visible.astype(bool)
            else:
                trust_issue = f"visible_mask_shape_mismatch: raw={raw_visible.shape[0]}, expected={visible.shape[0]}"
            if trust_issue is None:
                visible &= renderer_mask
        else:
            trust_issue = "visible_mask_missing_from_renderer"

        # Финальная маска: угловой фильтр (+ окклюзии рендерера, если доступны)
        visible &= np.isfinite(normals_camera).all(axis=1)
        
        # [FIX-A2] Сохраняем trust issue для downstream обработки
        if trust_issue:
            self._last_reconstruction_trust_issue = trust_issue
        
        return visible


# --- Дисковый кэш реконструкции (ТЗ: один раз извлечь 3D, дальше только пары / выравнивание) ---
RECONSTRUCTION_CACHE_NAME = "reconstruction_v1.pkl"


def _sanitize_payload_for_disk(payload: dict[str, Any]) -> dict[str, Any]:
    """Убираем raw_result и тензоры; оставляем только то, что нужно UV/маскам и сравнению."""
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if k == "raw_result":
            continue
        if hasattr(v, "detach"):
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {
                sk: sv
                for sk, sv in v.items()
                if isinstance(sv, (str, int, float, bool)) or sv is None
            }
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
    return out


def save_reconstruction_cache(
    entry_dir: Path,
    result: ReconstructionResult,
    neutral_expression: bool,
    identity_only: bool = False,
) -> Path:
    """Сохраняет геометрию 3DDFA с MD5-хэшем для верификации."""
    entry_dir.mkdir(parents=True, exist_ok=True)
    path = entry_dir / RECONSTRUCTION_CACHE_NAME

    file_hash = compute_image_hash(result.image_path)

    slim = replace(result, payload=_sanitize_payload_for_disk(result.payload))
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "v": 1,
                "neutral_expression": neutral_expression,
                "identity_only": identity_only,
                "file_hash": file_hash,
                "result": slim
            },
            f,
            protocol=4,
        )
    tmp.replace(path)
    return path


def load_reconstruction_cache(
    entry_dir: Path,
    image_path: Path,
    neutral_expression: bool,
    identity_only: bool = False,
    *,
    verify_hash: bool = True,
) -> ReconstructionResult | None:
    """Загрузка кэша по MD5-хэшу исходника (verify_hash=False — только pkl на диске)."""
    path = entry_dir / RECONSTRUCTION_CACHE_NAME
    if not path.exists():
        return None

    file_hash = compute_image_hash(image_path) if verify_hash else None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("v") != 1:
        return None
    if bool(data.get("neutral_expression")) != bool(neutral_expression):
        return None
    if bool(data.get("identity_only", False)) != bool(identity_only):
        return None

    if verify_hash and data.get("file_hash") != file_hash:
        return None

    r = data.get("result")
    if not isinstance(r, ReconstructionResult):
        return None
    return replace(r, image_path=image_path)


def resolve_reconstruction(
    adapter: ReconstructionAdapter,
    image_path: Path,
    entry_dir: Path,
    neutral_expression: bool,
    identity_only: bool = False,
) -> ReconstructionResult:
    """Кэш на диске или полный reconstruct + сохранение."""
    cached = load_reconstruction_cache(
        entry_dir, image_path, neutral_expression, identity_only
    )
    if cached is not None:
        log_progress(f"Reconstruction disk cache hit: {image_path.name}")
        return cached
    r = adapter.reconstruct(
        image_path,
        neutral_expression=neutral_expression,
        identity_only=identity_only,
    )
    save_reconstruction_cache(entry_dir, r, neutral_expression, identity_only)
    return r
