from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def infer_backend_from_model(model_path: str | Path) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".trt", ".engine"}:
        return "tensorrt"
    return "ultralytics"


@dataclass(slots=True)
class ModelSettings:
    backend: str = "auto"
    model_path: str = "artifacts/models/best.pt"
    class_names: list[str] = field(default_factory=list)
    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda:0"
    use_fp16: bool = True

    @property
    def effective_backend(self) -> str:
        return infer_backend_from_model(self.model_path) if self.backend == "auto" else self.backend


@dataclass(slots=True)
class SourceSettings:
    kind: str = "image"
    path: str = ""
    camera_index: int = 0
    loop_video: bool = False


@dataclass(slots=True)
class RuntimeSettings:
    save_dir: str = "artifacts/runtime"
    save_ng_images: bool = True
    save_all_frames: bool = False
    save_csv_log: bool = True
    save_annotated_images: bool = True
    line_thickness: int = 2
    max_queue_size: int = 4


@dataclass(slots=True)
class ApiSettings:
    host: str = "0.0.0.0"
    port: int = 18080
    reload_model_on_startup: bool = True


@dataclass(slots=True)
class AlarmSettings:
    enabled: bool = True
    ng_hold_frames: int = 1
    ok_text: str = "OK"
    ng_text: str = "NG"


@dataclass(slots=True)
class TrainingSettings:
    data_yaml: str = "configs/dataset/weld.yaml"
    weights: str = "yolov8n.pt"
    epochs: int = 120
    imgsz: int = 640
    batch: int = 8
    workers: int = 4
    patience: int = 30
    project: str = "artifacts/models"
    name: str = "yolov8_train"
    device: str = "0"
    close_mosaic: int = 10
    cache: bool | str = False
    lr0: float = 0.01
    lrf: float = 0.01
    cos_lr: bool = False
    label_smoothing: float = 0.0
    dropout: float = 0.0
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    project_name: str = "weld-qc-system"
    model: ModelSettings = field(default_factory=ModelSettings)
    source: SourceSettings = field(default_factory=SourceSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    alarm: AlarmSettings = field(default_factory=AlarmSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_yaml() -> None:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML 未安装，请先执行 `pip install pyyaml`。")


def _merge_dataclass(instance: Any, overrides: dict[str, Any] | None) -> Any:
    if not overrides:
        return instance
    payload = asdict(instance)
    field_names = {item.name for item in fields(instance)}
    extra_field = "extra_args" if "extra_args" in field_names else None
    extras: dict[str, Any] = {}

    for key, value in overrides.items():
        if key in field_names:
            if extra_field == key and isinstance(payload.get(key), dict) and isinstance(value, dict):
                payload[key].update(value)
            else:
                payload[key] = value
        else:
            extras[key] = value

    if extra_field and extras:
        merged_extras = dict(payload.get(extra_field, {}))
        merged_extras.update(extras)
        payload[extra_field] = merged_extras

    return type(instance)(**payload)


def _normalize_class_names(raw_names: Any) -> list[str] | None:
    if isinstance(raw_names, dict):
        try:
            keys = sorted(raw_names, key=lambda item: int(item))
        except (TypeError, ValueError):
            keys = sorted(raw_names)
        return [str(raw_names[key]) for key in keys]

    if isinstance(raw_names, (list, tuple)):
        return [str(name) for name in raw_names]

    return None


def _load_class_names_from_dataset_yaml(dataset_yaml_path: str | Path) -> list[str] | None:
    path = resolve_project_path(dataset_yaml_path)
    if not path.exists():
        return None

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _normalize_class_names(raw.get("names"))


def load_app_config(config_path: str | Path) -> AppConfig:
    _require_yaml()
    path = resolve_project_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_model = raw.get("model") or {}
    raw_training = raw.get("training") or {}
    model = _merge_dataclass(ModelSettings(), raw_model)
    training = _merge_dataclass(TrainingSettings(), raw_training)

    normalized_class_names = _normalize_class_names(model.class_names)
    if normalized_class_names:
        model.class_names = normalized_class_names
    else:
        model.class_names = _load_class_names_from_dataset_yaml(training.data_yaml) or []

    return AppConfig(
        project_name=raw.get("project_name", AppConfig().project_name),
        model=model,
        source=_merge_dataclass(SourceSettings(), raw.get("source")),
        runtime=_merge_dataclass(RuntimeSettings(), raw.get("runtime")),
        api=_merge_dataclass(ApiSettings(), raw.get("api")),
        alarm=_merge_dataclass(AlarmSettings(), raw.get("alarm")),
        training=training,
    )


def save_app_config(config: AppConfig, config_path: str | Path) -> Path:
    _require_yaml()
    path = resolve_project_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path
