from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class FrameResult:
    frame_id: int
    source: str
    detections: list[Detection]
    inference_ms: float
    fps: float
    status: str
    backend: str
    model_path: str
    image_size: tuple[int, int]
    saved_path: str | None = None
    defect_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.defect_count = len(self.detections)

    @property
    def has_defect(self) -> bool:
        return self.defect_count > 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["has_defect"] = self.has_defect
        return payload


def format_result_summary(result: FrameResult) -> str:
    if not result.detections:
        return (
            f"[{result.status}] 无缺陷 | backend={result.backend} | "
            f"{result.inference_ms:.1f} ms | {result.fps:.1f} FPS"
        )
    lines = [
        (
            f"[{result.status}] {result.defect_count} 个缺陷 | "
            f"backend={result.backend} | {result.inference_ms:.1f} ms | {result.fps:.1f} FPS"
        )
    ]
    for idx, detection in enumerate(result.detections, start=1):
        x1, y1, x2, y2 = detection.bbox
        lines.append(
            f"{idx}. {detection.label} {detection.confidence:.3f} "
            f"bbox=({x1},{y1},{x2},{y2})"
        )
    return "\n".join(lines)

