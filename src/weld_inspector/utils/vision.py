from __future__ import annotations

import base64
from typing import Sequence

import cv2
import numpy as np

from ..schemas import Detection, FrameResult


def letterbox(
    image: np.ndarray,
    new_shape: int | tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    original_height, original_width = image.shape[:2]
    ratio = min(new_shape[0] / original_height, new_shape[1] / original_width)
    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_width = new_shape[1] - resized_width
    pad_height = new_shape[0] - resized_height
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))
    bordered = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return bordered, ratio, (left, top)


def preprocess_image(
    image: np.ndarray,
    input_size: int,
    use_fp16: bool = False,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    padded, ratio, pad = letterbox(image, input_size)
    rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    if use_fp16:
        tensor = tensor.astype(np.float16)
    return tensor, ratio, pad


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.zeros_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _prepare_predictions(raw_output: np.ndarray, class_count: int) -> np.ndarray:
    predictions = np.asarray(raw_output)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"YOLO 输出维度不正确: {predictions.shape}")
    if predictions.shape[0] <= class_count + 5:
        predictions = predictions.T
    return predictions


def _resolve_class_names(class_names: list[str], inferred_class_count: int) -> list[str]:
    if inferred_class_count <= 0:
        raise ValueError(f"推断得到的类别数无效: {inferred_class_count}")

    normalized = [str(name) for name in class_names]
    if len(normalized) == inferred_class_count:
        return normalized
    if inferred_class_count == 1:
        return ["defect"]

    resolved = normalized[:inferred_class_count]
    while len(resolved) < inferred_class_count:
        resolved.append(f"class_{len(resolved)}")
    return resolved


def _infer_output_layout(
    predictions: np.ndarray,
    configured_class_count: int,
) -> tuple[bool, int]:
    column_count = predictions.shape[1]
    candidates: list[tuple[bool, int]] = []

    class_count_without_objectness = column_count - 4
    if class_count_without_objectness >= 1:
        candidates.append((False, class_count_without_objectness))

    class_count_with_objectness = column_count - 5
    if class_count_with_objectness >= 1:
        candidates.append((True, class_count_with_objectness))

    if not candidates:
        raise ValueError(f"无法从输出 shape={predictions.shape} 推断类别数。")

    for has_objectness, inferred_class_count in candidates:
        if inferred_class_count == configured_class_count:
            return has_objectness, inferred_class_count

    if len(candidates) == 1:
        return candidates[0]

    def candidate_distance(candidate: tuple[bool, int]) -> tuple[int, int]:
        has_objectness, inferred_class_count = candidate
        return abs(inferred_class_count - configured_class_count), int(has_objectness)

    return min(candidates, key=candidate_distance)


def postprocess_yolov8_output(
    raw_output: np.ndarray,
    original_shape: Sequence[int],
    ratio: float,
    pad: tuple[int, int],
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    predictions = _prepare_predictions(raw_output, class_count=len(class_names))
    has_objectness, inferred_class_count = _infer_output_layout(
        predictions=predictions,
        configured_class_count=len(class_names),
    )
    effective_class_names = _resolve_class_names(
        class_names=class_names,
        inferred_class_count=inferred_class_count,
    )

    if has_objectness:
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        confidences = objectness * class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
    else:
        class_scores = predictions[:, 4:]
        confidences = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

    keep_mask = confidences >= conf_threshold
    if not np.any(keep_mask):
        return []

    boxes = predictions[keep_mask, :4]
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]

    boxes = _xywh_to_xyxy(boxes)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio
    height, width = original_shape[:2]
    boxes = _clip_boxes(boxes, width=width, height=height)

    nms_boxes = [
        [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]
        for box in boxes
    ]
    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=confidences.tolist(),
        score_threshold=float(conf_threshold),
        nms_threshold=float(iou_threshold),
    )
    if len(indices) == 0:
        return []

    detections: list[Detection] = []
    flat_indices = np.array(indices).reshape(-1)
    for index in flat_indices:
        box = boxes[index].astype(int)
        class_id = int(class_ids[index])
        label = (
            effective_class_names[class_id]
            if class_id < len(effective_class_names)
            else str(class_id)
        )
        detections.append(
            Detection(
                label=label,
                confidence=float(confidences[index]),
                bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            )
        )
    return detections


def annotate_frame(
    image: np.ndarray,
    result: FrameResult,
    line_thickness: int = 2,
) -> np.ndarray:
    canvas = image.copy()
    status_color = (0, 180, 0) if result.status.upper() == "OK" else (0, 0, 255)
    banner_text = (
        f"{result.status} | defects={result.defect_count} | "
        f"{result.inference_ms:.1f} ms | {result.fps:.1f} FPS | {result.backend}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (24, 24, 24), -1)
    cv2.putText(
        canvas,
        banner_text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA,
    )

    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 144, 255), line_thickness)
        label = f"{detection.label} {detection.confidence:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_width, text_height = text_size
        top = max(45, y1 - text_height - 10)
        cv2.rectangle(
            canvas,
            (x1, top),
            (x1 + text_width + 10, top + text_height + 8),
            (30, 144, 255),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 5, top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def encode_image_to_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("图像编码失败。")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
