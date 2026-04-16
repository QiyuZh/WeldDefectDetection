from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import load_app_config
from .detector import InspectionEngine
from .utils.vision import encode_image_to_base64


def create_app(config_path: str | Path = "configs/app.yaml") -> FastAPI:
    app = FastAPI(title="Weld QC API", version="0.1.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        config = load_app_config(config_path)
        app.state.config = config
        app.state.engine = InspectionEngine(config)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        engine: InspectionEngine | None = getattr(app.state, "engine", None)
        if engine is not None:
            engine.close()

    @app.get("/health")
    async def health() -> dict[str, object]:
        engine: InspectionEngine = app.state.engine
        return {
            "status": "ok",
            "backend": engine.backend_name,
            "model_path": engine.config.model.model_path,
        }

    @app.post("/reload-model")
    async def reload_model(model_path: str | None = None, backend: str | None = None) -> dict[str, object]:
        engine: InspectionEngine = app.state.engine
        try:
            engine.reload_model(model_path=model_path, backend=backend)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "reloaded", "backend": engine.backend_name, "model_path": engine.config.model.model_path}

    @app.post("/infer-image")
    async def infer_image(file: UploadFile = File(...)) -> dict[str, object]:
        data = await file.read()
        np_buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="上传文件不是有效图像。")

        engine: InspectionEngine = app.state.engine
        result, annotated = engine.infer(image, source_name=file.filename or "upload", frame_id=0)
        return {
            "result": result.to_dict(),
            "annotated_image_base64": encode_image_to_base64(annotated),
        }

    return app

