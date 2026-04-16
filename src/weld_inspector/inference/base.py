from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import ModelSettings
from ..schemas import Detection


class DetectionBackend(ABC):
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    @property
    def backend_name(self) -> str:
        return self.settings.effective_backend

    @abstractmethod
    def predict(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        return None

