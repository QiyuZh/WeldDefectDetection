from __future__ import annotations

from pathlib import Path


class HikRobotCamera:
    """
    海康工业相机扩展点。

    当前仓库默认交付图片 / 视频 / USB 摄像头链路，工业相机接入需要在目标环境
    安装 MVS SDK，并根据现场型号补齐 ctypes 或官方 Python 绑定。
    """

    def __init__(self, sdk_path: str | None = None) -> None:
        self.sdk_path = Path(sdk_path) if sdk_path else None

    def open(self) -> None:
        raise NotImplementedError(
            "海康工业相机接口为预留扩展点。请在已安装 MVS SDK 的环境中，"
            "基于该类补齐枚举设备、取流和释放资源逻辑。"
        )

