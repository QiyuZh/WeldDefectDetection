from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from ..config import infer_backend_from_model, load_app_config, save_app_config
from ..detector import InspectionEngine
from ..schemas import FrameResult, format_result_summary


class CaptureThread(QThread):
    frame_ready = Signal(object, object)
    message = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        engine: InspectionEngine,
        source: str,
        source_kind: str,
        camera_index: int = 0,
        loop_video: bool = False,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.source = source
        self.source_kind = source_kind
        self.camera_index = camera_index
        self.loop_video = loop_video
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:  # pragma: no cover - UI 线程逻辑
        capture = None
        try:
            if self.source_kind == "video":
                capture = cv2.VideoCapture(self.source)
            elif self.source_kind == "camera":
                capture = cv2.VideoCapture(self.camera_index)
            else:
                raise ValueError(f"不支持的流类型: {self.source_kind}")

            if not capture.isOpened():
                raise RuntimeError(f"无法打开输入源: {self.source or self.camera_index}")

            frame_id = 0
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    if self.loop_video and self.source_kind == "video":
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.message.emit("视频流结束。")
                    break

                source_name = self.source if self.source_kind == "video" else f"camera:{self.camera_index}"
                result, annotated = self.engine.infer(frame, source_name=source_name, frame_id=frame_id)
                self.frame_ready.emit(annotated, result)
                frame_id += 1
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()


class MainWindow(QMainWindow):
    def __init__(self, config_path: str = "configs/app.yaml") -> None:
        super().__init__()
        self.config_path = config_path
        self.config = load_app_config(config_path)
        self.engine = InspectionEngine(self.config)
        self.capture_thread: CaptureThread | None = None
        self.setWindowTitle("YOLOv8 焊缝质检系统")
        self.resize(1480, 920)
        self._build_ui()
        self._sync_form_from_config()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        control_group = QGroupBox("运行控制")
        control_layout = QGridLayout(control_group)
        layout.addWidget(control_group)

        self.config_label = QLabel(self.config_path)
        self.model_path_edit = QLineEdit()
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setReadOnly(True)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["auto", "ultralytics", "onnx", "tensorrt"])
        self.inference_mode_combo = QComboBox()
        self.inference_mode_combo.addItem("彩色原图", "color")
        self.inference_mode_combo.addItem("灰度增强", "grayscale_weld")
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 8)

        self.reload_button = QPushButton("重载模型")
        self.select_model_button = QPushButton("选择模型")
        self.image_button = QPushButton("打开图片")
        self.video_button = QPushButton("打开视频")
        self.camera_button = QPushButton("启动摄像头")
        self.stop_button = QPushButton("停止")

        control_layout.addWidget(QLabel("配置文件"), 0, 0)
        control_layout.addWidget(self.config_label, 0, 1, 1, 3)
        control_layout.addWidget(QLabel("模型路径"), 1, 0)
        control_layout.addWidget(self.model_path_edit, 1, 1, 1, 2)
        control_layout.addWidget(self.select_model_button, 1, 3)
        control_layout.addWidget(QLabel("后端"), 2, 0)
        control_layout.addWidget(self.backend_combo, 2, 1)
        control_layout.addWidget(self.reload_button, 2, 2)
        control_layout.addWidget(QLabel("摄像头索引"), 2, 3)
        control_layout.addWidget(self.camera_index_spin, 2, 4)
        control_layout.addWidget(QLabel("推理模式"), 3, 0)
        control_layout.addWidget(self.inference_mode_combo, 3, 1)
        control_layout.addWidget(QLabel("当前输入"), 4, 0)
        control_layout.addWidget(self.source_path_edit, 4, 1, 1, 4)

        button_row = QHBoxLayout()
        button_row.addWidget(self.image_button)
        button_row.addWidget(self.video_button)
        button_row.addWidget(self.camera_button)
        button_row.addWidget(self.stop_button)
        control_layout.addLayout(button_row, 5, 0, 1, 5)

        content_row = QHBoxLayout()
        layout.addLayout(content_row, 1)

        self.preview_label = QLabel("请选择图片、视频或摄像头开始检测")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(960, 640)
        self.preview_label.setStyleSheet("background-color:#111; color:#ddd; border:1px solid #444;")
        content_row.addWidget(self.preview_label, 3)

        side_panel = QVBoxLayout()
        content_row.addLayout(side_panel, 2)

        stat_group = QGroupBox("检测状态")
        stat_layout = QGridLayout(stat_group)
        side_panel.addWidget(stat_group)

        self.status_value = QLabel("待机")
        self.backend_value = QLabel(self.engine.backend_name)
        self.preprocess_value = QLabel()
        self.fps_value = QLabel("0.0")
        self.defect_value = QLabel("0")
        stat_layout.addWidget(QLabel("状态"), 0, 0)
        stat_layout.addWidget(self.status_value, 0, 1)
        stat_layout.addWidget(QLabel("后端"), 1, 0)
        stat_layout.addWidget(self.backend_value, 1, 1)
        stat_layout.addWidget(QLabel("推理模式"), 2, 0)
        stat_layout.addWidget(self.preprocess_value, 2, 1)
        stat_layout.addWidget(QLabel("FPS"), 3, 0)
        stat_layout.addWidget(self.fps_value, 3, 1)
        stat_layout.addWidget(QLabel("缺陷数"), 4, 0)
        stat_layout.addWidget(self.defect_value, 4, 1)

        self.result_box = QPlainTextEdit()
        self.result_box.setReadOnly(True)
        side_panel.addWidget(self.result_box, 1)

        self.select_model_button.clicked.connect(self.select_model)
        self.reload_button.clicked.connect(self.reload_model)
        self.inference_mode_combo.currentIndexChanged.connect(self.apply_inference_mode)
        self.image_button.clicked.connect(self.open_image)
        self.video_button.clicked.connect(self.open_video)
        self.camera_button.clicked.connect(self.open_camera)
        self.stop_button.clicked.connect(self.stop_stream)

    def _sync_form_from_config(self) -> None:
        self.model_path_edit.setText(self.config.model.model_path)
        self.backend_combo.setCurrentText(self.config.model.backend)
        self.camera_index_spin.setValue(self.config.source.camera_index)
        self.backend_value.setText(self.engine.backend_name)
        self._sync_inference_mode_from_config()

    def _active_inference_mode(self) -> str:
        if self.config.inference_preprocess.is_active:
            return self.config.inference_preprocess.mode
        return "color"

    def _sync_inference_mode_from_config(self) -> None:
        active_mode = self._active_inference_mode()
        target_index = self.inference_mode_combo.findData(active_mode)
        if target_index < 0:
            target_index = self.inference_mode_combo.findData("color")

        self.inference_mode_combo.blockSignals(True)
        self.inference_mode_combo.setCurrentIndex(target_index)
        self.inference_mode_combo.blockSignals(False)
        self.preprocess_value.setText(self.inference_mode_combo.currentText())

    def apply_inference_mode(self, _index: int | None = None) -> None:
        previous_enabled = self.config.inference_preprocess.enabled
        previous_mode = self.config.inference_preprocess.mode
        selected_mode = self.inference_mode_combo.currentData() or "color"
        if selected_mode == "color":
            self.config.inference_preprocess.enabled = False
            self.config.inference_preprocess.mode = "color"
        else:
            self.config.inference_preprocess.enabled = True
            self.config.inference_preprocess.mode = selected_mode

        try:
            save_app_config(self.config, self.config_path)
        except Exception as exc:
            self.config.inference_preprocess.enabled = previous_enabled
            self.config.inference_preprocess.mode = previous_mode
            self._sync_inference_mode_from_config()
            QMessageBox.critical(self, "推理模式切换失败", str(exc))
            return

        self.preprocess_value.setText(self.inference_mode_combo.currentText())
        self._append_message(f"推理模式已切换: {self.inference_mode_combo.currentText()}")

    def select_model(self) -> None:
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型",
            str(Path(self.config.model.model_path).parent),
            "Model (*.pt *.onnx *.trt *.engine)",
        )
        if not model_path:
            return
        self.model_path_edit.setText(model_path)
        if self.backend_combo.currentText() == "auto":
            self.backend_value.setText(infer_backend_from_model(model_path))

    def reload_model(self) -> None:
        try:
            model_path = self.model_path_edit.text().strip()
            backend = self.backend_combo.currentText()
            self.engine.reload_model(model_path=model_path, backend=backend)
            self.config.model.model_path = model_path
            self.config.model.backend = backend
            save_app_config(self.config, self.config_path)
            self.backend_value.setText(self.engine.backend_name)
            self._append_message(f"模型已重载: {model_path}")
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def open_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Image (*.jpg *.jpeg *.png *.bmp)",
        )
        if not image_path:
            return
        self.stop_stream()
        self._ensure_model_synced()
        self.source_path_edit.setText(image_path)
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.critical(self, "读取失败", f"无法读取图片: {image_path}")
            return
        try:
            result, annotated = self.engine.infer(image, source_name=image_path, frame_id=0)
        except Exception as exc:
            QMessageBox.critical(self, "推理失败", str(exc))
            return
        self._display_result(annotated, result)

    def open_video(self) -> None:
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "Video (*.mp4 *.avi *.mov *.mkv)",
        )
        if not video_path:
            return
        self.source_path_edit.setText(video_path)
        self._start_stream(source=video_path, source_kind="video")

    def open_camera(self) -> None:
        self.source_path_edit.setText(f"camera:{self.camera_index_spin.value()}")
        self._start_stream(source="", source_kind="camera")

    def _start_stream(self, source: str, source_kind: str) -> None:
        self.stop_stream()
        self._ensure_model_synced()
        self.capture_thread = CaptureThread(
            engine=self.engine,
            source=source,
            source_kind=source_kind,
            camera_index=self.camera_index_spin.value(),
            loop_video=self.config.source.loop_video,
        )
        self.capture_thread.frame_ready.connect(self._display_result)
        self.capture_thread.message.connect(self._append_message)
        self.capture_thread.failed.connect(self._show_stream_error)
        self.capture_thread.start()
        self._append_message(f"开始检测: {self.source_path_edit.text()}")

    def _ensure_model_synced(self) -> None:
        current_model = self.model_path_edit.text().strip()
        current_backend = self.backend_combo.currentText()
        if (
            current_model != self.config.model.model_path
            or current_backend != self.config.model.backend
        ):
            self.reload_model()

    def stop_stream(self) -> None:
        if self.capture_thread is None:
            return
        self.capture_thread.stop()
        self.capture_thread.wait(1500)
        self.capture_thread = None
        self._append_message("检测已停止。")

    def _display_result(self, frame, result: FrameResult) -> None:
        self._show_frame(frame)
        self.status_value.setText(result.status)
        self.backend_value.setText(result.backend)
        self.fps_value.setText(f"{result.fps:.1f}")
        self.defect_value.setText(str(result.defect_count))
        self.result_box.setPlainText(format_result_summary(result))

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def _append_message(self, message: str) -> None:
        current = self.result_box.toPlainText().strip()
        if current:
            self.result_box.setPlainText(f"{current}\n\n{message}")
        else:
            self.result_box.setPlainText(message)

    def _show_stream_error(self, message: str) -> None:
        self.stop_stream()
        QMessageBox.critical(self, "流检测失败", message)

    def closeEvent(self, event) -> None:  # pragma: no cover
        self.stop_stream()
        self.engine.close()
        return super().closeEvent(event)
