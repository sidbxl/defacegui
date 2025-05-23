import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QListWidget, QCheckBox, QComboBox, QLineEdit, QProgressBar, QGroupBox, QMessageBox, QListWidgetItem, QDialog, QSpinBox, QDoubleSpinBox, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import imageio
import cv2
from deface.deface_main import get_anonymized_image, get_file_type, video_detect
from deface.centerface import CenterFace

if getattr(sys, 'frozen', False):
    # Running in a bundle
    base_path = sys._MEIPASS
else:
    # Running in normal Python
    base_path = os.path.dirname(__file__)

default_onnx_path = os.path.join(base_path, 'centerface.onnx')

class ImagePreviewDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Preview - Anonymized Image')
        vbox = QVBoxLayout()
        label = QLabel()
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            label.setText('Could not load image.')
        vbox.addWidget(label)
        self.setLayout(vbox)
        self.resize(520, 520)

class Worker(QThread):
    progress_update = pyqtSignal(int, int)  # file_idx, percent
    file_done = pyqtSignal(str, str, bool)  # file_path, output_path, success
    error = pyqtSignal(str)
    all_done = pyqtSignal(list, list)  # successes, failures

    def __init__(self, files, out_dir, settings, parent=None):
        super().__init__(parent)
        self.files = files
        self.out_dir = out_dir
        self.settings = settings
        self._is_running = True
        self._cancel_current = False
        self.successes = []
        self.failures = []

    def get_unique_out_path(self, base_path):
        if not os.path.exists(base_path):
            return base_path
        root, ext = os.path.splitext(base_path)
        i = 1
        while True:
            new_path = f"{root}_{i}{ext}"
            if not os.path.exists(new_path):
                return new_path
            i += 1

    def cancel_current(self):
        self._cancel_current = True

    def cancel_all(self):
        self._is_running = False
        self._cancel_current = True

    def run(self):
        import imageio  # ensure always imported
        try:
            for idx, file_path in enumerate(self.files):
                if not self._is_running:
                    break
                self._cancel_current = False
                ext = os.path.splitext(file_path)[1].lower()
                filetype = get_file_type(file_path)
                out_name = os.path.basename(file_path)
                out_path = os.path.join(self.out_dir, f"anonymized_{out_name}")
                out_path = self.get_unique_out_path(out_path)
                try:
                    if filetype == 'image':
                        img = imageio.imread(file_path)
                        mask_type = self.settings['mask_type'].lower()
                        ellipse = not self.settings['use_box'] and (mask_type != 'solid' and mask_type != 'mosaic')
                        mask_scale = self.settings['mask_scale']
                        threshold = self.settings['threshold']
                        replaceimg = None
                        mosaicsize = self.settings['mosaicsize']
                        keep_metadata = not self.settings['erase_metadata']
                        result = get_anonymized_image(
                            img,
                            threshold=threshold,
                            replacewith=mask_type,
                            mask_scale=mask_scale,
                            ellipse=ellipse,
                            draw_scores=False,
                            replaceimg=replaceimg,
                            mosaicsize=mosaicsize
                        )
                        if keep_metadata:
                            try:
                                import imageio.v3
                                metadata = imageio.v3.immeta(file_path)
                                exif_dict = metadata.get("exif", None)
                                imageio.imwrite(out_path, result, exif=exif_dict)
                            except Exception:
                                imageio.imwrite(out_path, result)
                        else:
                            imageio.imwrite(out_path, result)
                        self.progress_update.emit(idx, 100)
                        self.file_done.emit(file_path, out_path, True)
                    elif filetype == 'video':
                        try:
                            centerface = CenterFace(in_shape=None, backend=self.settings['backend'], override_execution_provider=self.settings['exec_provider'] or None)
                        except Exception as e:
                            if 'onnxruntime' in str(e) or 'DLL load failed' in str(e):
                                self.error.emit('ONNX Runtime error: GPU/onnxrt mode is not available. Please check your CUDA/cuDNN/onnxruntime-gpu installation or use CPU mode.')
                                centerface = CenterFace(in_shape=None, backend='opencv')
                            else:
                                self.error.emit(f'Failed to initialize backend: {e}')
                                self.failures.append(file_path)
                                continue
                        ffmpeg_config = {"codec": "libx264"}
                        nframes = None
                        try:
                            import imageio.plugins.ffmpeg
                            reader = imageio.get_reader(file_path)
                            nframes = reader.count_frames()
                            reader.close()
                        except Exception:
                            pass
                        def progress_callback(frames_done):
                            if nframes:
                                percent = int(100 * frames_done / nframes)
                                self.progress_update.emit(idx, percent)
                            if self._cancel_current:
                                raise Exception('Cancelled by user')
                        try:
                            video_detect(
                                ipath=file_path,
                                opath=out_path,
                                centerface=centerface,
                                threshold=self.settings['threshold'],
                                enable_preview=False,
                                cam=False,
                                nested=False,
                                replacewith=self.settings['mask_type'].lower(),
                                mask_scale=self.settings['mask_scale'],
                                ellipse=not self.settings['use_box'] and (self.settings['mask_type'].lower() != 'solid' and self.settings['mask_type'].lower() != 'mosaic'),
                                draw_scores=False,
                                ffmpeg_config=ffmpeg_config,
                                replaceimg=None,
                                keep_audio=self.settings['keep_audio'],
                                mosaicsize=self.settings['mosaicsize'],
                                disable_progress_output=True,
                                progress_callback=progress_callback
                            )
                            self.progress_update.emit(idx, 100)
                            self.file_done.emit(file_path, out_path, True)
                        except Exception as e:
                            self.error.emit(f"Video processing failed for {file_path}: {e}")
                            self.failures.append(file_path)
                            continue
                    else:
                        self.error.emit(f"Unsupported file type: {file_path}")
                        self.failures.append(file_path)
                        continue
                    self.successes.append(file_path)
                except Exception as e:
                    self.error.emit(f"Processing failed for {file_path}: {e}")
                    self.failures.append(file_path)
            self.all_done.emit(self.successes, self.failures)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False
        self._cancel_current = True

class DefaceGUIMain(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DefaceGUI')
        self.setGeometry(100, 100, 1100, 600)
        self.worker = None
        self.backend_label = QLabel()
        self.cancelled = False
        self.init_ui()
        self.detect_backend()

    def init_ui(self):
        main_layout = QVBoxLayout()
        columns_layout = QHBoxLayout()

        # --- Left column: Input/Output ---
        left_col = QVBoxLayout()
        file_group = QGroupBox('Input Files')
        file_layout = QVBoxLayout()
        self.file_list = QListWidget()
        add_file_btn = QPushButton('Add Files')
        add_file_btn.clicked.connect(self.add_files)
        remove_file_btn = QPushButton('Remove Selected')
        remove_file_btn.clicked.connect(self.remove_selected_files)
        file_layout.addWidget(self.file_list)
        file_layout.addWidget(add_file_btn)
        file_layout.addWidget(remove_file_btn)
        file_group.setLayout(file_layout)
        left_col.addWidget(file_group)

        out_group = QGroupBox('Output Directory')
        out_layout = QHBoxLayout()
        self.out_dir_edit = QLineEdit()
        out_dir_btn = QPushButton('Browse')
        out_dir_btn.clicked.connect(self.select_output_dir)
        out_layout.addWidget(self.out_dir_edit)
        out_layout.addWidget(out_dir_btn)
        out_group.setLayout(out_layout)
        left_col.addWidget(out_group)
        left_col.addStretch(1)

        # --- Middle column: Settings ---
        middle_col = QVBoxLayout()
        settings_group = QGroupBox('Settings')
        settings_layout = QVBoxLayout()
        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItems(['Blur', 'Solid', 'Mosaic', 'None'])
        self.mask_type_combo.currentTextChanged.connect(self.on_mask_type_changed)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.2)
        self.box_checkbox = QCheckBox('Use box mask instead of ellipse')
        self.keep_audio_checkbox = QCheckBox('Keep audio (videos)')
        self.keep_audio_checkbox.setChecked(True)
        self.mosaic_spin = QSpinBox()
        self.mosaic_spin.setRange(2, 200)
        self.mosaic_spin.setValue(20)
        self.mosaic_spin.setEnabled(False)
        self.erase_metadata_checkbox = QCheckBox('Erase metadata (EXIF, etc.)')
        self.erase_metadata_checkbox.setChecked(True)
        self.remove_after_checkbox = QCheckBox('Remove files from list after processing')
        self.remove_after_checkbox.setChecked(True)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(['auto', 'onnxrt', 'opencv'])
        self.backend_combo.currentTextChanged.connect(self.update_backend_label)
        self.exec_provider_edit = QLineEdit()
        self.exec_provider_edit.setPlaceholderText('Execution provider (optional)')

        settings_layout.addWidget(QLabel('Mask Type:'))
        settings_layout.addWidget(self.mask_type_combo)
        settings_layout.addWidget(QLabel('Detection Threshold:'))
        settings_layout.addWidget(self.threshold_spin)
        settings_layout.addWidget(self.box_checkbox)
        settings_layout.addWidget(self.keep_audio_checkbox)
        self.mosaic_label = QLabel('Mosaic size:')
        mosaic_hbox = QHBoxLayout()
        mosaic_hbox.addWidget(self.mosaic_label)
        mosaic_hbox.addWidget(self.mosaic_spin)
        self.mosaic_hbox = mosaic_hbox
        settings_layout.addLayout(mosaic_hbox)
        settings_layout.addWidget(self.erase_metadata_checkbox)
        settings_layout.addWidget(self.remove_after_checkbox)
        settings_layout.addWidget(QLabel('Backend:'))
        settings_layout.addWidget(self.backend_combo)
        settings_layout.addWidget(self.exec_provider_edit)

        # Advanced settings toggle
        self.advanced_group = QGroupBox('Advanced Settings')
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        self.advanced_group.setLayout(settings_layout)
        middle_col.addWidget(self.advanced_group)
        self.advanced_group.toggled.connect(self.toggle_advanced_settings)
        self.toggle_advanced_settings(False)
        middle_col.addWidget(self.backend_label)
        middle_col.addStretch(1)

        # --- Right column: (empty, no preview) ---
        right_col = QVBoxLayout()
        right_col.addStretch(1)

        # Add columns to main layout
        columns_layout.addLayout(left_col, 2)
        columns_layout.addLayout(middle_col, 2)
        columns_layout.addLayout(right_col, 3)
        main_layout.addLayout(columns_layout)

        # --- Progress bars and buttons at the bottom ---
        bottom_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat('Total Progress: %p%')
        self.file_progress = QProgressBar()
        self.file_progress.setValue(0)
        self.file_progress.setFormat('Current File: %p%')
        bottom_layout.addWidget(self.progress, 2)
        bottom_layout.addWidget(self.file_progress, 2)

        # Start and Cancel buttons
        self.start_btn = QPushButton('Start')
        self.start_btn.setStyleSheet('font-size: 18px; font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;')
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_current_btn = QPushButton('Cancel Current')
        self.cancel_current_btn.clicked.connect(self.cancel_current)
        self.cancel_all_btn = QPushButton('Cancel All')
        self.cancel_all_btn.clicked.connect(self.cancel_all)
        self.cancel_current_btn.setEnabled(True)
        self.cancel_all_btn.setEnabled(True)
        bottom_layout.addWidget(self.start_btn, 1)
        bottom_layout.addWidget(self.cancel_current_btn)
        bottom_layout.addWidget(self.cancel_all_btn)

        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

    def toggle_advanced_settings(self, checked):
        self.advanced_group.setFlat(not checked)
        for i in range(self.advanced_group.layout().count()):
            item = self.advanced_group.layout().itemAt(i)
            widget = item.widget()
            if widget:
                widget.setVisible(checked)
            # Special handling for mosaic_hbox
            if hasattr(self, 'mosaic_hbox') and item.layout() == self.mosaic_hbox:
                self.mosaic_label.setVisible(checked)
                self.mosaic_spin.setVisible(checked)
        self.advanced_group.setVisible(True)

    def on_mask_type_changed(self, text):
        self.mosaic_spin.setEnabled(text.lower() == 'mosaic')

    def detect_backend(self):
        try:
            backend = self.backend_combo.currentText()
            exec_provider = self.exec_provider_edit.text() or None
            centerface = CenterFace(in_shape=None, backend=backend, override_execution_provider=exec_provider)
            if centerface.backend == 'onnxrt' and hasattr(centerface, 'sess'):
                provider = centerface.sess.get_providers()[0]
                self.backend_label.setText(f'Backend: ONNX Runtime (GPU mode: {provider})')
            elif centerface.backend == 'opencv':
                self.backend_label.setText('Backend: OpenCV (CPU mode)')
            else:
                self.backend_label.setText(f'Backend: {centerface.backend}')
        except Exception as e:
            self.backend_label.setText(f'Backend: Error ({e})')

    def update_backend_label(self):
        self.detect_backend()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select Images or Videos', '',
                                                'Images/Videos (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv);;All Files (*)')
        for f in files:
            if not any(self.file_list.item(i).text() == f for i in range(self.file_list.count())):
                self.file_list.addItem(f)

    def remove_selected_files(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def select_output_dir(self):
        dir_ = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if dir_:
            self.out_dir_edit.setText(dir_)

    def start_processing(self):
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        out_dir = self.out_dir_edit.text()
        if not files:
            QMessageBox.warning(self, 'No files', 'Please add at least one file to process.')
            return
        if not out_dir:
            QMessageBox.warning(self, 'No output directory', 'Please select an output directory.')
            return
        self.start_btn.setEnabled(False)
        self.progress.setValue(0)
        self.file_progress.setValue(0)
        self.cancelled = False
        settings = {
            'mask_type': self.mask_type_combo.currentText(),
            'mask_scale': 1.3,  # could add to GUI
            'threshold': self.threshold_spin.value(),
            'mosaicsize': self.mosaic_spin.value(),
            'use_box': self.box_checkbox.isChecked(),
            'keep_audio': self.keep_audio_checkbox.isChecked(),
            'erase_metadata': self.erase_metadata_checkbox.isChecked(),
            'remove_after': self.remove_after_checkbox.isChecked(),
            'backend': self.backend_combo.currentText(),
            'exec_provider': self.exec_provider_edit.text() or None,
        }
        self.worker = Worker(files, out_dir, settings, self)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.error.connect(self.on_error)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.start()
        self.total_files = len(files)
        self.files_done = 0

    def cancel_current(self):
        if self.worker:
            self.worker.cancel_current()
        self.cancelled = True

    def cancel_all(self):
        if self.worker:
            self.worker.cancel_all()
        self.cancelled = True

    def on_progress_update(self, file_idx, percent):
        self.file_progress.setValue(percent)
        self.progress.setValue(int((file_idx / self.total_files) * 100))

    def on_file_done(self, file_path, out_path, success):
        self.files_done += 1
        self.progress.setValue(int((self.files_done / self.total_files) * 100))
        self.file_progress.setValue(0)
        if self.remove_after_checkbox.isChecked():
            items = self.file_list.findItems(file_path, Qt.MatchExactly)
            for item in items:
                self.file_list.takeItem(self.file_list.row(item))

    def on_error(self, msg):
        QMessageBox.critical(self, 'Error', msg)
        self.start_btn.setEnabled(True)
        self.cancel_current_btn.setEnabled(True)
        self.cancel_all_btn.setEnabled(True)

    def on_all_done(self, successes, failures):
        self.progress.setValue(100)
        self.file_progress.setValue(0)
        self.start_btn.setEnabled(True)
        self.cancel_current_btn.setEnabled(True)
        self.cancel_all_btn.setEnabled(True)
        if self.cancelled:
            msg = f"Processing cancelled.\n\nSuccesses: {len(successes)}\nFailures: {len(failures)}"
        else:
            msg = f"All files processed!\n\nSuccesses: {len(successes)}\nFailures: {len(failures)}"
        if failures:
            msg += f"\nFailed files:\n" + '\n'.join(failures)
        QMessageBox.information(self, 'Done', msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DefaceGUIMain()
    window.show()
    sys.exit(app.exec_()) 