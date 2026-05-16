import sys
import os
import cv2
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QMessageBox, QProgressBar)
from PySide6.QtCore import QTimer, Qt, QThread, QRunnable, QThreadPool
from PySide6.QtGui import QKeyEvent, QImage, QPixmap


import module_cameras_5goki as cam_ctr
from module_yolo_csv3 import ImageProcessor  # 既存の画像処理ロジックを利用

RPI_URL = "http://192.168.2.1:5000"
TARGET_SAMPLE_COUNT = 15  # 自動で撮影するサクランボの個数

# ==========================================================
# 非同期ネットワークリクエスト用
# ==========================================================
class NetworkWorker(QRunnable):
    def __init__(self, endpoint):
        super().__init__()
        self.endpoint = endpoint
    def run(self):
        try:
            requests.get(f"{RPI_URL}{self.endpoint}", timeout=2)
        except Exception as e:
            print(f"RPi Communication Error: {e}")

# ==========================================================
# HSVキャリブレーション・解析クラス（改良版）
# ==========================================================
class HsvAnalyzer:
    def __init__(self):
        self.samples_h = []
        self.samples_s = []
        self.samples_v = []
        self.captured_count = 0

    def add_sample(self, frame, target_info):
        """
        ImageProcessor.get_target_info の結果を用いて
        サクランボの領域からのみ色を抽出する
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        stat = target_info['stat']
        x, y, w, h, area = stat
        
        # サクランボの矩形領域をROIとして切り出し
        roi_hsv = hsv[y:y+h, x:x+w]
        # 彩度が低い背景成分（白・灰）を除外
        mask = roi_hsv[:, :, 1] > 70
        cherry_pixels = roi_hsv[mask]

        if len(cherry_pixels) > 50:
            self.samples_h.extend(cherry_pixels[:, 0].tolist())
            self.samples_s.extend(cherry_pixels[:, 1].tolist())
            self.samples_v.extend(cherry_pixels[:, 2].tolist())
            self.captured_count += 1
            return True
        return False

    def analyze_and_save(self, save_img_path="hsv_histogram.png", config_path="hsv_config.json"):
        if not self.samples_h: return None

        h_adj = np.where(np.array(self.samples_h) > 150, np.array(self.samples_h) - 180, np.array(self.samples_h))
        k = 2.0
        h_mean, h_std = np.mean(h_adj), np.std(h_adj)
        s_mean, s_std = np.mean(self.samples_s), np.std(self.samples_s)
        v_mean, v_std = np.mean(self.samples_v), np.std(self.samples_v)

        h_min, h_max = h_mean - k*h_std, h_mean + k*h_std
        s_min = max(60, int(s_mean - k*s_std))
        v_min = max(50, int(v_mean - k*v_std))

        config = {
            "lower1": [0, s_min, v_min],
            "upper1": [int(max(0, h_max)), 255, 255],
            "lower2": [int(min(180, 180 + h_min)), s_min, v_min],
            "upper2": [180, 255, 255]
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        self._plot_histogram(h_adj, self.samples_s, self.samples_v, h_min, h_max, s_min, v_min, save_img_path)
        return config

    def _plot_histogram(self, h, s, v, h_min, h_max, s_min, v_min, path):
        plt.figure(figsize=(12, 4))
        titles = ["Hue (Adjusted)", "Saturation", "Value"]
        data = [h, s, v]
        ranges = [(h_min, h_max), (s_min, 255), (v_min, 255)]
        colors = ['red', 'green', 'blue']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.hist(data[i], bins=180 if i==0 else 255, color=colors[i], alpha=0.7)
            if i == 0:
                plt.axvspan(ranges[i][0], ranges[i][1], color='gray', alpha=0.3)
            else:
                plt.axvline(ranges[i][0], color='black', linestyle='--')
            plt.title(titles[i])
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

# ==========================================================
# キャリブレーションGUI
# ==========================================================
class CalibrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cherry Auto Calibration")
        self.thread_pool = QThreadPool()
        self.analyzer = HsvAnalyzer()
        
        self.is_auto_sampling = False
        self.cooldown_frames = 0
        
        # カメラ初期化
        self.cameras = cam_ctr.CameraManager()
        self.cameras.init_cameras()
        self.cameras.start_all_get_frame()

        # UI
        layout = QVBoxLayout()
        self.img_label = QLabel("Waiting for Camera...")
        self.img_label.setFixedSize(640, 480)
        layout.addWidget(self.img_label)

        self.progress = QProgressBar()
        self.progress.setMaximum(TARGET_SAMPLE_COUNT)
        layout.addWidget(self.progress)

        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        self.btn_auto = QPushButton("Start Auto Calibration")
        self.btn_auto.clicked.connect(self.toggle_auto)
        self.btn_save = QPushButton("Save & Finish")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.finish_calibration)
        
        btn_layout.addWidget(self.btn_auto)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

    def toggle_auto(self):
        if not self.is_auto_sampling:
            self.is_auto_sampling = True
            self.btn_auto.setText("Stop (Emergency)")
            self.status_label.setText("Status: Rotating & Sampling...")
            # 回転開始コマンド送信
            self.thread_pool.start(NetworkWorker("/rotate"))
        else:
            self.stop_auto()

    def stop_auto(self):
        self.is_auto_sampling = False
        self.btn_auto.setText("Start Auto Calibration")
        self.status_label.setText("Status: Stopped")
        self.thread_pool.start(NetworkWorker("/stop"))

    def update_frame(self):
        # cam_topを使用して解析
        controller = next((c for c in self.cameras.controllers if c.name == "cam_top"), self.cameras.controllers[0])
        frame = controller.get_current_frame()
        
        if frame is not None:
            # サクランボ検出（module_yolo_csv3 のロジックを使用）
            target = ImageProcessor.get_target_info(frame)
            
            # 自動サンプリングロジック
            if self.is_auto_sampling and target:
                # 画面中央付近(x偏差100以内)にあり、前回の撮影から20フレーム以上経過している場合
                center_x = frame.shape[1] // 2
                if abs(target['mx'] - center_x) < 80 and self.cooldown_frames == 0:
                    if self.analyzer.add_sample(frame, target):
                        self.progress.setValue(self.analyzer.captured_count)
                        self.cooldown_frames = 20 # 同じサクランボを連続撮影しないためのクールダウン
                        
                        if self.analyzer.captured_count >= TARGET_SAMPLE_COUNT:
                            self.stop_auto()
                            self.btn_save.setEnabled(True)
                            QMessageBox.information(self, "Done", "Sampling finished! Please check the results.")

            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1

            # 表示用
            display_frame = frame.copy()
            if target:
                cv2.circle(display_frame, (target['mx'], target['my']), 10, (0, 255, 0), -1)
            
            rgb = cv2.cvtColor(cv2.resize(display_frame, (640, 480)), cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, 640, 480, 640*3, QImage.Format_RGB888)
            self.img_label.setPixmap(QPixmap.fromImage(qimg))

    def finish_calibration(self):
        res = self.analyzer.analyze_and_save()
        if res:
            QMessageBox.information(self, "Success", "HSV Config & Histogram saved!\nNow you can run the main program.")
            self.cameras.stop_all_get_frame()
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CalibrationWindow()
    win.show()
    sys.exit(app.exec())