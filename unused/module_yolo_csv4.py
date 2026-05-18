import cv2
import numpy as np
import datetime
import os
import csv
from ultralytics import YOLO

# ==========================================================
# 定数定義
# ==========================================================
USE_CROP = False
CENTER_THRESHOLD_X = 100

MODEL_PATH = "Trained_Models/best2.pt"
YOLO_IMG_SIZE = 640
CONF_THRESHOLD = 0.8
TWIN_STRICT_THRESHOLD = 0.93
UNRIPE_STRICT_THRESHOLD = 0.93

SAVE_DIR_VIDEO = "evaluated_videos"
SAVE_DIR_CSV = "evaluated_csv"
SAVE_DIR_IMG = "evaluated_images"
FPS = 20.0
TILE_VIDEO_SIZE = (YOLO_IMG_SIZE, YOLO_IMG_SIZE)

# ==========================================================
# 判定結果データクラス
# ==========================================================
class YoloResult:
    def __init__(self, obj_id, label_name, confidence):
        self.id = obj_id
        self.label_name = label_name
        self.confidence = confidence

    def to_csv_row(self):
        return [self.id, self.label_name, f"{self.confidence:.2f}"]

# ==========================================================
# 画像保存・CSV出力クラス
# ==========================================================
class OutputLogger:
    def __init__(self):
        os.makedirs(SAVE_DIR_VIDEO, exist_ok=True)
        os.makedirs(SAVE_DIR_CSV, exist_ok=True)

        self.img_dirs = {}
        cam_names = ['cam_top', 'cam_under', 'cam_inside', 'cam_outside']
        for cam in cam_names:
            path = os.path.join(SAVE_DIR_IMG, cam)
            os.makedirs(path, exist_ok=True)
            self.img_dirs[cam] = path
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(SAVE_DIR_VIDEO, f"eval_{timestamp}.mp4")
        self.csv_path = os.path.join(SAVE_DIR_CSV, f"eval_{timestamp}.csv")
        
        self.video_writer = None
        self._init_csv()

    def _init_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, FPS, TILE_VIDEO_SIZE)

    def _init_csv(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "LabelName", "Confidence"])

    def write_video(self, tile_frame):
        if self.video_writer is None:
            self._init_video()
        self.video_writer.write(tile_frame)

    def write_csv(self, result_obj):
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(result_obj.to_csv_row())

    def write_image(self, cam_name, frame, obj_id):
        if cam_name in self.img_dirs:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{cam_name}_{timestamp}_id{obj_id:04d}.jpg"
            filepath = os.path.join(self.img_dirs[cam_name], filename)
            cv2.imwrite(filepath, frame)

    def close(self):
        if self.video_writer:
            self.video_writer.release()

# ==========================================================
# 画像処理ユーティリティクラス
# ==========================================================
class ImageProcessor:
    @staticmethod
    def get_target_info_list(frame):
        """全ての有効なターゲットをリストで返す"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 60, 50]), np.array([35, 255, 255])
        lower_red2, upper_red2 = np.array([160, 60, 50]), np.array([180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2), cv2.MORPH_CLOSE, kernel, iterations=2)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        targets = []
        h, w = frame.shape[:2]
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 500: continue
            s = stats[i]
            # 画面端に近いものは除外
            if (s[0] <= 5) or (s[1] <= 5) or ((s[0]+s[2]) >= (w-5)) or ((s[1]+s[3]) >= (h-5)): continue
            
            targets.append({
                'mx': int(centroids[i][0]), 
                'my': int(centroids[i][1]), 
                'area': area, 
                'stat': s
            })
        return targets

# ==========================================================
# YOLO検出クラス (マルチトラッキング対応)
# ==========================================================
class YoloDetector:
    def __init__(self, model_path=MODEL_PATH):
        print(f"YOLOモデル {model_path} をロード中...")
        self.model = YOLO(model_path)
        self.logger = OutputLogger()
        
        # 内部状態管理
        self.global_frame_count = 0
        self.active_tracks = {}  # { yolo_id: { 'serial_id': int, 'all_results': [], 'best_frames': {cam: (frame, conf)}, 'last_seen': int } }
        self.serial_id_counter = 1
        self.MAX_EMPTY_FRAMES = 8 # 何フレーム見失ったら終了とするか
        
        self.frame_buffer = {'cam_top': None, 'cam_under': None, 'cam_inside': None, 'cam_outside': None}

    def _resolve_best_result(self, detections, serial_id):
        if not detections: return None
        # 以前のロジックと同様に、不良品優先の判定
        healthy = [d for d in detections if d.label_name == "healthy"]
        damaged = []
        for d in detections:
            if d.label_name in ["healthy", "None"]: continue
            if d.label_name == "twin" and d.confidence < TWIN_STRICT_THRESHOLD: continue
            if d.label_name == "unripe" and d.confidence < UNRIPE_STRICT_THRESHOLD: continue
            damaged.append(d)

        best = None
        if healthy and damaged:
            best = max(healthy, key=lambda x: x.confidence) if max(healthy, key=lambda x: x.confidence).confidence >= 0.9 and max(damaged, key=lambda x: x.confidence).confidence < 0.9 else max(damaged, key=lambda x: x.confidence)
        elif damaged: best = max(damaged, key=lambda x: x.confidence)
        elif healthy: best = max(healthy, key=lambda x: x.confidence)
        else: best = max(detections, key=lambda x: x.confidence)
        
        # IDをシリアル番号に書き換えて返す
        return YoloResult(serial_id, best.label_name, best.confidence)

    def evaluate_frame(self, frame, cam_name, obj_id=None):
        targets = ImageProcessor.get_target_info_list(frame)
        
        # YOLO推論 (persist=True でトラッキング有効化)
        input_img = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
        results = self.model.track(input_img, persist=True, verbose=False, conf=CONF_THRESHOLD, tracker="bytetrack.yaml")
        
        annotated_frame = input_img.copy()
        
        # トラッキング中の各ボックスを処理
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls_idx, conf in zip(boxes, ids, clss, confs):
                label = self.model.names[cls_idx]
                
                # 新規個体の登録
                if track_id not in self.active_tracks:
                    self.active_tracks[track_id] = {
                        'serial_id': self.serial_id_counter,
                        'all_results': [],
                        'best_frames': {},
                        'last_seen': self.global_frame_count
                    }
                    self.serial_id_counter += 1
                
                track_info = self.active_tracks[track_id]
                track_info['last_seen'] = self.global_frame_count
                track_info['all_results'].append(YoloResult(track_info['serial_id'], label, conf))
                
                # カメラごとのベストショット更新
                if cam_name not in track_info['best_frames'] or conf > track_info['best_frames'][cam_name][1]:
                    track_info['best_frames'][cam_name] = (annotated_frame.copy(), conf)

                # 描画 (シリアルIDを表示)
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID:{track_info['serial_id']} {label}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self._buffer_frame(cam_name, annotated_frame)
        
        # タイムアウトした個体の確定処理
        finalized_list = []
        to_delete = []
        for tid, info in self.active_tracks.items():
            if self.global_frame_count - info['last_seen'] > self.MAX_EMPTY_FRAMES:
                best_overall = self._resolve_best_result(info['all_results'], info['serial_id'])
                if best_overall:
                    self.logger.write_csv(best_overall)
                    for c_name, (f_data, _) in info['best_frames'].items():
                        self.logger.write_image(c_name, f_data, best_overall.id)
                    finalized_list.append(best_overall)
                to_delete.append(tid)
        
        for tid in to_delete:
            del self.active_tracks[tid]

        # 戻り値の互換性維持（必要に応じてリストを返すように調整可能）
        return annotated_frame, None, finalized_list

    def _buffer_frame(self, cam_name, frame):
        self.frame_buffer[cam_name] = frame
        if all(f is not None for f in self.frame_buffer.values()):
            self.global_frame_count += 1
            h_left = np.vstack((self.frame_buffer['cam_inside'], self.frame_buffer['cam_under']))
            h_right = np.vstack((self.frame_buffer['cam_outside'], self.frame_buffer['cam_top']))
            tile = cv2.resize(np.hstack((h_left, h_right)), TILE_VIDEO_SIZE)
            self.logger.write_video(tile)
            for key in self.frame_buffer.keys(): self.frame_buffer[key] = None

    def close(self):
        # 残っている全個体を強制終了
        for tid, info in self.active_tracks.items():
            best_overall = self._resolve_best_result(info['all_results'], info['serial_id'])
            if best_overall:
                self.logger.write_csv(best_overall)
                for c_name, (f_data, _) in info['best_frames'].items():
                    self.logger.write_image(c_name, f_data, best_overall.id)
        self.logger.close()