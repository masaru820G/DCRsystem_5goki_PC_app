import cv2
import numpy as np
import datetime
import os
import csv
import json
from ultralytics import YOLO

# ==========================================================
# 定数定義
# ==========================================================
USE_CROP = False
CENTER_THRESHOLD_X = 100

MODEL_PATH = "Trained_Models/best2.pt"
YOLO_IMG_SIZE = 640
CONF_THRESHOLD = 0.75
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
    def __init__(self, obj_id, label_name, confidence, cam_name):
        self.id = obj_id
        self.label_name = label_name
        self.confidence = confidence
        self.cam_name = cam_name

    def to_csv_row(self):
        # Confidenceの次の列にCameraを追加
        return [self.id, self.label_name, f"{self.confidence:.2f}", self.cam_name]

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
            writer.writerow(["ID", "LabelName", "Confidence", "Camera"])

    def write_video(self, tile_frame):
        if self.video_writer is None:
            self._init_video()
        h, w = tile_frame.shape[:2]
        if (w, h) != TILE_VIDEO_SIZE:
            tile_frame = cv2.resize(tile_frame, TILE_VIDEO_SIZE)
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
    def get_target_info(frame):
        # 保存された設定があれば読み込む
        config_path = "hsv_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            lower_red1, upper_red1 = np.array(cfg['lower1']), np.array(cfg['upper1'])
            lower_red2, upper_red2 = np.array(cfg['lower2']), np.array(cfg['upper2'])
        else:
            # デフォルト値
            lower_red1, upper_red1 = np.array([0, 60, 50]), np.array([35, 255, 255])
            lower_red2, upper_red2 = np.array([160, 60, 50]), np.array([180, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 60, 50]), np.array([35, 255, 255])
        lower_red2, upper_red2 = np.array([160, 60, 50]), np.array([180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2), cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1: return None
        max_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        if stats[max_index, cv2.CC_STAT_AREA] < 500: return None
        h, w = frame.shape[:2]
        s = stats[max_index]
        if (s[0] <= 5) or (s[1] <= 5) or ((s[0]+s[2]) >= (w-5)) or ((s[1]+s[3]) >= (h-5)): return None
        return {'mx': int(centroids[max_index][0]), 'my': int(centroids[max_index][1]), 'area': s[4], 'stat': s}

    @staticmethod
    def dynamic_crop(frame, target):
        if not USE_CROP: return frame
        h, w = frame.shape[:2]
        size = min(int(max(target['stat'][2], target['stat'][3]) * 1.5), w, h)
        x1, y1 = max(0, target['mx'] - size // 2), max(0, target['my'] - size // 2)
        x2, y2 = min(w, x1 + size), min(h, y1 + size)
        if x2 == w: x1 = max(0, w - size)
        if y2 == h: y1 = max(0, h - size)
        return frame[y1:y2, x1:x2]

# ==========================================================
# YOLO検出クラス
# ==========================================================
class YoloDetector:
    def __init__(self, model_path=MODEL_PATH):
        print(f"YOLOモデル {model_path} をロード中...")
        self.model = YOLO(model_path)
        dummy_img = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
        self.model.predict(dummy_img, verbose=False)
        self.logger = OutputLogger()
        self.current_cherry_id = 1
        self.current_detections = []
        self.empty_frames_count = 0
        self.MAX_EMPTY_FRAMES = 8
        self.frame_buffer = {'cam_top': None, 'cam_under': None, 'cam_inside': None, 'cam_outside': None}
        self.best_frames_per_id = {}

    def _resolve_best_result(self, detections):
        """
        全カメラの履歴から最終判定を決定する
        ルール：未熟果(unripe)以外が1つでも検出されていれば、それを優先する
        """
        if not detections: return None

        # カテゴリ分け
        healthy_list = [d for d in detections if d.label_name == "healthy"]
        # unripe以外の不良品 (twin, mold, stemcrack, birddamage)
        other_damaged_list = []
        unripe_list = []

        for d in detections:
            if d.label_name == "healthy" or d.label_name == "None":
                continue
            if d.label_name == "twin":
                if d.confidence >= TWIN_STRICT_THRESHOLD:
                    other_damaged_list.append(d)
            elif d.label_name == "unripe":
                if d.confidence >= UNRIPE_STRICT_THRESHOLD:
                    unripe_list.append(d)
            else:
                # mold, stemcrack, birddamage 等
                other_damaged_list.append(d)

        # 優先順位1: 未熟果以外の有効な判定（健康 または 不良品）がある場合
        if healthy_list or other_damaged_list:
            best_healthy = max(healthy_list, key=lambda x: x.confidence) if healthy_list else None
            best_other_damaged = max(other_damaged_list, key=lambda x: x.confidence) if other_damaged_list else None

            if best_healthy and best_other_damaged:
                # 健康判定が非常に高く(0.9以上)、かつ不良判定がそれほど高くない場合は健康を優先
                if best_healthy.confidence >= 0.9 and best_other_damaged.confidence < 0.9:
                    return best_healthy
                else:
                    return best_other_damaged
            
            if best_other_damaged: return best_other_damaged
            if best_healthy: return best_healthy
        
        # 優先順位2: 他のクラスが1つもなく、未熟果のみが検出されている場合
        if unripe_list:
            return max(unripe_list, key=lambda x: x.confidence)

        # 全てが閾値未満などで有効な判定がない場合のフォールバック（最高信頼度を返す）
        valid_ones = [d for d in detections if d.label_name != "None"]
        if valid_ones:
            return max(valid_ones, key=lambda x: x.confidence)
        return max(detections, key=lambda x: x.confidence)

    def evaluate_frame(self, frame, cam_name, obj_id=None):
        target = ImageProcessor.get_target_info(frame)
        found = target is not None
        if found: self.empty_frames_count = 0
        else: self.empty_frames_count = min(self.empty_frames_count + 1, 1000)
        
        finalized_result = None
        if self.empty_frames_count == self.MAX_EMPTY_FRAMES:
            if len(self.current_detections) > 0:
                best_overall = self._resolve_best_result(self.current_detections)
                if best_overall:
                    self.logger.write_csv(best_overall)
                    finalized_result = best_overall
                    for cam, data in self.best_frames_per_id.items():
                        self.logger.write_image(cam, data['frame'], best_overall.id)
                
                self.current_cherry_id += 1
                self.current_detections = []
                self.best_frames_per_id = {}
        
        actual_obj_id = self.current_cherry_id
        if not found:
            output_frame = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            self._buffer_frame(cam_name, output_frame)
            return output_frame, YoloResult(actual_obj_id, "None", 0.0, cam_name), finalized_result
        
        input_img = ImageProcessor.dynamic_crop(frame, target) if abs(target['mx'] - frame.shape[1] // 2) < CENTER_THRESHOLD_X else frame
        input_img_resized = cv2.resize(input_img, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
        results = self.model.track(input_img_resized, persist=True, verbose=False, conf=CONF_THRESHOLD, tracker="bytetrack.yaml")
        
        annotated_frame = input_img_resized.copy()
        colors = {"birddamage": (255, 255, 0), "healthy": (255, 255, 255), "mold": (238, 130, 238), 
                  "stemcrack": (255, 0, 0), "twin": (0, 0, 255), "unripe": (0, 255, 255)}
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                label = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors.get(label, (0, 255, 0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                label_text = f"{label} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                back_y1 = max(0, y1 - text_h - 10)
                cv2.rectangle(annotated_frame, (x1, back_y1), (x1 + text_w, y1), color, -1)
                cv2.putText(annotated_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        self._buffer_frame(cam_name, annotated_frame)
        best_result = self._parse_results(results, actual_obj_id, cam_name)
        
        if best_result.label_name != "None":
            self.current_detections.append(best_result)
            if cam_name not in self.best_frames_per_id or best_result.confidence > self.best_frames_per_id[cam_name]['conf']:
                self.best_frames_per_id[cam_name] = {
                    'frame': annotated_frame.copy(),
                    'conf': best_result.confidence
                }

        return annotated_frame, best_result, finalized_result

    def _parse_results(self, results, obj_id, cam_name):
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            return YoloResult(obj_id, self.model.names[int(box.cls)], float(box.conf), cam_name)
        return YoloResult(obj_id, "None", 0.0, cam_name)

    def _buffer_frame(self, cam_name, frame):
        self.frame_buffer[cam_name] = frame
        if all(f is not None for f in self.frame_buffer.values()):
            self.logger.write_video(self._create_tile_frame())
            for key in self.frame_buffer.keys(): self.frame_buffer[key] = None

    def _create_tile_frame(self):
        h_left = np.vstack((self.frame_buffer['cam_inside'], self.frame_buffer['cam_under']))
        h_right = np.vstack((self.frame_buffer['cam_outside'], self.frame_buffer['cam_top']))
        return cv2.resize(np.hstack((h_left, h_right)), TILE_VIDEO_SIZE)

    def close(self):
        if len(self.current_detections) > 0:
            best_overall = self._resolve_best_result(self.current_detections)
            if best_overall:
                self.logger.write_csv(best_overall)
                for cam, data in self.best_frames_per_id.items():
                    self.logger.write_image(cam, data['frame'], best_overall.id)
        self.logger.close()