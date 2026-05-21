import cv2
import numpy as np
import datetime
import os
import csv
from ultralytics import YOLO

# ==========================================================
# 定数定義
# ==========================================================
# 動作設定
USE_CROP = False                # ダイナミッククロップを使用するか
CENTER_THRESHOLD_X = 100        # クロップを発動する中心からの許容ピクセル幅

# YOLO設定
MODEL_PATH = "Trained_Models/best2.pt"
YOLO_IMG_SIZE = 640             # 推論およびアノテーション画像のサイズ
CONF_THRESHOLD = 0.7           # 推論の信頼度閾値

# 保存設定
SAVE_DIR_VIDEO = "evaluated_videos"                 # タイル動画の保存先
SAVE_DIR_CSV = "evaluated_csv"                      # CSVの保存先
SAVE_DIR_IMG = "evaluated_images"                   # 画像の保存先親フォルダ
FPS = 20.0                                          # 保存動画のFPS
TILE_VIDEO_SIZE = (YOLO_IMG_SIZE, YOLO_IMG_SIZE)    # (横, 縦)

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

        # カメラごとの画像保存フォルダ作成
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
        if not self.video_writer.isOpened():
            print(f"!!動画ファイルのオープンに失敗しました: {self.video_path}")

    def _init_csv(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "LabelName", "Confidence"])

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
            filename = f"id{obj_id:04d}_{cam_name}_{timestamp}.jpg"
            filepath = os.path.join(self.img_dirs[cam_name], filename)
            cv2.imwrite(filepath, frame)

    def close(self):
        if self.video_writer:
            self.video_writer.release()
            print(f"保存完了: {self.video_path}")
            print(f"保存完了: {self.csv_path}")

# ==========================================================
# 画像処理ユーティリティクラス
# ==========================================================
class ImageProcessor:
    @staticmethod
    def get_target_info(frame):
        """サクランボのマスク抽出と中心・面積取得"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 60, 50])
        upper_red1 = np.array([35, 255, 255])
        lower_red2 = np.array([160, 60, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        if num_labels <= 1:
            return None
        
        max_area = 0
        max_index = -1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_index = i
        
        if max_index == -1 or max_area < 500:
            return None

        height, width = frame.shape[:2]
        stat = stats[max_index]
        x, y, w, h, area = stat
        
        margin = 5
        
        if (x <= margin) or (y <= margin) or ((x + w) >= (width - margin)) or ((y + h) >= (height - margin)):
            return None
        
        mx, my = centroids[max_index]
        return {
            'mx': int(mx),
            'my': int(my),
            'area': max_area,
            'stat': stats[max_index]
        }

    @staticmethod
    def dynamic_crop(frame, target):
        """サクランボを中心にアスペクト比1:1でクロップ"""
        if not USE_CROP:
            return frame
            
        img_h, img_w = frame.shape[:2]
        stat = target['stat']
        
        obj_w, obj_h = stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT]
        
        crop_size = int(max(obj_w, obj_h) * 1.5)
        crop_size = min(crop_size, img_w, img_h)
        
        x1 = max(0, target['mx'] - crop_size // 2)
        y1 = max(0, target['my'] - crop_size // 2)
        x2 = min(img_w, x1 + crop_size)
        y2 = min(img_h, y1 + crop_size)
        
        if x2 == img_w: x1 = max(0, img_w - crop_size)
        if y2 == img_h: y1 = max(0, img_h - crop_size)
        
        return frame[y1:y2, x1:x2]

# ==========================================================
# YOLO検出クラス
# ==========================================================
class YoloDetector:
    def __init__(self, model_path=MODEL_PATH):
        print(f"YOLOモデル {model_path} をロード中...")
        self.model = YOLO(model_path)
        # =======================================================
        # 追加: 初回推論ラグ（ウォームアップ）の解消処理
        # =======================================================
        print("YOLOのウォームアップ（事前推論）を実行しています...")
        # 推論サイズと同じ真っ黒なダミー画像を作成
        dummy_img = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
        # 強制的に1回推論させて、メモリ確保などの初期化を完了させる
        self.model.predict(dummy_img, verbose=False)
        print("ウォームアップ完了。")
        # =======================================================
        self.logger = OutputLogger()
        
        self.current_cherry_id = 1
        self.current_detections = []    
        self.empty_frames_count = 0
        self.MAX_EMPTY_FRAMES = 8       

        # 各カメラの「推論済みフラグ」を管理
        self.has_inferred = {
            'cam_top': False,
            'cam_under': False,
            'cam_inside': False,
            'cam_outside': False
        }

        self.frame_buffer = {
            'cam_top': None,
            'cam_under': None,
            'cam_inside': None,
            'cam_outside': None
        }

    def evaluate_frame(self, frame, cam_name, obj_id=None):
        """画像処理、推論、保存のメインフロー"""
        target = ImageProcessor.get_target_info(frame)
        found = target is not None
        
        if found:
            self.empty_frames_count = 0
        else:
            self.empty_frames_count = min(self.empty_frames_count + 1, 1000)
        
        finalized_result = None

        if self.empty_frames_count == self.MAX_EMPTY_FRAMES:
            if len(self.current_detections) > 0:
                best_overall = max(self.current_detections, key=lambda x: x.confidence)
                self.logger.write_csv(best_overall)
                finalized_result = best_overall
                
            self.current_cherry_id += 1
            self.current_detections = []
            
            # 対象が切り替わったら推論済みフラグを全てリセット
            for key in self.has_inferred.keys():
                self.has_inferred[key] = False
        
        actual_obj_id = self.current_cherry_id
        
        if not found:
            output_frame = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            self._buffer_frame(cam_name, output_frame)
            return output_frame, YoloResult(actual_obj_id, "None", 0.0), finalized_result
        
        img_w = frame.shape[1]
        is_centered = abs(target['mx'] - img_w // 2) < CENTER_THRESHOLD_X

        # 中心付近に到達 ＆ このサクランボに対して対象カメラでまだ推論していない場合のみ実行
        if is_centered and not self.has_inferred[cam_name]:
            input_img = ImageProcessor.dynamic_crop(frame, target) if USE_CROP else frame
            input_img_resized = cv2.resize(input_img, (YOLO_IMG_SIZE, YOLO_IMG_SIZE), interpolation=cv2.INTER_AREA)

            # 1回だけの推論なので track ではなく predict を使用
            results = self.model.predict(input_img_resized, verbose=False, conf=CONF_THRESHOLD)
            # --- ここから変更 ---
            # annotated_frame = results[0].plot()  <-- これを使わず、元の画像をコピーして直接描画する
            annotated_frame = input_img_resized.copy()

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # 1. クラス名と座標を取得
                    class_id = int(box.cls[0])
                    label_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 2. クラス名に応じて色を設定 (BGR形式: (青, 緑, 赤))
                    if label_name == "healthy":
                        box_color = (0, 255, 0)      # 緑色
                    elif label_name == "birdcrack":
                        box_color = (255, 0, 0)      # 青色 (※お好きな色に変更してください)
                    elif label_name == "bruise":
                        box_color = (0, 0, 255)      # 赤色
                    else:
                        box_color = (0, 255, 255)    # 黄色 (その他の欠陥)

                    # 3. バウンディングボックスを描画
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                    # 4. ラベルを描画
                    text = f"{label_name} {confidence:.2f}"
                    cv2.putText(annotated_frame, text, (x1, max(y1 - 10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        # --- ここまで変更 ---
            
            best_result = self._parse_results(results, cam_name, actual_obj_id, found)
            
            if best_result.label_name != "None":
                self.current_detections.append(best_result)

            self.logger.write_image(cam_name, annotated_frame, actual_obj_id)
            
            # このカメラでの推論を完了とする
            self.has_inferred[cam_name] = True

            self._buffer_frame(cam_name, annotated_frame)
            return annotated_frame, best_result, finalized_result
            
        else:
            # 中心にいない、または既に推論済みの場合はスルー（推論しない）
            output_frame = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            self._buffer_frame(cam_name, output_frame)
            return output_frame, YoloResult(actual_obj_id, "None", 0.0), finalized_result

    def _parse_results(self, results, cam_name, obj_id, found):
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            return YoloResult(obj_id, self.model.names[int(box.cls)], float(box.conf))
        return YoloResult(obj_id, "None", 0.0)

    # ==========================================================
    # フレーム同期とタイル合成メソッド
    # ==========================================================
    def _buffer_frame(self, cam_name, frame):
        if cam_name in self.frame_buffer:
            if frame.shape[:2] != (YOLO_IMG_SIZE, YOLO_IMG_SIZE):
                frame = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            self.frame_buffer[cam_name] = frame
        
        if all(f is not None for f in self.frame_buffer.values()):
            tile_frame = self._create_tile_frame()
            self.logger.write_video(tile_frame)
            self._clear_buffer()

    def _create_tile_frame(self):
        v_tl = self.frame_buffer['cam_inside']
        v_bl = self.frame_buffer['cam_under']
        v_tr = self.frame_buffer['cam_outside']
        v_br = self.frame_buffer['cam_top']
        
        h_left = np.vstack((v_tl, v_bl))
        h_right = np.vstack((v_tr, v_br))
        tile_image = np.hstack((h_left, h_right))
        
        return cv2.resize(tile_image, TILE_VIDEO_SIZE)

    def _clear_buffer(self):
        for key in self.frame_buffer.keys():
            self.frame_buffer[key] = None

    def close(self):
        if len(self.current_detections) > 0:
            best_overall = max(self.current_detections, key=lambda x: x.confidence)
            self.logger.write_csv(best_overall)
        
        if any(f is not None for f in self.frame_buffer.values()):
            black_img = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
            for key, frame in self.frame_buffer.items():
                if frame is None: self.frame_buffer[key] = black_img
            tile_frame = self._create_tile_frame()
            self.logger.write_video(tile_frame)

        self.logger.close()