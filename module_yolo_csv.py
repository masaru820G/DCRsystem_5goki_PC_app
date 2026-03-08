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
USE_CROP = True                 # ダイナミッククロップを使用するか
CENTER_THRESHOLD_X = 50          # クロップを発動する中心からの許容ピクセル幅

# YOLO設定
MODEL_PATH = "C:/gamo/yolo_v1/runs/detect/train7/weights/best.pt"
YOLO_IMG_SIZE = 320             # 推論およびアノテーション画像のサイズ
CONF_THRESHOLD = 0.5            # 推論の信頼度閾値

# 保存設定
SAVE_DIR_VIDEO = "evaluated_videos" # タイル動画の保存先
SAVE_DIR_CSV = "evaluated_csv"     # CSVの保存先
FPS = 10.0                      # 保存動画のFPS（実測に合わせて調整してください）

# ★追加：4分割画面のサイズ
TILE_VIDEO_SIZE = (YOLO_IMG_SIZE * 2, YOLO_IMG_SIZE * 2) # (横, 縦)

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
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(SAVE_DIR_VIDEO, f"eval_{timestamp}.mp4")
        self.csv_path = os.path.join(SAVE_DIR_CSV, f"eval_{timestamp}.csv")
        
        self.video_writer = None
        self._init_csv()

    def _init_video(self):
        """動画ライターを★TILE_VIDEO_SIZEで初期化"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # TILE_VIDEO_SIZE を使用
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, FPS, TILE_VIDEO_SIZE)
        if not self.video_writer.isOpened():
            print(f"[Error] 動画ファイルのオープンに失敗しました: {self.video_path}")

    def _init_csv(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "LabelName", "Confidence"])

    def write_video(self, tile_frame):
        """★4分割合成されたタイル画像を動画に書き込む"""
        if self.video_writer is None:
            self._init_video()
        
        # 入力画像のサイズチェック（念のため）
        h, w = tile_frame.shape[:2]
        if (w, h) != TILE_VIDEO_SIZE:
             tile_frame = cv2.resize(tile_frame, TILE_VIDEO_SIZE)
             
        self.video_writer.write(tile_frame)

    def write_csv(self, result_obj):
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(result_obj.to_csv_row())

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
        # HSVに変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 赤色の範囲（2つの範囲を結合）
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        H_LOW = np.array([10, 40, 120])
        H_HIGH = np.array([179, 255, 255])
        
        #mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        #mask = mask1 + mask2
        mask = cv2.inRange(hsv, H_LOW, H_HIGH)
        
        # モルフォロジー処理（ノイズ除去と結合）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 接続成分の解析
        # ★バグ修正: 戻り値を4つに変更
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        if num_labels <= 1: # 背景のみ
            return None
        
        # 最大の面積を持つラベルを探す（背景を除く）
        max_area = 0
        max_index = -1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_index = i
        
        if max_index == -1 or max_area < 500: # 面積が小さすぎる場合は無視
            return None
        
        # 結果を辞書で返す
        mx, my = centroids[max_index]
        return {
            'mx': int(mx),
            'my': int(my),
            'area': max_area,
            'stat': stats[max_index] # [x, y, w, h, area]
        }

    @staticmethod
    def dynamic_crop(frame, target):
        """サクランボを中心にアスペクト比1:1でクロップ"""
        if not USE_CROP:
            return frame
            
        img_h, img_w = frame.shape[:2]
        stat = target['stat']
        
        # サクランボの外接矩形の幅と高さ
        obj_w, obj_h = stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT]
        
        # クロップサイズ（外接矩形の大きい方に余白を追加）
        crop_size = int(max(obj_w, obj_h) * 1.5)
        crop_size = min(crop_size, img_w, img_h) # 画像サイズを超えないように
        
        # クロップ範囲の計算（中心を合わせる）
        x1 = max(0, target['mx'] - crop_size // 2)
        y1 = max(0, target['my'] - crop_size // 2)
        x2 = min(img_w, x1 + crop_size)
        y2 = min(img_h, y1 + crop_size)
        
        # 範囲が画像外に出た場合の調整
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
        self.logger = OutputLogger()
        
        self.current_cherry_id = 1
        self.current_detections = []    # 1つのサクランボに対する複数カメラ/フレームの検出結果を溜める
        self.empty_frames_count = 0
        self.MAX_EMPTY_FRAMES = 8       # 4台のカメラ×2サイクル分連続で未検出なら「完全に画面外」とみなす

        # ★追加：カメラフレーム同期用のバッファ
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
        
        # --- 画面内外の判定とID管理 ---
        if found:
            self.empty_frames_count = 0
        else:
            self.empty_frames_count = min(self.empty_frames_count + 1, 100)
        
        finalized_result = None

        if self.empty_frames_count == self.MAX_EMPTY_FRAMES:
            if len(self.current_detections) > 0:
                best_overall = max(self.current_detections, key=lambda x: x.confidence)
                self.logger.write_csv(best_overall)
                finalized_result = best_overall
                self.current_cherry_id += 1
                self.current_detections = []
        
        actual_obj_id = self.current_cherry_id
        
        # ターゲット未検出時は推論をスキップ
        if not found:
            # GUI用にはリサイズ画像を用意
            output_frame = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            
            # ★追加：処理済みフレームをバッファに保存（保存処理はここで行わない）
            self._buffer_frame(cam_name, output_frame)
            # ★バグ修正：戻り値を3つにする
            return output_frame, YoloResult(actual_obj_id, "None", 0.0), finalized_result
        
        # 中心判定とクロップ
        img_w = frame.shape[1]
        is_centered = abs(target['mx'] - img_w // 2) < CENTER_THRESHOLD_X

        input_img = ImageProcessor.dynamic_crop(frame, target) if is_centered else frame
        input_img_resized = cv2.resize(input_img, (YOLO_IMG_SIZE, YOLO_IMG_SIZE), interpolation=cv2.INTER_AREA)

        # YOLO推論
        results = self.model.track(input_img_resized, persist=True, verbose=False, conf=CONF_THRESHOLD, tracker="bytetrack.yaml")
        annotated_frame = results[0].plot()
        
        # ★追加：アノテーション済みフレームをバッファに保存
        self._buffer_frame(cam_name, annotated_frame)
        
        best_result = self._parse_results(results, cam_name, actual_obj_id, found)
        
        # 推論結果が有効ならバッファに追加（ここではまだCSVに書かない）
        if best_result.label_name != "None":
            self.current_detections.append(best_result)

        return annotated_frame, best_result, finalized_result

    def _parse_results(self, results, cam_name, obj_id, found):
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            return YoloResult(obj_id, self.model.names[int(box.cls)], float(box.conf))
        return YoloResult(obj_id, "None", 0.0)

    # ==========================================================
    # ★追加：フレーム同期とタイル合成メソッド
    # ==========================================================
    def _buffer_frame(self, cam_name, frame):
        """各カメラの処理済みフレームをバッファに溜める"""
        if cam_name in self.frame_buffer:
            # 入力フレームがYOLO_IMG_SIZEと異なる場合はリサイズして保存
            if frame.shape[:2] != (YOLO_IMG_SIZE, YOLO_IMG_SIZE):
                frame = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            self.frame_buffer[cam_name] = frame
        
        # すべてのカメラのフレームが揃ったか確認
        if all(f is not None for f in self.frame_buffer.values()):
            # 揃ったら合成して動画に書き込む
            tile_frame = self._create_tile_frame()
            self.logger.write_video(tile_frame)
            # バッファをクリア（次のフレームの揃いを待つ）
            self._clear_buffer()

    def _create_tile_frame(self):
        """★バッファのフレームを指定配置で合成する"""
        # 配置に従ってフレームを取得
        v_tl = self.frame_buffer['cam_inside']      # 左上
        v_bl = self.frame_buffer['cam_under']    # 左下
        v_tr = self.frame_buffer['cam_outside']   # 右上
        v_br = self.frame_buffer['cam_top']  # 右下
        
        # 合成ロジック (左側縦結合、右側縦結合、最後に横結合)
        h_left = np.vstack((v_tl, v_bl))
        h_right = np.vstack((v_tr, v_br))
        tile_image = np.hstack((h_left, h_right))
        
        # 最終アスペクト比を1:1 (TILE_VIDEO_SIZE) に調整
        return cv2.resize(tile_image, TILE_VIDEO_SIZE)

    def _clear_buffer(self):
        """バッファをクリア（Noneに戻す）"""
        for key in self.frame_buffer.keys():
            self.frame_buffer[key] = None

    def close(self):
        # プログラム終了時にバッファに残っていれば強制的に出力
        if len(self.current_detections) > 0:
            best_overall = max(self.current_detections, key=lambda x: x.confidence)
            self.logger.write_csv(best_overall)
        
        # ★追加：もしバッファにフレームが残っていたら、最後のタイルを作って書き込む（同期は無視）
        if any(f is not None for f in self.frame_buffer.values()):
            # Noneのフレームを真っ黒な画像で埋めて合成
            black_img = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
            for key, frame in self.frame_buffer.items():
                if frame is None: self.frame_buffer[key] = black_img
            tile_frame = self._create_tile_frame()
            self.logger.write_video(tile_frame)

        self.logger.close()