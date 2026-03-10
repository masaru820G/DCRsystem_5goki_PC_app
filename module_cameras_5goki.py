import os
import time
import sys
import cv2
import threading
import re
from pypylon import pylon
from collections import deque  # 追加: フレームバッファ用


# ==========================================================
# 定数定義
# ==========================================================
TARGET_SERIALS = [
    ("25308967", "cam_top"),
    ("21905526", "cam_under"),
    ("25308969", "cam_inside"),
    ("25308968", "cam_outside")
]

FOLDER_PARENT = "cam_video"
FOLDER_CHILD = [
    "cam_video_top",
    "cam_video_under",
    "cam_video_inside",
    "cam_video_outside"
]

VIDEO_CODEC = 'mp4v'
VIDEO_EXIT = '.mp4'
FPS = 20.0

def setup_folders():
    try:
        if not os.path.exists(FOLDER_PARENT):
            os.makedirs(FOLDER_PARENT)
        paths = []
        for folder in FOLDER_CHILD:
            path = os.path.join(FOLDER_PARENT, folder)
            if not os.path.exists(path):
                os.makedirs(path)
            paths.append(path)
        return paths
    except OSError as e:
        print(f"!!フォルダ作成エラー: {e}")
        sys.exit(1)

# ==========================================================
# PFSファイルを正確に読み込むための補助関数
# ==========================================================
def load_pfs_custom(camera, pfs_path):
    if not os.path.exists(pfs_path):
        return False
        
    try:
        pylon.FeaturePersistence.Load(pfs_path, camera.GetNodeMap(), True)        # pylon公式のPFSロード機能
        return True
    except Exception as e:
        print(f"!! PFSロードエラー ({pfs_path}): {e}")
        return False

# ==========================================================
# カメラ制御クラス（色再現改善版）
# ==========================================================
class CameraController:
    def __init__(self, device_info, save_path, cam_name = "unknown"):
        self.device_info = device_info
        self.save_path = save_path
        self.name = cam_name
        self.settings_file = f"cam_pfs/{self.name}.pfs"

        self.camera = None
        self.video_writer = None
        self.is_recording = False
        self.thread = None
        self.video_filename = ""
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # 表示同期用の設定
        self.delay_seconds = 0.0  # 遅延秒数
        self.frame_queue = deque()

        # Pylon Viewerと同じ色再現を行うためのコンバーター
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        self.width = 1280
        self.height = 960

    def init_camera(self):
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.device_info))
            self.camera.Open()
            
            # --- カスタムPFSロード ---
            if os.path.exists(self.settings_file):
                success = load_pfs_custom(self.camera, self.settings_file)
                if success:
                    print(f"[Success] {self.name}: 設定を精密に適用しました")
                else:
                    print(f"!!エラー!! {self.name}: ファイルはありますが、中身の解析に失敗しました ({self.settings_file})")
            else:
                # ファイルが存在しない場合、絶対パスを表示して原因を探る
                abs_path = os.path.abspath(self.settings_file)
                print(f"!!警告!! {self.name}: 設定ファイルが見つかりません。")
                print(f"探している場所: {abs_path}")
            
            # 設定後の解像度を取得
            self.width = self.camera.Width.Value
            self.height = self.camera.Height.Value
            print(f"[Info] {self.name} Resolution: {self.width}x{self.height}")

            return True
        except Exception as e:
            print(f"!!カメラ初期化エラー ({self.name}): {e}")
            return False

    def start_recording(self):
        if not self.camera or not self.camera.IsOpen():
            return
        
        folder_name = os.path.basename(self.save_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self.video_filename = os.path.join(self.save_path, f"{folder_name}_{timestamp}{VIDEO_EXIT}")
        
        self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, FPS, (self.width, self.height))
        
        self.is_recording = True
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"録画開始: {self.name}")

    def _capture_loop(self):
        while self.is_recording and self.camera.IsGrabbing():
            try:
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    converted = self.converter.Convert(grab_result)
                    frame_bgr = converted.GetArray()

                    if self.video_writer:
                        self.video_writer.write(frame_bgr)

                    # --- 秒数からフレーム数を計算して遅延実行 ---
                    with self.lock:
                        delay_frames = int(self.delay_seconds * FPS)
                        
                        if delay_frames > 0:
                            self.frame_queue.append(frame_bgr.copy())
                            if len(self.frame_queue) > delay_frames:
                                self.latest_frame = self.frame_queue.popleft()
                            else:
                                self.latest_frame = None
                        else:
                            self.latest_frame = frame_bgr.copy()
                            self.frame_queue.clear() # 遅延0ならキューを空にする

                    self.video_writer.write(frame_bgr)
                
                grab_result.Release()
            except Exception as e:
                print(f"!!Loop Error ({self.name}): {e}")
                break

    def stop_recording(self):
        self.is_recording = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.camera and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        if self.video_writer:
            self.video_writer.release()

    def get_current_frame(self):
        with self.lock:
            return self.latest_frame

    def close(self):
        self.stop_recording()
        if self.camera and self.camera.IsOpen():
            self.camera.Close()

class CameraManager:
    def __init__(self):
        self.controllers = []
        setup_folders()

    def init_cameras(self):
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
        except Exception as e:
            print(f"!!Pylon初期化エラー: {e}")
            return False

        if not devices:
            return False

        for i, (target_serial, cam_name) in enumerate(TARGET_SERIALS):
            found_device_info = next((d for d in devices if d.GetSerialNumber() == target_serial), None)
            if found_device_info:
                save_path = os.path.join(FOLDER_PARENT, FOLDER_CHILD[i])
                controller = CameraController(found_device_info, save_path, cam_name)
                if controller.init_camera():
                    self.controllers.append(controller)

        return len(self.controllers) > 0

    def start_all_get_frame(self):
        for controller in self.controllers:
            controller.start_recording()

    def stop_all_get_frame(self):
        for controller in self.controllers:
            controller.close()