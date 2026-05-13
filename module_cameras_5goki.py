import os
import time
import sys
import cv2
import threading
import gc
from pypylon import pylon
from collections import deque
from PySide6.QtCore import QObject, Signal # 追加

# ==========================================================
# 定数定義
# ==========================================================
TARGET_SERIALS = [
    ("25308967", "cam_top"),
    ("21905526", "cam_under"),
    ("25308969", "cam_inside"),
    ("25308968", "cam_outside")
]

FPS = 20.0

# ==========================================================
# PFSファイルを正確に読み込むための補助関数
# ==========================================================
def load_pfs_custom(camera, pfs_path):
    if not os.path.exists(pfs_path):
        return False
        
    try:
        pylon.FeaturePersistence.Load(pfs_path, camera.GetNodeMap(), True)
        return True
    except Exception as e:
        print(f"!! PFSロードエラー ({pfs_path}): {e}")
        return False

# ==========================================================
# シグナル送信用のヘルパークラス
# ==========================================================
class CameraSignals(QObject):
    connection_lost = Signal(str)

# ==========================================================
# カメラ制御クラス（映像取得・表示専用版）
# ==========================================================
class CameraController(QObject):
    def __init__(self, device_info, cam_name = "unknown"):
        super().__init__()
        self.signals = CameraSignals() # シグナルオブジェクトの初期化
        self.device_info = device_info
        self.name = cam_name
        self.settings_file = f"cam_pfs/{self.name}.pfs"

        self.camera = None
        self.is_capturing = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # 表示同期用の設定
        self.delay_seconds = 0.0
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
            
            self.camera.MaxNumBuffer = 20

            # 各カメラが最大 100MB/s 以上出さないように制限（4台で400MB/s = USB3.0の実効限界付近）
            if hasattr(self.camera, 'DeviceLinkThroughputLimit'):
                self.camera.DeviceLinkThroughputLimitMode.Value = "On"
                self.camera.DeviceLinkThroughputLimit.Value = 100000000 # 100MB/s
                
            if os.path.exists(self.settings_file):
                success = load_pfs_custom(self.camera, self.settings_file)
                if success:
                    print(f"[Success] {self.name}: 設定を適用しました")
                else:
                    print(f"!!エラー!! {self.name}: 設定解析失敗")
            
            self.width = self.camera.Width.Value
            self.height = self.camera.Height.Value
            return True
        except Exception as e:
            print(f"!!カメラ初期化エラー ({self.name}): {e}")
            return False

    def start_capture(self):
        """キャプチャループを開始（保存はしない）"""
        if not self.camera or not self.camera.IsOpen():
            return
        
        self.is_capturing = True
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"キャプチャ開始: {self.name}")

    def _capture_loop(self):
        while self.is_capturing:
            if not self.camera or not self.camera.IsGrabbing():
                # グラビングが停止した＝接続が切れたと判断
                print(f"!![Lost] {self.name}: Camera stopped grabbing.")
                self.signals.connection_lost.emit(self.name)
                break
            try:
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    converted = self.converter.Convert(grab_result)
                    frame_bgr = converted.GetArray()

                    # フレームの更新処理
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
                            self.frame_queue.clear()
                else:
                    print(f"!![Lost] {self.name}: Grab failed.")
                    self.signals.connection_lost.emit(self.name)
                    break
                grab_result.Release()
            except Exception as e:
                print(f"!!Loop Error ({self.name}): {e}")
                self.signals.connection_lost.emit(self.name) # エラー時に通知
                break

    def stop_capture(self):
        self.is_capturing = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.camera and self.camera.IsGrabbing():
            self.camera.StopGrabbing()

    def get_current_frame(self):
        with self.lock:
            return self.latest_frame

    def close(self):
        """カメラリソースを完全に解放する"""
        self.stop_capture()
        if self.camera is not None:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
                # 専有権を解放するために重要
                self.camera.DetachDevice()
                self.camera.Destroy()
            except Exception as e:
                print(f"!!カメラ解放エラー ({self.name}): {e}")
            finally:
                self.camera = None
        print(f">>> {self.name}: リソース解放完了")

class CameraManager:
    def __init__(self):
        self.controllers = []

    def init_cameras(self):
        # 1. 既存のコントローラーを完全に破棄
        for controller in self.controllers:
            controller.close()
        self.controllers.clear()
        
        # 2. Pythonのガベージコレクションを強制実行してOSにリソースを戻す
        gc.collect()
        time.sleep(0.5) # ドライバが専有権を戻すための物理的な猶予

        try:
            tl_factory = pylon.TlFactory.GetInstance()
            # キャッシュをクリア（これが重要！）
            devices = tl_factory.EnumerateDevices()
        except Exception as e:
            print(f"!!Pylon初期化エラー: {e}")
            return False

        if not devices:
            print("!!カメラが見つかりません")
            return False

        # 3. 指定したシリアルに一致するカメラのみを開く
        for target_serial, cam_name in TARGET_SERIALS:
            found_device_info = next((d for d in devices if d.GetSerialNumber() == target_serial), None)
            if found_device_info:
                controller = CameraController(found_device_info, cam_name)
                if controller.init_camera():
                    self.controllers.append(controller)
                else:
                    # 初期化に失敗した場合はリソースを即座に捨てる
                    controller.close()
                    del controller

        return len(self.controllers) > 0

    def start_all_get_frame(self):
        for controller in self.controllers:
            controller.start_capture()

    def stop_all_get_frame(self):
        for controller in self.controllers:
            controller.close()