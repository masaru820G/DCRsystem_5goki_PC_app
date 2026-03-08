# -------------------------------------------------
# main.py
# -------------------------------------------------
import sys
import requests
import random
import cv2

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Slot, Qt, QRunnable, QThreadPool, QTimer
from PySide6.QtGui import QKeyEvent, QImage, QPixmap

# デザインファイルを読み込む
import module_gui

# 制御モジュール
import module_patlite as p_ctr
import module_relay as r_ctr
import module_cameras_5goki as cam_ctr
#import module_yolo_csv as yolo_ctr
import module_yolo_csv as yolo_ctr

RPI_IP_ADDRESS = "192.168.2.1"
RPI_PORT = 5000

# ==========================================================
# 汎用バックグラウンドタスク用クラス
# ==========================================================
class TaskWorker(QRunnable):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.func(*self.args, **self.kwargs)            # 渡された関数を実行 (引数付き)
        except Exception as e:
            print(f"!![Background Task Error]: {e}")

# ==========================================================
# スタートアップウィンドウ
# ==========================================================
class StartupWindow(module_gui.StartupWindowUI):
    def __init__(self):
        super().__init__()
        self.button_start.clicked.connect(self.launch_main)
    def launch_main(self):
        self.main_window = MainWindow()
        self.main_window.showFullScreen()
        self.close()

# ==========================================================
# サブウィンドウ
# ==========================================================
class SubWindow(module_gui.SubWindowUI):
    # --- 戻るボタン押下イベント -------------------
    def __init__(self, parent_window, initial_speed):
        super().__init__()
        self.button_up_speed.clicked.connect(self.on_up_speed)
        self.button_down_speed.clicked.connect(self.on_down_speed)
        self.button_back.clicked.connect(self.go_back)

        self.parent_window = parent_window
        # スピード値の管理と初期化
        self.current_speed = initial_speed  # 親ウィンドウから現在の速度を受け取る変数
        self.update_speed_ui()  # 画面更新

    # --- 画面の表示とボタンの状態を更新する関数 ---
    def update_speed_ui(self):
        self.label_current_speed.setText(str(self.current_speed))   # ラベルの数字を更新

        # 上限チェック (10になったらUpボタンをロック)
        if self.current_speed >= 10:
            self.button_up_speed.set_locked(True)   # ロック＆半透明
        else:
            self.button_up_speed.set_locked(False)  # 解除

        # 下限チェック (1になったらDownボタンをロック)
        if self.current_speed <= 1:
            self.button_down_speed.set_locked(True) # ロック＆半透明
        else:
            self.button_down_speed.set_locked(False)# 解除

    # --- speed upボタン押下イベント -------------------
    @Slot()
    def on_up_speed(self):
        if self.current_speed < 10:
            self.current_speed += 1
            self.update_speed_ui()

    # --- speed downボタン押下イベント -------------------
    @Slot()
    def on_down_speed(self):
        if self.current_speed > 1:
            self.current_speed -= 1
            self.update_speed_ui()

    # --- 戻るボタン押下イベント -------------------
    @Slot()
    def go_back(self):
        self.parent_window.saved_speed = self.current_speed        # メインウインドウにスピード設定値を渡す
        self.close()

# ==========================================================
# メインウィンドウ
# ==========================================================
class MainWindow(module_gui.MainWindowUI):
    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()    # スレッド管理プールの作成

        # 各デバイス接続処理
        self.patlite = p_ctr.PatliteController()
        if not self.patlite.init():
            print("!!パトライトの接続に失敗しました")
            self.close()
        self.relay = r_ctr.RelayController()
        if not self.relay.init():
            print("!!リレーボードの接続に失敗しました")
            self.close()
        self.cameras = cam_ctr.CameraManager()
        if not self.cameras.init_cameras():
            print("!!カメラの接続に失敗しました")
            self.close()

        # YOLO初期化・一度だけロード
        self.detector = yolo_ctr.YoloDetector("C:/gamo/yolo_v1/runs/detect/train7/weights/best.pt")

        # --- 表示同期の設定 ---
        DELAY_TIME_SEC = 2.5

        for controller in self.cameras.controllers:
            # 先に映る下流側のカメラ（under, inside）に遅延を設定
            if controller.name in ["cam_under", "cam_inside"]:
                controller.delay_seconds = DELAY_TIME_SEC
                print(f"[Success] {controller.name} に {DELAY_TIME_SEC} 秒の表示遅延を設定しました")
        self.cameras.start_all_get_frame() # 起動と同時にキャプチャ開始

        # イベント接続
        self.toggle_switch.toggled.connect(self.on_main_toggled)
        self.button_setting.clicked.connect(self.on_setting_button)
        self.button_power.clicked.connect(self.on_power_bottom)

        self.saved_speed = 5        # 速度設定値を記憶しておく変数 初期値5

        # 履歴管理用の変数
        self.history_data = []  # 履歴データリスト [(id, result, conf), ...]
        self.current_id = 1     # IDカウンタ

        # 映像更新用タイマー設定
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feeds)
        self.timer.start(50)  # 50msごとに更新 (約20fps)

    # --- カメラ映像をGUIに反映する関数 ---
    def update_video_feeds(self):
        # トグルスイッチがOFFの場合は推論や更新を行わないならここで return してもOKです
        for controller in self.cameras.controllers:
            # タイマー更新時
            frame = controller.get_current_frame()  # 最新フレームを取得 (BGR形式)
            if frame is not None:
                # ★戻り値を3つ(annotated_frame, result, finalized_result)で受け取るように修正
                annotated_frame, result, finalized_result = self.detector.evaluate_frame(frame, controller.name, self.current_id)

                # ★もしサクランボが画面から出て最終結果が確定していたら、GUIとパトライトを更新する
                if finalized_result is not None:
                    self.process_final_result(finalized_result)

                # 描画用には annotated_frame を使用
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # ラベルのサイズに合わせてリサイズ (アスペクト比保持)
                # controller.name に応じて貼り付けるラベルを決める
                target_label = None
                if controller.name == "cam_inside":
                    target_label = self.cam_in
                elif controller.name == "cam_outside":
                    target_label = self.cam_out
                elif controller.name == "cam_under":
                    target_label = self.cam_under
                elif controller.name == "cam_top":
                    target_label = self.cam_top

                if target_label:
                    pixmap = QPixmap.fromImage(qt_image)
                    scaled_pixmap = pixmap.scaled(
                        target_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    target_label.setPixmap(scaled_pixmap)

    # --- バックグラウンドで渡された関数を実行するヘルパー関数 ---
    def run_in_background(self, func, *args, **kwargs):
        worker = TaskWorker(func, *args, **kwargs)
        self.thread_pool.start(worker)

    # --- ラズパイと通信する関数 -------------------
    def __async_raspi_request(self, command):
        url = f"http://{RPI_IP_ADDRESS}:{RPI_PORT}{command}"
        try:
            print(f">>>[Sending]: {url}")
            requests.get(url, timeout=2)
        except Exception as e:
            print(f"!![Net Error]: {e}")

    # --- 設定ボタン押下イベント -------------------
    @Slot()
    def on_setting_button(self):
        self.settings_window = SubWindow(parent_window=self, initial_speed=self.saved_speed)
        self.settings_window.show()

    # --- 電源ボタン押下イベント -------------------
    @Slot()
    def on_power_bottom(self):
        print("\n電源ボタンが押されました。終了します。\n")
        self.timer.stop()

        # デバイス停止処理
        self.patlite.close()
        self.relay.close()
        self.cameras.stop_all_get_frame() # カメラ停止
        self.run_in_background(self.__async_raspi_request, "/cleanup_system")
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close() # これを必ず呼ぶ

        self.close()                    # アプリケーションを閉じる

    # --- 確定した推論結果をGUIとパトライトに反映する関数 ------------
    def process_final_result(self, result_obj):
        # トグルスイッチがOFFなら、動作させない
        if not self.toggle_switch.isChecked():
            return

        disease_name = result_obj.label_name
        confidence_percent = int(result_obj.confidence * 100)
        obj_id = result_obj.id

        pattern = None
        channel = None
        display_name = ""

        if disease_name == "mold":
            pattern = p_ctr.LedPattern.VIOLET
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "カビ"
            self.label_dam.setText("カビ")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #800080;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "unripe":
            pattern = p_ctr.LedPattern.YELLOW
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "未熟果"
            self.label_dam.setText("未熟果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #000000; background-color: #FFFF00;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "bowlcrack":
            pattern = p_ctr.LedPattern.BLUE
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "果梗裂果"
            self.label_dam.setText("果梗裂果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #000000; background-color: #0040FF;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)

        # 判定処理が行われた場合のみ履歴更新
        if pattern is not None:
            # デバイス制御 (非同期)
            self.run_in_background(self.patlite.set_color, pattern)
            self.run_in_background(self.relay.move, channel, self.saved_speed)

            # 履歴データの追加処理 (キー入力時のランダム値ではなく、実際のYOLO結果を使用)
            record = {
                "id": obj_id,
                "result": display_name,
                "conf": confidence_percent
            }
            self.history_data.append(record)

            # 古いものを削除
            if len(self.history_data) > 10:
                self.history_data.pop(0)

            # 確認用ログ
            #_, color_name = pattern
            #print(f"Latest History: | ID: {record['id']:03} | 判定結果: {record['result']}({color_name}) | 信頼度: {record['conf']} % |")

            # 画面更新
            self.update_history_display()

    # --- 履歴表示を更新する関数 (HTMLテーブル版) -------------------
    def update_history_display(self):
        # 履歴データをHTMLのテーブル行(tr)に変換する
        rows_html = ""
        for item in self.history_data:
            # IDの作成 (半角->全角変換)
            id_txt = f"{item['id']:03}".translate(str.maketrans("0123456789", "０１２３４５６７８９"))

            # 結果の作成 (空白除去 & 色判定)
            raw_text = item['result']
            if "カビ" in raw_text:
                color_code = "#EE82EE"
            elif "未熟果" in raw_text:
                color_code = "#FFFF00"
            elif "果梗裂果" in raw_text:
                color_code = "#0040FF"

            # 信頼度の作成
            conf_txt = f"{item['conf']} ％".translate(str.maketrans("0123456789", "０１２３４５６７８９"))

            # 行の組み立て (HTMLテーブルのタグを使用)
            rows_html += f"""
            <tr>
                <td align="center" style="border-right: 1px solid #00FF00;">{id_txt}</td>
                <td align="center" style="border-right: 1px solid #00FF00; color:{color_code};">{raw_text}</td>
                <td align="center" style="border-right: 1px solid #00FF00;">{conf_txt}</td>
            </tr>
            """

        # 全体のHTMLを組み立てる
        full_html = f"""
        <html>
        <head>
        <style>
            table {{
                border-collapse: collapse; /* 線の隙間をなくす */
                width: 100%;
                border: 1px solid #00FF00; /* これで消えていた「IDの左」と「信頼度の右」の線が復活します */
            }}
            /* ヘッダーセルの設定 */
            th {{
                font-family: "MS Gothic"; font-size: 20px; font-weight: bold; color: #00FF00;
                border-right: 1px solid #00FF00;   /* 縦の区切り線 */
                padding: 4px;
            }}
            /* データセルの設定 */
            td {{
                font-family: "MS Gothic"; font-size: 16px; font-weight: bold; color: #00FF00;
                padding: 3px;
            }}
        </style>
        </head>
        <body style="background-color:#000000;">
            <table cellspacing="0">
                <tr>
                    <th width="20%">ＩＤ</th>
                    <th width="40%">結果</th>
                    <th width="40%">信頼度</th>
                </tr>
                {rows_html}
            </table>
        </body>
        </html>
        """

        self.label_history.setText(full_html)

    # --- トグルスイッチ状態変更イベント --------------------------
    @Slot(bool)
    def on_main_toggled(self, checked):
        self.button_setting.set_locked(checked)
        if checked:
            self.label_toggle_status.setText("動作中")
            self.label_toggle_status.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #32CD32; qproperty-alignment: 'AlignCenter';
            """)

            self.run_in_background(self.__async_raspi_request, f"/set_speed/{self.saved_speed}")
            print(f"\nSpeed settings saved to Main: {self.saved_speed}")
            self.run_in_background(self.__async_raspi_request, "/rotate")
        else:
            self.label_toggle_status.setText("停止中")
            self.label_toggle_status.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #888888; qproperty-alignment: 'AlignCenter';
            """)
            # 2. 裏でコマンド送信 (非同期)
            self.run_in_background(self.__async_raspi_request, "/stop")
            self.run_in_background(self.relay.stop)  # リレーボード停止
            self.run_in_background(self.patlite.set_color, p_ctr.LedPattern.OFF)
# ==========================================================
# 実行ブロック
# ==========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartupWindow()
    window.show()
    sys.exit(app.exec())