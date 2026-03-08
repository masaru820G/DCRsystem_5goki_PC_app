# -------------------------------------------------
# DCR_systemのGUIデザイン定義ファイル
# -------------------------------------------------
import sys
import os
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QLabel, QPushButton, QCheckBox, QVBoxLayout, QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, Property, QPropertyAnimation, QEasingCurve, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QBrush, QPixmap

# ==========================================================
# 定数定義
# ==========================================================
MARGIN_X = 10                           # 画面の余白x
MARGIN_Y = 10                           # 画面の余白y

# --- メインウインドウゾーン ----------------------------
#WINDOW_W, WINDOW_H = 1280, 800          # 1920×1200 拡大/縮小 150%
#WINDOW_W, WINDOW_H = 1280, 720          # 1920×1080 拡大/縮小 150%
WINDOW_W, WINDOW_H = 1706, 1066          # 2560×1600 拡大/縮小 150%

VIEW_CAM_SIZE_W = int(WINDOW_W * 0.34) // 10 * 10     # カメラ表示サイズw (10の倍数に丸める)
VIEW_CAM_SIZE_H = (WINDOW_H - MARGIN_Y * 3) / 2       # カメラ表示サイズh

ICON_SETTING_SIZE = int(WINDOW_W * 0.1) // 10 * 10    # 設定アイコンサイズ
ICON_POWER_SIZE = int(WINDOW_W * 0.1) // 10 * 10      # 電源アイコンサイズ

LABEL_HISTORY_SIZE_W = int(WINDOW_W * 0.3) // 10 * 10                  # 履歴ラベルの幅
LABEL_HISTORY_SIZE_H = int(WINDOW_H * 0.4) // 10 * 10     # 履歴ラベルの高さ

LABEL_DAMAGE_SIZE_W = int(WINDOW_W * 0.3) // 10 * 10                      # 病害名表示ラベルサイズw
LABEL_DAMAGE_SIZE_H = int(WINDOW_H * 0.1) // 10 * 10                      # 病害名表示ラベルサイズh

LABEL_MANAGEMENT_SIZE_W = int(WINDOW_W * 0.2) // 10 * 10               # 「システム管理」ラベルサイズw
LABEL_MANAGEMENT_SIZE_H = int(WINDOW_H * 0.1) // 10 * 10                # 「システム管理」ラベルサイズh

SWITCH_TOGGLE_SIZE_W = int(WINDOW_W * 0.2) // 10 * 10                  # トグルスイッチサイズw
SWITCH_TOGGLE_SIZE_H = SWITCH_TOGGLE_SIZE_W / 2                  # トグルスイッチサイズh
LABEL_TOGGLE_SIZE_W = SWITCH_TOGGLE_SIZE_W                   # トグルスイッチ状態表示ラベルサイズw
LABEL_TOGGLE_SIZE_H = LABEL_TOGGLE_SIZE_W / 7                    # トグルスイッチ状態表示ラベルサイズh

BASE_X = (MARGIN_X + VIEW_CAM_SIZE_W) * 2   # システム管理エリア基準x
BASE_Y = MARGIN_Y * 2 + VIEW_CAM_SIZE_H     # システム管理エリア基準y

# --- サブウインドウゾーン ----------------------------
SUB_WINDOW_W, SUB_WINDOW_H = 600, 500   # サブウインドウサイズ

ICON_UP_SPEED_SIZE_W = 300                  # speedアップアイコンサイズw
ICON_UP_SPEED_SIZE_H = 150                  # speedアップアイコンサイズh
LABEL_SPEED_SIZE_W = 300                    # speedラベルサイズw
LABEL_SPEED_SIZE_H = 150                    # speedラベルサイズh
ICON_DOWN_SPEED_SIZE_W = 300                # speedダウンアイコンサイズw
ICON_DOWN_SPEED_SIZE_H = 150                # speedダウンアイコンサイズh

BUTTON_BACK_SIZE = 130                      # 戻るボタンサイズ

# ==========================================================
# スタイル定義
# ==========================================================
# --- メインウインドウゾーン ----------------------------
LABEL_CAM_STYLE = """
background-color: #333333; color: #FFFFFF;
font-size: 20px; font-weight: bold; border-radius: 5px;
qproperty-alignment: 'AlignCenter';
"""
LABEL_HISTORY_STYLE = """
font-family: "MS Gothic";
font-size: 18px; font-weight: bold;
color: #00FF00; background-color: #000000;
border: 2px solid #555555; border-radius: 5px;
qproperty-alignment: 'AlignTop | AlignLeft';;
"""
LABEL_DAMAGE_STYLE = """
font-family: "Meiryo"; font-size: 30px; font-weight: bold;
color: #000000; background-color: #FFFFFF;
border: 1px solid #000000;
qproperty-alignment: 'AlignCenter';
"""
LABEL_MANAGEMENT_STYLE = """
font-family: "Meiryo"; font-size: 40px; font-weight: bold;
color: #000000; background-color: #FFFFFF;
qproperty-alignment: 'AlignCenter';
"""
LABEL_TOGGLE_STYLE = """
font-family: "Meiryo"; font-size: 30px; font-weight: bold;
color: #888888; qproperty-alignment: 'AlignCenter';
"""

# --- サブウインドウゾーン ----------------------------
BUTTON_SUB_STYLE = """
font-family: "Meiryo"; font-size: 20px; font-weight: bold;
color: #00FFFF; background-color: #333333; border-radius: 50px;
"""
LABEL_SPEED_STYLE = """
font-family: "Meiryo"; font-size: 60px; font-weight: bold;
color: #000000; background-color: transparent;
qproperty-alignment: 'AlignCenter';
"""

# ==========================================================
# パス取得関数
# ==========================================================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# ==========================================================
# リサイズ・スムージング処理関数
# ==========================================================
def resize_smooth_image(pixmap, button):
    if not pixmap.isNull():
            # 画像をラベルのサイズに合わせてリサイズ（スムージング処理付き）
            scaled_pixmap = pixmap.scaled(
                button.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            button.setPixmap(scaled_pixmap)

# ==========================================
# クリック可能なラベルクラス
# ==========================================
class ClickableLabel(QLabel):
    clicked = Signal()      # クリックされたときに反応するシグナル
    # --- 左クリック時のみ送信する関数 ------------------------------
    def mousePressEvent(self, event):
        if self.isEnabled and event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
    # --- ボタンをロック・アンロックする関数 ------------------------------
    def set_locked(self, locked: bool):
        """
        locked = True   :   クリック無効＆半透明
        locked = False  :   クリック有効＆通常表示
        """
        self.setEnabled(not locked)
        opacity_effect = QGraphicsOpacityEffect(self)
        if locked:
            opacity_effect.setOpacity(0.3)
        else:
            opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(opacity_effect)

# ==========================================
# ToggleSwitch クラス
# ==========================================
class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None, width=60, height=30):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setCursor(Qt.PointingHandCursor)
        self._bg_color_off = QColor("#B0B0B0")
        self._bg_color_on = QColor("#32CD32")
        self._circle_color = QColor("#FFFFFF")
        self._position = 0.0
        self._animation = QPropertyAnimation(self, b"position")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.stateChanged.connect(self.setup_animation)

    @Property(float)
    def position(self): return self._position
    @position.setter
    def position(self, pos):
        self._position = pos
        self.update()

    def setup_animation(self, value):
        self._animation.stop()
        self._animation.setEndValue(1.0 if value else 0.0)
        self._animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        radius = height / 2

        # 背景色計算
        curr_col = QColor()
        r = self._bg_color_off.red() + (self._bg_color_on.red() - self._bg_color_off.red()) * self._position
        g = self._bg_color_off.green() + (self._bg_color_on.green() - self._bg_color_off.green()) * self._position
        b = self._bg_color_off.blue() + (self._bg_color_on.blue() - self._bg_color_off.blue()) * self._position
        curr_col.setRgb(int(r), int(g), int(b))

        painter.setBrush(QBrush(curr_col))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, width, height, radius, radius)

        # 丸の描画
        circle_r = radius - 3
        circle_x = radius + (width - 2*radius) * self._position
        painter.setBrush(QBrush(self._circle_color))
        painter.drawEllipse(QPointF(circle_x, radius), circle_r, circle_r)

    def hitButton(self, pos): return self.contentsRect().contains(pos)

# ==========================================
# スタートアップウインドウUI
# ==========================================
class StartupWindowUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("起動確認")
        self.setFixedSize(500, 300)
        self.setStyleSheet("background-color: #CCCCCC;")

        self.label_msg = QLabel("システムを開始しますか？", self)
        self.label_msg.setFixedSize(400, 50)
        self.label_msg.move(50, 50)
        self.label_msg.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")

        self.button_start = QPushButton("START SYSTEM", self)
        self.button_start.setFixedSize(260, 80)
        self.button_start.move(120, 150)
        self.button_start.setStyleSheet("background-color: #FF4500; color: white; font-size: 24px; border-radius: 40px;")

# ==========================================
# サブウインドウUI
# ==========================================
class SubWindowUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("設定画面")
        self.setFixedSize(SUB_WINDOW_W, SUB_WINDOW_H)
        self.setStyleSheet("background-color: #EEEEEE;")
        self.setWindowModality(Qt.ApplicationModal)

        # --- speed変更エリア ------------------------------
        # 配置座標
        up_speed_x = MARGIN_X * 5
        up_speed_y = MARGIN_Y * 3  # 上からの余白
        speed_label_x = MARGIN_X * 5
        speed_label_y = ICON_UP_SPEED_SIZE_H + MARGIN_Y * 3
        down_speed_x = MARGIN_X * 5
        down_speed_y = SUB_WINDOW_H - ICON_DOWN_SPEED_SIZE_H - MARGIN_Y * 3  # 下からの余白

        # up_speedボタン
        pixmap = QPixmap(resource_path("Icon/up_speed.png"))
        self.button_up_speed = ClickableLabel(self)
        self.button_up_speed.setFixedSize(ICON_UP_SPEED_SIZE_W, ICON_UP_SPEED_SIZE_H)
        self.button_up_speed.move(up_speed_x, up_speed_y)
        self.button_up_speed.setCursor(Qt.PointingHandCursor)        # マウスを乗せたときに指カーソルにする
        resize_smooth_image(pixmap, self.button_up_speed)

        # 現在の速度数値を表示するラベル
        self.label_current_speed = QLabel("5", self) # 初期値5
        self.label_current_speed.setFixedSize(LABEL_SPEED_SIZE_W, LABEL_SPEED_SIZE_H)
        self.label_current_speed.setStyleSheet(LABEL_SPEED_STYLE)
        self.label_current_speed.move(speed_label_x, speed_label_y)

        # down_speedボタン
        pixmap = QPixmap(resource_path("Icon/down_speed.png"))
        self.button_down_speed = ClickableLabel(self)
        self.button_down_speed.setFixedSize(ICON_DOWN_SPEED_SIZE_W, ICON_DOWN_SPEED_SIZE_H)
        self.button_down_speed.move(down_speed_x, down_speed_y)
        self.button_down_speed.setCursor(Qt.PointingHandCursor)
        resize_smooth_image(pixmap, self.button_down_speed)

        # --- 戻るボタンエリア ------------------------------
        # 配置座標
        back_x = SUB_WINDOW_W - BUTTON_BACK_SIZE - MARGIN_X * 3
        back_y = SUB_WINDOW_H - BUTTON_BACK_SIZE - MARGIN_Y * 3

        self.button_back = QPushButton("Back", self)
        self.button_back.setFixedSize(BUTTON_BACK_SIZE, BUTTON_BACK_SIZE)
        self.button_back.setStyleSheet(BUTTON_SUB_STYLE)
        self.button_back.move(back_x, back_y)
        self.button_back.setCursor(Qt.PointingHandCursor)


# ==========================================
# メインウインドウUI
# ==========================================
class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("サクランボ病害虫除去システム")
        self.setFixedSize(WINDOW_W, WINDOW_H)
        self.setStyleSheet("background-color: #FFFFFF;")

        # --- カメラ管理エリア ------------------------------
        # 配置座標
        cam_x_left = MARGIN_X
        cam_x_right = VIEW_CAM_SIZE_W + MARGIN_X * 2
        cam_y_upper = MARGIN_Y
        cam_y_lower = VIEW_CAM_SIZE_H + MARGIN_Y * 2

        self.cam_in = QLabel("camera_inside", self)
        self.cam_in.setFixedSize(VIEW_CAM_SIZE_W, VIEW_CAM_SIZE_H); self.cam_in.setStyleSheet(LABEL_CAM_STYLE)
        self.cam_in.move(cam_x_left, cam_y_upper)

        self.cam_out = QLabel("camera_outside", self)
        self.cam_out.setFixedSize(VIEW_CAM_SIZE_W, VIEW_CAM_SIZE_H); self.cam_out.setStyleSheet(LABEL_CAM_STYLE)
        self.cam_out.move(cam_x_right, cam_y_upper)

        self.cam_under = QLabel("camera_underside", self)
        self.cam_under.setFixedSize(VIEW_CAM_SIZE_W, VIEW_CAM_SIZE_H); self.cam_under.setStyleSheet(LABEL_CAM_STYLE)
        self.cam_under.move(cam_x_left, cam_y_lower)

        self.cam_top = QLabel("camera_topside", self)
        self.cam_top.setFixedSize(VIEW_CAM_SIZE_W, VIEW_CAM_SIZE_H); self.cam_top.setStyleSheet(LABEL_CAM_STYLE)
        self.cam_top.move(cam_x_right, cam_y_lower)

        # --- 設定エリア ------------------------------
        # 配置座標
        setting_x = BASE_X + MARGIN_X
        setting_y = MARGIN_Y

        pixmap = QPixmap(resource_path("Icon/setting.png"))
        self.button_setting = ClickableLabel(self)
        self.button_setting.setFixedSize(ICON_SETTING_SIZE, ICON_SETTING_SIZE)
        self.button_setting.move(setting_x, setting_y)
        self.button_setting.setCursor(Qt.PointingHandCursor)    # マウスを乗せたときに指カーソルにする
        resize_smooth_image(pixmap, self.button_setting)

        # --- 電源エリア ------------------------------
        # 配置座標
        power_x = WINDOW_W - ICON_POWER_SIZE - MARGIN_X
        power_y = MARGIN_Y

        pixmap = QPixmap(resource_path("Icon/power_supply.png"))
        self.button_power = ClickableLabel(self)
        self.button_power.setFixedSize(ICON_POWER_SIZE, ICON_POWER_SIZE)
        self.button_power.move(power_x, power_y)
        self.button_power.setCursor(Qt.PointingHandCursor)
        resize_smooth_image(pixmap, self.button_power)

        # --- 判定履歴表示エリア ------------------------------
        # 配置座標
        history_x = WINDOW_W - LABEL_HISTORY_SIZE_W - MARGIN_X
        history_y = setting_y + ICON_SETTING_SIZE + MARGIN_Y

        self.label_history = QLabel("Wait for input...", self)
        self.label_history.setFixedSize(LABEL_HISTORY_SIZE_W, LABEL_HISTORY_SIZE_H)
        self.label_history.setStyleSheet(LABEL_HISTORY_STYLE)
        self.label_history.move(history_x, history_y)

        # --- 病害管理エリア ------------------------------
        # 配置座標
        dam_x = WINDOW_W - LABEL_DAMAGE_SIZE_W - MARGIN_X
        dam_y = history_y + LABEL_HISTORY_SIZE_H + MARGIN_Y

        # 病害名表示ラベル
        self.label_dam = QLabel("病害結果", self)
        self.label_dam.setFixedSize(LABEL_DAMAGE_SIZE_W, LABEL_DAMAGE_SIZE_H)
        self.label_dam.setStyleSheet(LABEL_DAMAGE_STYLE)
        self.label_dam.move(dam_x, dam_y)

        # --- システム管理エリア ------------------------------
        # 配置座標
        label_management_x = (WINDOW_W * 0.95) - LABEL_MANAGEMENT_SIZE_W - MARGIN_X
        label_management_y = dam_y + LABEL_DAMAGE_SIZE_H
        toggle_switch_x = (WINDOW_W * 0.95) - LABEL_MANAGEMENT_SIZE_W - MARGIN_X
        toggle_switch_y = label_management_y + LABEL_MANAGEMENT_SIZE_H
        toggle_status_x = toggle_switch_x
        toggle_status_y = toggle_switch_y + SWITCH_TOGGLE_SIZE_H + MARGIN_Y

        # システム状態表示ラベル
        self.label_panel = QLabel("システム管理", self)
        self.label_panel.setFixedSize(LABEL_MANAGEMENT_SIZE_W, LABEL_MANAGEMENT_SIZE_H)
        self.label_panel.setStyleSheet(LABEL_MANAGEMENT_STYLE)
        self.label_panel.move(label_management_x, label_management_y)

        # トグルスイッチ
        self.toggle_switch = ToggleSwitch(self, SWITCH_TOGGLE_SIZE_W, SWITCH_TOGGLE_SIZE_H)
        self.toggle_switch.move(toggle_switch_x, toggle_switch_y)
        self.button_power.setCursor(Qt.PointingHandCursor)

        # トグルスイッチ状態表示ラベル
        self.label_toggle_status = QLabel("停止中", self)
        self.label_toggle_status.setFixedSize(LABEL_TOGGLE_SIZE_W, LABEL_TOGGLE_SIZE_H)
        self.label_toggle_status.setStyleSheet(LABEL_TOGGLE_STYLE)
        self.label_toggle_status.move(toggle_status_x, toggle_status_y)