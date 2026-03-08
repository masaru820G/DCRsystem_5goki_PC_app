# -------------------------------------------------
# パトライトをYOLOの結果により制御するプログラムmodule
# -------------------------------------------------
import hid
import time

# ================================================
# 定数・設定定義
# ================================================
VENDER_ID = 0x191a   # ベンダーID指定
PRODUCT_ID = 0x6001  # 製品ID指定
class LedPattern():
    """
    >>> LEDの制御値とデバッグ表示名を管理するクラス
    構成: (Byte5の16進数値, デバッグ用表示名)
    """
    OFF    = (0x00, "消灯")
    RED    = (0x11, "赤")
    GREEN  = (0x21, "緑")
    YELLOW = (0x31, "黄")
    BLUE   = (0x41, "青")
    VIOLET = (0x51, "紫")
    SKY    = (0x61, "空")
    WHITE  = (0x71, "白")

# ================================================
# メインクラス定義
# ================================================
class PatliteController():
    def __init__(self):
        # 変数の初期化
        self.device = None

    # --- パトライト初期化(hid device）) + 接続関数 -------------------
    def init(self):
        try:
            # 既に接続されている場合は何もしない
            if self.device:
                return True

            # デバイスに接続
            self.device = hid.device()
            self.device.open(VENDER_ID, PRODUCT_ID)
            print(">>> パトライト接続成功")

            # 初期状態として消灯・ブザー停止
            time.sleep(1.0)
            self.set_color(LedPattern.OFF)
            return True
        except Exception as e:
            print(f"接続エラー: {e}")
            self.device = None
            return False

    # --- パトライトに制御コマンドを送信する関数 -------------------
    def _send_command(self, data):
        if self.device is None:
            print("エラー: パトライトが初期化されていません。")
            return False

        try:
            self.device.write(data)
            return True
        except Exception as e:
            print(f"{e}: 書き込み失敗。")
        return False

    # --- パトライト制御関数 -------------------
    def set_color(self, pattern = LedPattern.OFF):
        # patternは (数値, 名前) のタプルなので分解する
        self.led_byte, self.color_name = pattern
        self.data = [0] * 9  #データの初期化9Bytes

        self.data[1] = 0x00      # コマンドバージョン
        self.data[2] = 0x00      # コマンドID
        self.data[3] = 0x07      # ブザー制御
        self.data[4] = 0x00      # ブザー音量
        self.data[5] = self.led_byte  # LED制御
        self.data[6] = 0x00
        self.data[7] = 0x00
        self.data[8] = 0x00

        return self._send_command(self.data), self.color_name

    # --- パトライト接続終了関数 -------------------
    def close(self):
        if self.device:
            self.set_color(LedPattern.OFF) # 終了時に消灯
            self.device.close()
            self.device = None
            print(">>> パトライト切断完了")