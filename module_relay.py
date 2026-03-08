# -------------------------------------------------
# リレーボードをYOLOの結果により制御するプログラムmodule
# -------------------------------------------------
import platform
import ctypes
import time
import os
from enum import IntEnum

#import module_yolo_csv as yolo_csv
# ================================================
# 定数・設定定義
# ================================================
YDCI_RESULT_SUCCESS = 0     # 正常終了
YDCI_OPEN_NORMAL = 0        # YdciOpen
RELAY_OPEN_TIME = 0.2       # リレーの開閉時間（秒）
RATIO = 1.0                 # 基本補正係数
MICRO_STATUS = 32           # マイクロステップ設定
class RelayState(IntEnum):
    """リレーの状態定義"""
    OPEN = 0   # 回路を開く
    CLOSE = 1  # 回路を閉じる
class RelayChannel(IntEnum):
    """チャンネル定義"""
    REMOVE = 0    # 被害果除去用
    TRANSPORT = 1 # 健全果運搬用

SPEED_MAP = {
    1: 0.0010,  # 回転遅い
    2: 0.0009,
    3: 0.0008,
    4: 0.0007,
    5: 0.0006,  # 基準 (デフォルト)
    6: 0.0005,
    7: 0.0004,
    8: 0.0003,
    9: 0.0002,
    10: 0.0001  # 回転速い
}
# ================================================
# メインクラス定義
# ================================================
class RelayController():
    def __init__(self):
        """コンストラクタ: 変数の初期化"""
        self.ydci = None
        self.board_id = ctypes.c_ushort()
        self.is_connected = False

    # --- リレーボード初期化 + 接続関数 -------------------
    def init(self):
        # DLLのロード
        pf = platform.system()
        if pf == 'Windows':
            try:
                self.ydci = ctypes.windll.Ydci
            except OSError:
                print("エラー: Ydci.dll が見つかりません。")
                return False
        else:
            print(f"サポートされていないOSです: {pf}")
            return False

        # ボード識別スイッチが0のボードをオープン
        result_board = self.ydci.YdciOpen(self.board_id, b'RLY-P4/2/0B-UBT', ctypes.byref(self.board_id), YDCI_OPEN_NORMAL)
        if result_board != YDCI_RESULT_SUCCESS:
            print(f'オープンできません。エラーコード: {result_board}')
            self.ydci = None
            return False
        print(f">>> リレーボード({self.board_id.value})接続成功")
        self.is_connected = True

        # 初期状態設定:
        self._set_state(RelayChannel.REMOVE, RelayState.CLOSE)       # 被害果除去用Ch（0）をClose
        self._set_state(RelayChannel.TRANSPORT, RelayState.CLOSE)    # 健全果運搬用Ch（1）をClose
        return True

    # --- リレーの状態を設定する関数 -------------------
    def _set_state(self, channel, state):
        """
        [ydci.YdciRlyOutput()]
        board_id -> リレー制御ボードを識別するための変数。ctypes.c_ushort 型で定義し、YdciRlyOpen が正常に実行されると、この変数にボードIDが格納されます。
        ctypes.byref(output_data) -> relay_ON(0),relay_OFF(1)
        start_channel -> 操作を開始するチャネルの番号
        num_channel -> 操作するチャネルの総数
        """
        if self.ydci is None:
            print("エラー: リレーボードが初期化されていません。")
            return False
        output_data = ctypes.c_ubyte(state)
        result = self.ydci.YdciRlyOutput(self.board_id, ctypes.byref(output_data), channel, 1)
        if result != YDCI_RESULT_SUCCESS:
            print(f'リレー Ch{channel} の状態設定に失敗しました。エラーコード: {result}')
            return False
        return True

    # --- PCから受けとったspeed値から、上外カメラ撮影位置からの待機時間を計算し、セットする内部関数 -------------------
    def _set_wait_time(self, speed):
        if not self.is_connected:
            print("警告: ボード未接続のためパルス動作をスキップします。")
            return

        # 計算ロジック
        delay = SPEED_MAP[speed]
        t_one_pulse = delay * 2
        step_one_rotation = RATIO * (360 / 1.8) * MICRO_STATUS
        sec = t_one_pulse * step_one_rotation * 2   # ギア比が2なので

        # チャンネルごとに待機時間を調整してセット
        remove_channel_wait = sec * (90 / 360)
        transport_channel_wait = sec * (135 / 360)
        #print(delay)
        #print(remove_channel_wait)
        #print(transport_channel_wait)

        return remove_channel_wait, transport_channel_wait

    # --- 指定したChのリレーを動作させる関数 -------------------
    def move(self, channel, speed):
        remove_wait_sec, transport_wait_sec = self._set_wait_time(speed)

        if channel == RelayChannel.REMOVE:
            wait_sec = remove_wait_sec
        elif channel == RelayChannel.TRANSPORT:
            wait_sec = transport_wait_sec
        else:
            print("エラー: 不正なチャンネルが指定されました。")
            return

        # 動作シーケンス
        time.sleep(wait_sec)  # 待機時間
        self._set_state(channel, RelayState.OPEN)
        time.sleep(RELAY_OPEN_TIME) # 噴射時間
        self._set_state(channel, RelayState.CLOSE)

    # --- リレー停止関数 -------------------
    def stop(self):
        if not self.is_connected:
            print("警告: ボード未接続のため停止動作をスキップします。")
            return
        self._set_state(RelayChannel.REMOVE, RelayState.CLOSE)
        self._set_state(RelayChannel.TRANSPORT, RelayState.CLOSE)

    # --- リレーボード接続終了関数 -------------------
    def close(self):
        if self.ydci is not None and self.is_connected:
            # 安全のため終了前に閉じる
            self._set_state(RelayChannel.REMOVE, RelayState.CLOSE)
            self._set_state(RelayChannel.TRANSPORT, RelayState.CLOSE)

            self.ydci.YdciClose(self.board_id)
            self.ydci = None
            self.is_connected = False
        print(">>> リレーボード切断完了")

    # --- インスタンス破棄時に自動的に閉じる関数 -------------------
    def __del__(self):
        self.close()