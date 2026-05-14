# DCRsystem 5号機 PC アプリケーション

> **D**isease **C**herry **R**emoval System — サクランボ病害果除去システム

サクランボの病害果をリアルタイムで検出・除去するための制御アプリケーションです。  
4台の産業用カメラ（Basler）と YOLO 物体検出モデルを使用し、コンベア上を流れるサクランボを撮影・判定し、エアーブロー（リレーボード制御）で被害果を自動除去します。

---

## 📋 システム概要

### 検出クラス（6種類）

| クラス名 | 表示名（JP） | パトライト色 | 動作 |
|---|---|---|---|
| `healthy` | 健全果 | 白 | 運搬（通過） |
| `twin` | 双子果 | 赤 | 除去（エアーブロー） |
| `unripe` | 未熟果 | 黄 | 除去（エアーブロー） |
| `mold` | カビ | 紫 | 除去（エアーブロー） |
| `stemcrack` | 果梗裂果 | 青 | 除去（エアーブロー） |
| `birddamage` | 鳥害 | 空 | 除去（エアーブロー） |

### 主な機能

- **リアルタイム映像表示**: 4台のカメラ映像を GUI 上に同時表示
- **YOLOv8 推論**: 学習済みモデルによるサクランボの病害検出
- **自動除去制御**: 判定結果に基づくリレーボード制御（エアーブロー）
- **パトライト連動**: 検出クラスに応じた LED 色表示
- **Raspberry Pi 連携**: ステッピングモータの回転速度制御（HTTP通信）
- **速度調整**: GUI からコンベア速度を 10 段階で変更可能
- **カメラエラー復旧**: カメラ接続ロスト時の自動検知と復旧ウィンドウ
- **統計・履歴表示**: 検出カウントと直近 10 件の判定履歴を表示

---

## 🏗️ システム構成

### ハードウェア

```
┌──────────────┐    HTTP     ┌────────────────────┐
│   Windows PC │ ◄────────► │  Raspberry Pi      │
│  (本アプリ)   │             │  (モータ制御)       │
└──────┬───────┘             └────────────────────┘
       │ USB
       ├── Basler カメラ × 4台 (pypylon)
       │     ├── cam_top      (上面カメラ)
       │     ├── cam_under    (下面カメラ)
       │     ├── cam_inside   (内側カメラ)
       │     └── cam_outside  (外側カメラ)
       │
       ├── リレーボード (Ydci DLL / ctypes)
       │     ├── Ch0: 運搬用 (TRANSPORT)
       │     └── Ch1: 除去用 (REMOVE)
       │
       └── パトライト (HID USB)
             └── LED 色表示
```

### ソフトウェア構成

```
DCRsystem_5goki_PC_app/
│
├── main_5goki_JP.py           # メインエントリポイント（日本語版）
├── main_5goki_ENG.py          # メインエントリポイント（英語版）
├── main_5goki.py              # メインエントリポイント（旧版）
│
├── module_gui_JP.py           # GUI レイアウト定義（日本語版）
├── module_gui_ENG.py          # GUI レイアウト定義（英語版）
├── module_gui.py              # GUI レイアウト定義（旧版）
│
├── module_cameras_5goki.py    # Basler カメラ制御モジュール
├── module_relay.py            # リレーボード制御モジュール
├── module_patlite.py          # パトライト制御モジュール
├── module_yolo_csv3.py        # YOLO推論 & CSV出力モジュール
│
├── main_calibration.py        # カメラキャリブレーション用ツール
├── first.py                   # 初期動作確認用スクリプト
│
├── Trained_Models/            # YOLO学習済みモデル (※Git管理外)
│   └── best2.pt
├── cam_pfs/                   # カメラ設定ファイル (.pfs)
├── Icon/                      # GUIアイコン画像
│
├── pyproject.toml             # プロジェクト設定・依存関係
├── uv.lock                    # 依存関係ロックファイル
└── .python-version            # Python バージョン指定 (3.12.12)
```

### モジュール依存関係

```
main_5goki_JP.py
├── module_gui_JP.py        ... GUI（PySide6）
├── module_cameras_5goki.py ... カメラ制御（pypylon）
├── module_relay.py         ... リレーボード（ctypes / Ydci.dll）
├── module_patlite.py       ... パトライト（hidapi）
└── module_yolo_csv3.py     ... YOLO推論（ultralytics / PyTorch）
```

---

## 🚀 セットアップ・実行方法

### 前提条件

- **OS**: Windows 10/11
- **Python**: 3.12 以上
- **GPU**: CUDA 12.4 対応の NVIDIA GPU（推論高速化に必要）
- **パッケージマネージャ**: [uv](https://docs.astral.sh/uv/)（推奨）

### 1. リポジトリのクローン

```bash
git clone <リポジトリURL>
cd DCRsystem_5goki_PC_app
```

### 2. 依存パッケージのインストール

```bash
# uv を使用する場合（推奨）
uv sync

# pip を使用する場合
pip install -e .
```

> **注意**: PyTorch は CUDA 12.4 対応版が自動的にインストールされます（`pyproject.toml` の `[tool.uv.sources]` で設定済み）。

### 3. 学習済みモデルの配置

`Trained_Models/` ディレクトリに YOLO の学習済みモデル `best2.pt` を配置してください。

```
Trained_Models/
└── best2.pt
```

### 4. ハードウェア接続

以下のデバイスを PC に USB 接続してください：

1. **Basler カメラ × 4台** — Pylon ドライバがインストール済みであること
2. **リレーボード** — Ydci.dll がシステムに登録済みであること
3. **パトライト** — HID デバイスとして認識されること

### 5. Raspberry Pi の起動

Raspberry Pi 側のモータ制御サーバーを起動し、ネットワーク接続を確認してください。

- IP アドレス: `192.168.2.1`
- ポート: `5000`

### 6. アプリケーションの実行

```bash
# 日本語版
uv run python main_5goki_JP.py

# 英語版
uv run python main_5goki_ENG.py
```

起動すると確認ウィンドウが表示されます。「システム起動」ボタンを押すとメイン画面がフルスクリーンで開きます。

---

## 🎮 操作方法

| 操作 | 説明 |
|---|---|
| **トグルスイッチ ON** | コンベア回転開始 & YOLO推論開始 |
| **トグルスイッチ OFF** | コンベア停止 & プレビューモード（推論なし） |
| **設定アイコン** | サブウィンドウを開き、コンベア速度（1〜10）を調整 |
| **電源アイコン** | 全デバイスを安全に停止してアプリを終了 |

---

## 📦 主要な依存パッケージ

| パッケージ | 用途 |
|---|---|
| `PySide6` | GUI フレームワーク（Qt for Python） |
| `opencv-python` | 画像処理・フレーム変換 |
| `ultralytics` | YOLOv8 推論エンジン |
| `torch` / `torchvision` | 深層学習フレームワーク (CUDA 12.4) |
| `pypylon` | Basler カメラ SDK |
| `hidapi` | パトライト HID 通信 |
| `PyInstaller` | 実行ファイル (.exe) 生成 |
