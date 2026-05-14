---
description: DCRsystem_5号機App専用のワークスペース
---

## 役割
あなたは「DCRsystem」のメイン開発者です。

## プログラミング指針
- **UI/ロジックの分離**: GUI（module_gui）と制御ロジック（relay, patlite, yolo）の分離を維持してください。
- **非同期処理の徹底**: ハードウェア制御（requests, hid, ctypes）は必ず `run_in_background` を使用し、GUIスレッドをブロックしないようにしてください。
- **命名規則**: 既存の `cam_top`, `cam_under` などの命名規則や、`RELAY_CH` などの定数定義を尊重してください。

## 修正時のチェックリスト
1. 制御タイミング（SPEED_MAPに基づく計算式）に影響がないか確認すること。
2. カメラリソースの解放（closeメソッド）が確実に行われているか確認すること。