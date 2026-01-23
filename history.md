### 2025-12-04

- **リポジトリ更新**: Git のパス問題を解消し、`git pull` で最新状態に更新。
- **アニメーション保存修正**: `TwoEWInterfaceVessel.py` から切り出した `export_vessel_animation` を改良し、ffmpeg 非インストール環境では GIF 保存に自動切り替え。
- **モジュール分割**: モデル本体の関数を `vessel_model.py` に分離（`TwoVesselModel`, `plotVessel`, `export_vessel_animation`）。
- **パラメータサーチ用スクリプト**:  
  - `param_search_ks.py`: `ks=0〜9` を掃引し、最終状態をグリッド画像 `results/param_search_ks_grid.png` として保存（tqdm で進捗表示、有効な σ はデフォルト値）。  
  - `param_search_sigma.py`: `ks=0` 固定、`sigma=[1,10,100]` の結果を横並びにしたグリッド画像 `results/param_search_sigma_grid.png` を保存（tqdm で進捗表示）。
- **デフォルトパラメータ変更**: `vessel_model.py` 内でノイズ強度のデフォルトを `sigma=4` に変更し、この条件下での `ks` パラメータサーチが行えるように整理。

### 2026-01-09

- **README 更新**: `README.md` に `## 進捗` セクションを追加し、`history.md` を参照する旨を追記。
- **EW方程式の振幅パラメータ依存性調査**: `EW.ipynb` において、1次元 EW（Edwards–Wilkinson）方程式の振幅に対する各パラメータの依存性を調査。
  - `dh`（拡散係数）依存性: `dh=0.01〜0.2` の範囲で振幅を計算し、`√dh` との関係を可視化。
  - `ks`（バネ係数）依存性: `ks=0.2〜4` の範囲で振幅を計算し、`ks` との関係を可視化。
  - `L`（系のサイズ）依存性: `L=50〜1000` の範囲で振幅を計算し、`L` との関係を可視化。
  - `T`（時間）依存性: `T=100〜2400` の範囲で振幅を計算し、`T` との関係を可視化。
  - 各パラメータについて20回のシミュレーションを実行し、平均振幅を算出。