# TwoEWInterfaceVesselModel

血管壁の動きをシミュレーションする数値計算モデルです。2つの血管の界面の時間発展を、拡散、ランダムノイズ、弾性相互作用を考慮して計算します。

## 概要

このプロジェクトは、血管壁の界面位置の時間発展を記述する偏微分方程式を数値的に解くことで、血管の形態変化をシミュレーションします。特に、炎症時の細胞間接着の減少や、周辺組織との相互作用を考慮したモデルです。

## 進捗

- **2025-12-04**
  - Git のパス問題を解消し、`git pull` で最新状態に更新
  - アニメーション保存の改善: `export_vessel_animation` を改良し、ffmpeg 非インストール環境では GIF 保存に自動切り替え
  - モジュール分割: モデル本体の関数を `vessel_model.py` に分離（`TwoVesselModel`, `plotVessel`, `export_vessel_animation`）
  - パラメータサーチ用スクリプト追加:
    - `param_search_ks.py`: `ks=0〜9` 掃引
    - `param_search_sigma.py`: `ks=0` 固定、`sigma=[1,10,100]` を比較
  - デフォルトパラメータ変更: `vessel_model.py` 内でノイズ強度のデフォルトを `sigma=4` に変更
  - `EW.ipynb` 追加: 1 次元 EW（Edwards–Wilkinson）方程式の簡易シミュレーション/検証用

※詳細ログは `history.md` を参照してください。

## 支配方程式

4つの界面位置 $h_1, h_2, h_3, h_4$ の時間発展は以下の偏微分方程式で記述されます：

$$\partial_t h_1= d_h \partial_{x}^2 h_1 + \sigma \eta_1(x,t) -k_s/w_1\big(h_1-(H-w_1)\big)+k_e/(2r)(h_2-h_1+2r)$$

$$\partial_t h_2= d_h \partial_{x}^2 h_2 + \sigma \eta_2(x,t) -k_e/(2r)(h_2-h_1+2r)+k_s/w_2(h_3-h_2+w_2)$$

$$\partial_t h_3= d_h \partial_{x}^2 h_3 + \sigma \eta_3(x,t) -k_s/w_2(h_3-h_2+w_2)+k_e/(2r)(h_4-h_3+2r)$$

$$\partial_t h_4= d_h \partial_{x}^2 h_4 + \sigma \eta_4(x,t) -k_e/(2r)(h_4-h_3+2r)-k_s/w_1(h_4+H-w_1)$$

### パラメータの説明

- **$h_i(x,t)$**: 時刻$t$で場所$x$の$i$番目の血管の界面の位置
- **$d_h$**: 血管壁の表面張力。炎症時には細胞間接着が外れて減少する
- **$\sigma$**: 内皮細胞のランダムな運動の強さ
- **$\eta_i(x,t)$**: 平均0、分散1の正規分布の乱数（標準正規乱数）
- **$k_s$**: 周辺の結合組織の硬さ（単位面積あたりのバネ係数）
- **$k_e$**: 血管壁同士の相互作用の単位長さあたりのバネ係数
- **$H$**: 血管系全体の高さの範囲（$H = w_2/2 + 2r + w_1$で自動計算）
- **$w_1$**: 血管と表皮の間の間隔（デフォルト: 1）100µm
- **$w_2$**: 血管同士の間の間隔（デフォルト: 2）200µｍ
- **$r$**: 血管の半径（デフォルト: 0.5）8 µｍ

- 結合組織のヤング率　E_s = 300Pa [https://pmc.ncbi.nlm.nih.gov/articles/PMC12477866]
  - $k_s = E_s dx dz / w_1$ 
- 血管のヤング率 E_e = 1kPa [https://pmc.ncbi.nlm.nih.gov/articles/PMC3714930/]
- 基底膜の厚み：t_{BM} = 0.1µm
- 基底膜の物性：E_{BM} = 1MPa [https://www.sciencedirect.com/science/article/pii/S0945053X10000521?via%3Dihub]
- 基底膜由来の血管のバネ係数 k_{BM} = E_{BM} t_{BM} dx/r 




## セットアップ

### Windowsユーザー向け

Windowsでこのリポジトリをcloneするには、まず**Git for Windows**をインストールする必要があります。

1. **Git for Windowsのインストール**
   - [Git for Windows](https://git-scm.com/download/win) からダウンロードしてインストール
   - インストール中はデフォルト設定で問題ありません
   - インストール後、コマンドプロンプトまたはPowerShellを開いて `git --version` でインストールを確認

2. **GitHub Desktop（オプション）**
   - GUIツールを使いたい場合は、[GitHub Desktop](https://desktop.github.com/) も利用できます
   - GitHub Desktopを使う場合は、Git for Windowsは自動的にインストールされます

3. **日本語を含むパス名に関する注意**
   - Windowsで日本語を含むパス名（例：`C:\Users\ユーザー名\ドキュメント\研究`）を使用すると、文字化けやエラーが発生する可能性があります
   - **推奨**: リポジトリをcloneする際は、英語のみのパス名を使用してください
     - 例：`C:\Users\username\Documents\research` または `C:\projects\TwoEWInterfaceVesselModel`
   - 日本語パスを使用する場合は、以下の設定を試してください：
     ```bash
     git config --global core.quotepath false
     git config --global i18n.commitencoding utf-8
     git config --global i18n.logoutputencoding utf-8
     ```

4. **PowerShellでの環境変数の設定**
   - **現在のセッションのみ有効**（一時的）:
     ```powershell
     $env:変数名 = "値"
     ```
     例：`$env:PYTHONIOENCODING = "utf-8"`
   
   - **永続的に設定**（ユーザー環境変数）:
     ```powershell
     [System.Environment]::SetEnvironmentVariable("変数名", "値", "User")
     ```
     例：`[System.Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")`
   
   - **環境変数の確認**:
     ```powershell
     $env:変数名
     ```
     または
     ```powershell
     [System.Environment]::GetEnvironmentVariable("変数名", "User")
     ```
   
   - **文字エンコーディング関連の環境変数設定例**:
     ```powershell
     # UTF-8エンコーディングを設定
     [System.Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
     [System.Environment]::SetEnvironmentVariable("LANG", "ja_JP.UTF-8", "User")
     ```
     
     **注意**: 環境変数を変更した後は、PowerShellを再起動する必要があります。

5. **PATH環境変数の格納場所**
   - Windowsでは、`PATH`環境変数は**レジストリ**に格納されています
   - **ユーザー環境変数**:
     - レジストリパス: `HKEY_CURRENT_USER\Environment`
     - PowerShellでの確認:
       ```powershell
       [System.Environment]::GetEnvironmentVariable("PATH", "User")
       ```
   - **システム環境変数**:
     - レジストリパス: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment`
     - PowerShellでの確認:
       ```powershell
       [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
       ```
   - **現在のセッションで有効なPATH**（ユーザー + システム）:
     ```powershell
     $env:PATH
     ```
   - **PATHに新しいパスを追加**:
     ```powershell
     # ユーザー環境変数のPATHに追加
     $currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
     $newPath = "C:\新しい\パス"
     [System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;$newPath", "User")
     ```
   - **GUIで確認・編集する方法**:
     - 「システムのプロパティ」→「環境変数」から確認・編集可能
     - または、Windows検索で「環境変数」と入力して開く

### macOS/Linuxユーザー向け

macOSやLinuxでは、通常Gitは既にインストールされています。インストールされていない場合は、以下のコマンドでインストールできます：

- **macOS**: `brew install git` (Homebrewを使用する場合)
- **Linux (Ubuntu/Debian)**: `sudo apt-get install git`
- **Linux (CentOS/RHEL)**: `sudo yum install git`

## リポジトリの取得

このリポジトリをcloneするには、以下のいずれかの方法を使用してください：

**重要（Windowsユーザー）**: 日本語を含むパス名（例：`ドキュメント\研究`）は文字化けの原因となる可能性があります。可能な限り、英語のみのパス名（例：`Documents\research`）を使用してください。

### HTTPS（推奨）

```bash
git clone https://github.com/miuraTakashi/TwoEWInterfaceVesselModel.git
```

**注意**: publicリポジトリの場合、認証は不要です。パスワードを要求された場合は、リポジトリがprivateになっている可能性があります。その場合は、リポジトリの設定を確認するか、SSH方式を使用してください。

### SSH

```bash
git clone git@github.com:miuraTakashi/TwoEWInterfaceVesselModel.git
```

SSH方式を使用する場合は、事前にSSH鍵をGitHubに登録しておく必要があります。

## 必要な環境

- Python 3.x
- NumPy
- Matplotlib
- tqdm
- Jupyter Notebook（オプション、ノートブックを使用する場合のみ）

## インストール

### requirements.txtを使用する場合（推奨）

```bash
pip install -r requirements.txt
```

### 個別にインストールする場合

```bash
pip install numpy matplotlib tqdm jupyter
```

**注意**: Jupyter Notebookは、`.ipynb`ファイルを使用する場合のみ必要です。`.py`ファイルのみを使用する場合は不要です。

## 使い方

このプロジェクトには2つの形式のファイルがあります：
- **`TwoEWInterfaceVessel.py`**: Pythonスクリプト（推奨、LLMを介した修正が容易）
- **`TwoEWInterfaceVessel.ipynb`**: Jupyter Notebook（対話的な実行に便利）

### Pythonスクリプト（.py）を使用する場合

#### 基本的なシミュレーション

```python
from TwoEWInterfaceVessel import TwoVesselModel, plotVessel

# シミュレーション実行
result = TwoVesselModel(L=5, dh=0.1, sigma=1, ks=0.1, ke=0.1, T=100)

# 最終状態をプロット
plotVessel(result[-1])
```

#### コマンドラインから直接実行

```bash
python TwoEWInterfaceVessel.py
```

これにより、デフォルトパラメータでシミュレーションが実行され、使用例が表示されます。

### Jupyter Notebook（.ipynb）を使用する場合

ノートブックを開いて、セルを順番に実行してください。

### パラメータの説明

`TwoVesselModel`関数の主なパラメータ：

- **L**: 空間領域の長さ（デフォルト: 5）
- **dh**: 表面張力係数 $d_h$（デフォルト: 1）
- **sigma**: ノイズ強度 $\sigma$（デフォルト: 1）
- **ks**: 結合組織の硬さ $k_s$（デフォルト: 0.1）
- **ke**: 血管間相互作用 $k_e$（デフォルト: 0.1）
- **w1**: 血管と表皮の間隔（デフォルト: 1）
- **w2**: 血管間の間隔（デフォルト: 2）
- **r**: 血管の半径（デフォルト: 0.5）
- **T**: シミュレーション時間（デフォルト: 1000）
- **fps**: アニメーションのフレームレート（デフォルト: 20）
- **progress**: 進捗バーの表示（デフォルト: True）

### アニメーションの生成

```python
from TwoEWInterfaceVessel import export_vessel_animation

# MP4ファイルとして保存
filename = export_vessel_animation(dh=0.1, sigma=1, ks=0.1, ke=0.1, T=100)
```

生成されるファイル名は `vessel_dh{dh}_sigma{sigma}_ks{ks}_ke{ke}.mp4` の形式です。

### Jupyter Notebookでの可視化

```python
from matplotlib import animation
from IPython.display import HTML, display

fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    plotVessel(result[i], ax=ax)
    ax.set_title(f"Frame {i}")

ani = animation.FuncAnimation(fig, animate, frames=len(result), interval=50, repeat=False)
display(HTML(ani.to_jshtml()))
```

### 統計解析

複数のシミュレーションを実行してフーリエ解析を行う例：

```python
resultList = []
for i in range(100):
    result = TwoVesselModel(T=100)
    resultList.append(result)

# フーリエ変換
h1hatList = []
for i in range(100):
    h1hatList.append(np.abs(np.fft.fft(resultList[i][-1][0])))

h1hatMean = np.array(h1hatList).mean(axis=0)
plt.plot(h1hatMean)
plt.xscale('log')
plt.yscale('log')
plt.show()
```

## ファイル構成

- `TwoEWInterfaceVessel.py`: Pythonスクリプト（推奨、LLMを介した修正が容易）
- `TwoEWInterfaceVessel.ipynb`: Jupyterノートブック（対話的な実行に便利）
- `requirements.txt`: 必要なPythonパッケージのリスト
- `vessel_*.mp4`: 生成されたアニメーションファイル（パラメータごと）

## 数値計算の詳細

- 空間離散化: $\Delta x = 0.1$
- 時間離散化: $\Delta t = 0.001$
- 境界条件: 周期境界条件（両端の値は固定）
- 数値解法: オイラー法（拡散項は有限差分法で離散化）

## ライセンス

このプロジェクトは研究目的で作成されています。

## 参考文献

血管壁の動きの数理モデルに関する研究に基づいています。

