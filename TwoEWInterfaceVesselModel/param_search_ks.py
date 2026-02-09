"""
param_search_ks.py

`vessel_model.py` を用いて、ks を 0〜9 の範囲で振り、
最終時刻の形状をグリッド表示するスクリプト。
"""

from pathlib import Path

import matplotlib.pyplot as plt

from vessel_model import TwoVesselModel, plotVessel


def main():
    # 結果の保存ディレクトリ（必要なら生成）
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # グリッドレイアウトの設定（2 行 x 5 列 = 10 個）
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ks in enumerate(range(10)):
        ax = axes[i]

        # 計算（T は時間・計算量に応じて調整可）
        # progress=True として tqdm による進捗を表示
        result = TwoVesselModel(ks=ks, T=100, progress=True)

        # 最終フレームを描画
        plotVessel(result[-1], ax=ax, save_path=None)
        ax.set_title(f"ks={ks}")

    # 全体タイトルとレイアウト調整
    fig.suptitle("TwoVesselModel - ks parameter sweep (0–9)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # PNG 保存（画面には表示しない）
    output_path = results_dir / "param_search_ks_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"グリッド画像を保存しました: {output_path}")


if __name__ == "__main__":
    main()


