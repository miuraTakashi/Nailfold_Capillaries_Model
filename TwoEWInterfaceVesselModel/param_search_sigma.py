"""
param_search_sigma.py

`vessel_model.py` を用いて、ks=0 に固定し、
sigma を [1, 10, 100] で振った結果を横一列に並べて表示するスクリプト。
"""

from pathlib import Path

import matplotlib.pyplot as plt

from vessel_model import TwoVesselModel, plotVessel


def main():
    # パラメータ設定
    ks = 0
    sigma_list = list(range(11))

    # 結果保存ディレクトリ
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 1 行 N 列のグリッド
    n = len(sigma_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharex=True, sharey=True)

    # axes が 1 つだけのときに備えてリスト化
    if n == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigma_list):
        # シミュレーション実行（T は必要に応じて調整）
        # progress=True として tqdm による進捗を表示
        result = TwoVesselModel(ks=ks, sigma=sigma, T=100, progress=True)

        # 最終フレームを描画
        plotVessel(result[-1], ax=ax, save_path=None)
        ax.set_title(f"ks={ks}, sigma={sigma}")

    # タイトル & レイアウト
    fig.suptitle("TwoVesselModel - sigma parameter sweep (ks=0)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存 & 表示
    output_path = results_dir / "param_search_sigma_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"グリッド画像を保存しました: {output_path}")


if __name__ == "__main__":
    main()


