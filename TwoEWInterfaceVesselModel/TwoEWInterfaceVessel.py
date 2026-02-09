#%%
"""
TwoEWInterfaceVesselModel - 血管壁の動きのシミュレーションモデル

血管壁の動きをシミュレーションする数値計算モデルです。
2つの血管の界面の時間発展を、拡散、ランダムノイズ、弾性相互作用を考慮して計算します。
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from matplotlib import animation
import os
from pathlib import Path
import datetime


# 血管壁の動きの支配方程式：
# 
# ∂_t h_1 = d_h ∂_x^2 h_1 + σ η_1(x,t) - k_s/w_1(h_1-(H-w_1)) + k_e/(2r)(h_2-h_1+2r)
# ∂_t h_2 = d_h ∂_x^2 h_2 + σ η_2(x,t) - k_e/(2r)(h_2-h_1+2r) + k_s/w_2(h_3-h_2+w_2)
# ∂_t h_3 = d_h ∂_x^2 h_3 + σ η_3(x,t) - k_s/w_2(h_3-h_2+w_2) + k_e/(2r)(h_4-h_3+2r)
# ∂_t h_4 = d_h ∂_x^2 h_4 + σ η_4(x,t) - k_e/(2r)(h_4-h_3+2r) - k_s/w_1(h_4+H-w_1)
#
# パラメータの説明:
# - h_i(x,t): 時刻tで場所xのi番目の血管の界面の位置
# - d_h: 血管壁の表面張力。炎症時には細胞間接着が外れて減少する
# - σ: 内皮細胞のランダムな運動の強さ
# - η_i(x,t): 平均0、分散1の正規分布の乱数（標準正規乱数）
# - k_s: 周辺の結合組織の硬さ（単位長さあたりのバネ係数）
# - k_e: 血管壁同士の相互作用の単位長さあたりのバネ係数
# - H: 血管系全体の高さの範囲
# - w_1: 血管と表皮の間の間隔
# - w_2: 血管同士の間の間隔
# - r: 血管の半径

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vessel_model import TwoVesselModel, plotVessel, export_vessel_animation


if __name__ == "__main__":
    # 使用例1: 基本的なシミュレーション
    print("=== 基本的なシミュレーション ===")
    result = TwoVesselModel(L=5, dh=0.1, sigma=1, ks=0.1, ke=0.1, T=100)
    plotVessel(result[-1], save_path="results/vessel_basic.png")
    
    # 使用例2: 異なるパラメータでの比較
    print("\n=== 異なるパラメータでの比較 ===")
    for ks in [10, 0]:
        result = TwoVesselModel(ks=ks, ke=1, sigma=1, T=100)
        print(f"ks={ks}")
        plotVessel(result[-1], save_path=f"results/vessel_ks{ks}.png")
    
    # 使用例3: アニメーションの生成
    print("\n=== アニメーションの生成 ===")
    filename = export_vessel_animation(dh=0.1, sigma=1, ks=0.1, ke=0.1)
    print(f"アニメーションを保存しました: {filename}")
    
    # 使用例4: 統計解析（コメントアウト - 時間がかかるため）
    # print("\n=== 統計解析 ===")
    # resultList = []
    # for i in range(100):
    #     result = TwoVesselModel(T=100)
    #     resultList.append(result)
    # 
    # h1hatList = []
    # h2hatList = []
    # h3hatList = []
    # h4hatList = []
    # for i in range(100):
    #     h1hatList.append(np.abs(np.fft.fft(resultList[i][-1][0])))
    #     h2hatList.append(np.abs(np.fft.fft(resultList[i][-1][1])))
    #     h3hatList.append(np.abs(np.fft.fft(resultList[i][-1][2])))
    #     h4hatList.append(np.abs(np.fft.fft(resultList[i][-1][3])))
    # 
    # h1hatMean = np.array(h1hatList).mean(axis=0)
    # h2hatMean = np.array(h2hatList).mean(axis=0)
    # h3hatMean = np.array(h3hatList).mean(axis=0)
    # h4hatMean = np.array(h4hatList).mean(axis=0)
    # 
    # fig, ax = plt.subplots()
    # ax.plot(h1hatMean)
    # ax.plot(h2hatMean)
    # ax.plot(h3hatMean)
    # ax.plot(h4hatMean)
    # ax.set_xlim(1, 25)
    # ax.set_ylim(0.1, 10)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.legend(["h1", "h2", "h3", "h4"])
    # 
    # results_dir = Path("results")
    # results_dir.mkdir(exist_ok=True)
    # plt.savefig(results_dir / "fourier_analysis.png", dpi=150, bbox_inches='tight')
    # plt.close(fig)
    # print(f"統計解析結果を保存しました: {results_dir / 'fourier_analysis.png'}")


# %%
