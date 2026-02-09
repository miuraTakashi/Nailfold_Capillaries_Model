"""
vessel_model.py

血管壁の動きのモデル本体（関数定義）をまとめたモジュール。

- TwoVesselModel: シミュレーション本体
- plotVessel:     1フレームの可視化
- export_vessel_animation: アニメーションの保存
"""

import datetime
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm


def TwoVesselModel(L=5, dh=1, sigma=4, ks=0.1, ke=0.1,
                   w1=1, w2=2, r=0.5, fps=20, T=1000, progress=True):
    """
    2つの血管の界面位置の時間発展を計算する関数

    パラメータ:
    -----------
    L : float
        空間領域の長さ（デフォルト: 5）
    dh : float
        表面張力係数 d_h（デフォルト: 1）
    sigma : float
        ノイズ強度 σ（デフォルト: 1）
    ks : float
        結合組織の硬さ k_s（デフォルト: 0.1）
    ke : float
        血管間相互作用 k_e（デフォルト: 0.1）
    w1 : float
        血管と表皮の間隔（デフォルト: 1）
    w2 : float
        血管間の間隔（デフォルト: 2）
    r : float
        血管の半径（デフォルト: 0.5）
    fps : int
        アニメーションのフレームレート（デフォルト: 20）
    T : float
        シミュレーション時間（デフォルト: 1000）
    progress : bool
        進捗バーの表示（デフォルト: True）

    戻り値:
    -------
    hList : list
        各時刻での界面位置 [h1, h2, h3, h4] のリスト
    """
    H = w2 / 2 + 2 * r + w1
    dx = 0.1
    dt = 0.001
    sqrt_dt = np.sqrt(dt)

    n = int(L / dx)
    loop = int(T / dt)
    loopPerOutput = int(T / dt / 100)

    def Delta(h):
        """2階微分を計算（周期境界条件）"""
        return (np.roll(h, 1) + np.roll(h, -1) - 2 * h) / dx / dx

    hList = []
    h1 = np.zeros(n) + w2 / 2 + 2 * r
    h1New = deepcopy(h1)
    h2 = np.zeros(n) + w2 / 2
    h2New = deepcopy(h2)
    h3 = np.zeros(n) - w2 / 2
    h3New = deepcopy(h3)
    h4 = np.zeros(n) - w2 / 2 - 2 * r
    h4New = deepcopy(h4)

    iterator = tqdm(range(loop)) if progress else range(loop)

    for t in iterator:
        h1New += (
            dt
            * (
                -ks / w1 * (h1 - (H - w1))
                + ke / (2 * r) * (h2 - h1 + 2 * r)
                + dh * Delta(h1)
            )
            + sigma * np.random.randn(n) * sqrt_dt
        )

        h2New += (
            dt
            * (
                -ke / (2 * r) * (h2 - h1 + 2 * r)
                + ks / w2 * (h3 - h2 + w2)
                + dh * Delta(h2)
            )
            + sigma * np.random.randn(n) * sqrt_dt
        )

        h3New += (
            dt
            * (
                -ks / w2 * (h3 - h2 + w2)
                + ke / (2 * r) * (h4 - h3 + 2 * r)
                + dh * Delta(h3)
            )
            + sigma * np.random.randn(n) * sqrt_dt
        )

        h4New += (
            dt
            * (
                -ke / (2 * r) * (h4 - h3 + 2 * r)
                - ks / w1 * (h4 + H - w1)
                + dh * Delta(h4)
            )
            + sigma * np.random.randn(n) * sqrt_dt
        )

        h1 = deepcopy(h1New)
        h1[0] = h1[-1] = w2 / 2 + 2 * r
        h1[h1 > H] = H
        h2 = deepcopy(h2New)
        h2[0] = h2[-1] = w2 / 2
        h3 = deepcopy(h3New)
        h3[0] = h3[-1] = -w2 / 2
        h4 = deepcopy(h4New)
        h4[h4 < -H] = -H
        h4[0] = h4[-1] = -w2 / 2 - 2 * r

        if t % loopPerOutput == 0:
            hList.append([h1, h2, h3, h4])

    return hList


def plotVessel(h, ax=None, save_path=None):
    """
    血管の界面位置をプロットする関数

    パラメータ:
    -----------
    h : list
        [h1, h2, h3, h4] のリスト
    ax : matplotlib.axes.Axes, optional
        プロットする軸。指定しない場合は新しい図を作成
    save_path : str, optional
        保存先のパス。指定しない場合はresultsフォルダに自動保存
    """
    if ax is None:
        fig, ax = plt.subplots()
        create_new_figure = True
    else:
        create_new_figure = False

    ax.plot(h[0], color="red")
    ax.plot(h[1], color="red")
    ax.plot(h[2], color="red")
    ax.plot(h[3], color="red")
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.fill_between(range(len(h[0])), h[0], h[1], color="red", alpha=0.2)
    ax.fill_between(range(len(h[2])), h[2], h[3], color="red", alpha=0.2)

    if create_new_figure:
        # resultsフォルダを作成
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # 保存パスを決定
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = results_dir / f"vessel_{timestamp}.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"画像を保存しました: {save_path}")


def export_vessel_animation(
    dh=0.1, sigma=4, ks=0.001, ke=0.1, w1=1, w2=2, r=0.5, T=100, fps=20
):
    """
    引数で与えた dh, sigma, ks, ke を用いてシミュレーションし、
    形態変化のアニメーションを保存する。

    ffmpeg が利用可能なら MP4、そうでなければ GIF で保存する。

    戻り値:
    -------
    filename : str
        保存されたファイル名（通常は .mp4、ffmpeg が無い場合は .gif）
    """
    # インタラクティブ表示を抑止
    prev_interactive = plt.isinteractive()
    plt.ioff()
    try:
        # 進捗バーも抑止
        result = TwoVesselModel(
            dh=dh,
            sigma=sigma,
            ks=ks,
            ke=ke,
            w1=w1,
            w2=w2,
            r=r,
            T=T,
            fps=20,
            progress=False,
        )

        # resultsフォルダを作成
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # 利用可能な writer を判定
        use_ffmpeg = animation.writers.is_available("ffmpeg")
        if use_ffmpeg:
            ext = "mp4"
        else:
            # ffmpeg が無い環境では PillowWriter を使って GIF で保存
            ext = "gif"

        filename = results_dir / f"vessel_dh{dh}_sigma{sigma}_ks{ks}_ke{ke}.{ext}"
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            plotVessel(result[i], ax=ax)
            ax.set_title(f"Frame {i}")

        ani = animation.FuncAnimation(
            fig, animate, frames=len(result), interval=1000.0 / fps, repeat=False
        )

        try:
            if use_ffmpeg:
                from matplotlib.animation import FFMpegWriter

                writer = FFMpegWriter(
                    fps=fps, metadata=dict(artist=""), bitrate=1800
                )
            else:
                from matplotlib.animation import PillowWriter

                writer = PillowWriter(fps=fps)

            ani.save(filename, writer=writer)
        finally:
            plt.close(fig)
    finally:
        plt.interactive(prev_interactive)

    return filename


