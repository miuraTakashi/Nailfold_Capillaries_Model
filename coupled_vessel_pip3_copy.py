"""
PIP3–PTEN 駆動血管壁変形モデル（結合モデル）

TwoEWInterfaceVessel モデルのノイズ項 η_i(x,t) を
PIP3–PTEN 興奮系の u_{i,x}(t) に置き換えた結合モデル。

元のモデル:
  ∂_t h_i = d_h ∂²_x h_i + σ η_i(x,t) + [バネ力]
  η_i(x,t): ホワイトノイズ（各 i, x で独立）

結合モデル:
  ∂_t h_i = d_h ∂²_x h_i + σ_c s_i η_i(x,t) + [バネ力]

  du_{i,x} = f(u,v) dt + σ_u dW_u     (PIP3 ダイナミクス)
  dv_{i,x} = g(u,v) dt + σ_v dW_v     (PTEN ダイナミクス)

各界面 i=1..4, 各空間点 x で独立な PIP3–PTEN 興奮系が駆動力を生成。
η_i(x,t) は PIP3 の閾値超過分（正の部分）として
  η_i(x,t) = max(u_i(x,t), u_threshold) - u_threshold
           = max(u_i(x,t) - u_threshold, 0)
と定義し、「一定の閾値を超えたときのみ、その差分の力で押す」形にする。
ここで s_i は「血管の外向き」に変位するよう符号を揃える係数で、
(h1, h2, h3, h4) = (+, −, +, −) とする（η2, η4 を負にするのに対応）。

比較:
  疾患状態: vPI3K = 1.274 → 興奮あり → 間欠的な大きな変形
  正常状態: vPI3K = 0.75  → 興奮なし → 小さな揺らぎのみ
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve
from pathlib import Path


# ============================================================
# PIP3–PTEN 興奮系モデル関数（numpy 配列対応）
# ============================================================

def A_pip3(u, par):
    """PIP3 正帰還（Hill 関数）"""
    alpha, KA, n = par["alpha"], par["KA"], par["hill_n"]
    return 1.0 + alpha * (u**n) / (KA**n + u**n)


def k_on0_func(u, par):
    """PIP3 依存の基礎結合速度"""
    k_on0_base, gamma, Ku = par["k_on0_base"], par["gamma"], par["Ku_k_on0"]
    return k_on0_base * (1.0 + gamma * u / (Ku + u))


def f_pip3(u, v, par):
    """du/dt: PIP3 の反応項（ベクトル化対応）"""
    Ptot = par["Ptot"]
    p2 = Ptot - u
    vPI3K, KPI3K = par["vPI3K"], par["KPI3K"]
    vPTEN, KPTEN = par["vPTEN"], par["KPTEN"]
    k_leak = par["k_leak"]
    tau_u = par["tau_u"]
    return (1.0 / tau_u) * (
        vPI3K * A_pip3(u, par) * p2 / (KPI3K + p2)
        - vPTEN * v * u / (KPTEN + u)
        - k_leak * u
    )


def g_pten(u, v, par):
    """dv/dt: PTEN の反応項（ベクトル化対応）"""
    Ptot, Vtot = par["Ptot"], par["Vtot"]
    p2 = Ptot - u
    kon, koff = par["kon"], par["koff"]
    k_on0_u = k_on0_func(u, par)
    k_on_total = k_on0_u + kon * p2
    return k_on_total * (Vtot - v) - koff * v


def find_rest_state(par):
    """PIP3–PTEN 系の安定固定点（静止状態）を数値的に求める"""
    Ptot, Vtot = par["Ptot"], par["Vtot"]
    N = 2000
    u_arr = np.linspace(0.02, max(0.02, Ptot - 1e-6), N)
    p2 = Ptot - u_arr

    vPI3K, KPI3K = par["vPI3K"], par["KPI3K"]
    vPTEN, KPTEN = par["vPTEN"], par["KPTEN"]
    k_leak = par["k_leak"]
    prod = vPI3K * A_pip3(u_arr, par) * p2 / (KPI3K + p2)
    denom = vPTEN * (u_arr / (KPTEN + u_arr))
    numer = prod - k_leak * u_arr
    v_u = np.where(numer > 0, numer / denom, np.nan)

    kon, koff = par["kon"], par["koff"]
    k_on0_u = k_on0_func(u_arr, par)
    k_on_total = k_on0_u + kon * p2
    v_v = Vtot * k_on_total / (koff + k_on_total)

    diff = v_u - v_v
    valid = np.isfinite(diff)
    s0 = np.sign(diff[:-1])
    s1 = np.sign(diff[1:])
    idx = np.where(valid[:-1] & valid[1:] & (s0 != s1))[0]

    fps = []
    for i in idx:
        u0 = u_arr[i] - diff[i] * (u_arr[i + 1] - u_arr[i]) / (diff[i + 1] - diff[i])
        v0 = np.interp(u0, u_arr, v_v)

        def residual(y):
            return [f_pip3(y[0], y[1], par), g_pten(y[0], y[1], par)]

        sol, _, ier, _ = fsolve(residual, [u0, v0], full_output=True)
        if ier == 1:
            fps.append(sol)

    fps.sort(key=lambda p: p[0])
    return fps[0] if fps else np.array([0.03, 0.5])


# ============================================================
# 結合モデルシミュレーション
# ============================================================

def CoupledVesselPIP3Model(
    # 血管パラメータ
    L=5, dh=0.001, ks=0.1, ke=0.1,
    w1=1, w2=2, r=0.5,
    sigma_c=50.0,           # 結合強度: σ_c × η_i(x,t)
    u_threshold=0.4,        # 駆動の閾値（None のとき u_rest を使用）
    T=50,
    # PIP3-PTEN パラメータ
    pip3_par=None,
    sigma_u=3.0,
    sigma_v=0.0,
    n_substep=2,            # 血管1ステップあたりの PIP3 サブステップ数（dt_p=0.0005）
    seed=42,
    progress=True,
):
    """
    PIP3–PTEN 興奮系で駆動される血管壁変形モデル

    Parameters
    ----------
    L, dh, ks, ke, w1, w2, r : 血管モデルパラメータ
    sigma_c    : 結合強度（PIP3 → 血管壁の駆動力係数）
    T          : シミュレーション時間
    pip3_par   : PIP3–PTEN パラメータ辞書
    sigma_u    : PIP3 へのノイズ強度
    sigma_v    : PTEN へのノイズ強度
    n_substep  : 各血管ステップあたりの PIP3-PTEN サブステップ数
    seed       : 乱数シード
    progress   : 進捗バーの表示
    u_threshold: 駆動力の閾値（u がこれを超えた分だけ壁を押す）

    Returns
    -------
    dict with keys:
        'h_frames' : 各出力フレームでの [h1, h2, h3, h4]
        'u_trace'  : u(t) 時系列（界面1, 中央点）
        'h_trace'  : h1(t) 時系列（界面1, 中央点）
        't_trace'  : 時間軸
        'u_rest'   : u*（静止状態 PIP3）
        'v_rest'   : v*（静止状態 PTEN）
        'par'      : 使用した PIP3-PTEN パラメータ
    """
    if pip3_par is None:
        pip3_par = dict(
            Ptot=1.000, vPI3K=1.274, KPI3K=0.010, vPTEN=6.810,
            KPTEN=0.100, k_leak=5.667, alpha=5.000, KA=0.300,
            hill_n=4.021, kon=2.000, koff=18.000, k_on0_base=7.143,
            Vtot=0.984, gamma=13.571, Ku_k_on0=0.347, tau_u=0.010,
        )

    rng = np.random.default_rng(seed)

    # --- Grid ---
    H = w2 / 2 + 2 * r + w1
    dx = 0.1
    dt_v = 0.001                    # 血管の時間刻み
    n_x = int(L / dx)               # 空間格子点数
    n_steps = int(T / dt_v)
    n_out = 100                     # 出力フレーム数
    out_every = max(1, n_steps // n_out)

    dt_p = dt_v / n_substep         # PIP3 サブステップの時間刻み
    sqrt_dt_p = np.sqrt(dt_p)
    Ptot = pip3_par["Ptot"]
    Vtot_pip3 = pip3_par["Vtot"]

    # --- 安定固定点 ---
    rest = find_rest_state(pip3_par)
    u_rest, v_rest = rest[0], rest[1]
    print(f"  PIP3-PTEN rest state: u* = {u_rest:.6f}, v* = {v_rest:.6f}")
    if u_threshold is None:
        u_threshold = float(u_rest)

    # --- Laplacian（周期境界条件） ---
    def Lap(h_row):
        return (np.roll(h_row, 1) + np.roll(h_row, -1) - 2 * h_row) / dx**2

    # --- 血管壁の初期化 ---
    h = np.zeros((4, n_x))
    h[0] = w2 / 2 + 2 * r          # h1
    h[1] = w2 / 2                   # h2
    h[2] = -w2 / 2                  # h3
    h[3] = -w2 / 2 - 2 * r         # h4
    h_eq = h[:, 0].copy()           # 境界条件の平衡値

    # --- PIP3-PTEN の初期化: 4界面 × n_x 点 ---
    u_all = np.full((4, n_x), u_rest)
    v_all = np.full((4, n_x), v_rest)

    # --- 駆動項の符号（血管の外向きにノイズが入るように調整） ---
    # h1,h3: + 方向、h2,h4: - 方向（η2, η4 の符号を負にするのに対応）
    outward_sign = np.array([+1.0, -1.0, +1.0, -1.0])[:, None]  # (4,1) -> (4,n_x) にブロードキャスト

    # --- 記録用 ---
    h_frames = []

    trace_every = max(1, n_steps // 5000)
    n_trace_max = n_steps // trace_every + 1
    x_mid = n_x // 2

    u_trace = np.empty(n_trace_max)
    h_trace = np.empty(n_trace_max)
    t_trace = np.empty(n_trace_max)
    ti = 0

    desc = f"vPI3K={pip3_par['vPI3K']}"
    print_every = max(1, n_steps // 20)  # 5% ごとに進捗表示

    for step in range(n_steps):
        if progress and step % print_every == 0:
            pct = 100 * step / n_steps
            print(f"\r  [{desc}] {pct:5.1f}%", end="", flush=True)
        # ---- PIP3-PTEN サブステップ（全 4×n_x 点をベクトル化） ----
        for _ in range(n_substep):
            fu = f_pip3(u_all, v_all, pip3_par)
            fv = g_pten(u_all, v_all, pip3_par)
            u_all += fu * dt_p + sigma_u * sqrt_dt_p * rng.standard_normal((4, n_x))
            v_all += fv * dt_p + sigma_v * sqrt_dt_p * rng.standard_normal((4, n_x))
            np.clip(u_all, 0.0, Ptot, out=u_all)
            np.clip(v_all, 0.0, Vtot_pip3, out=v_all)

        # ---- 血管壁の更新（旧 h で全力を計算してから更新） ----
        # 閾値を超えた分だけ壁を押す:
        #   η_i(x,t) = max(u_i(x,t) - u_threshold, 0)
        # 外向き法線方向に合わせて符号を付ける（η2,η4 は負）。
        eta = np.maximum(u_all - u_threshold, 0.0)
        driving = sigma_c * outward_sign * eta

        # Laplacian を先に計算
        lap = np.array([Lap(h[i]) for i in range(4)])

        inc0 = dt_v * (
            dh * lap[0]
            - ks / w1 * (h[0] - (H - w1))
            + ke / (2 * r) * (h[1] - h[0] + 2 * r)
            + driving[0]
        )
        inc1 = dt_v * (
            dh * lap[1]
            - ke / (2 * r) * (h[1] - h[0] + 2 * r)
            + ks / w2 * (h[2] - h[1] + w2)
            + driving[1]
        )
        inc2 = dt_v * (
            dh * lap[2]
            - ks / w2 * (h[2] - h[1] + w2)
            + ke / (2 * r) * (h[3] - h[2] + 2 * r)
            + driving[2]
        )
        inc3 = dt_v * (
            dh * lap[3]
            - ke / (2 * r) * (h[3] - h[2] + 2 * r)
            - ks / w1 * (h[3] + H - w1)
            + driving[3]
        )

        h[0] += inc0
        h[1] += inc1
        h[2] += inc2
        h[3] += inc3

        # --- 境界条件 ---
        for i in range(4):
            h[i, 0] = h[i, -1] = h_eq[i]
        np.clip(h[0], None, H, out=h[0])
        np.clip(h[3], -H, None, out=h[3])

        # --- フレーム記録 ---
        if step % out_every == 0:
            h_frames.append([h[i].copy() for i in range(4)])

        # --- トレース記録 ---
        if step % trace_every == 0 and ti < n_trace_max:
            u_trace[ti] = u_all[0, x_mid]
            h_trace[ti] = h[0, x_mid]
            t_trace[ti] = step * dt_v
            ti += 1

    if progress:
        print(f"\r  [{desc}] 100.0% — done ({ti} trace points, {len(h_frames)} frames)")

    return {
        "h_frames": h_frames,
        "u_trace": u_trace[:ti],
        "h_trace": h_trace[:ti],
        "t_trace": t_trace[:ti],
        "u_rest": u_rest,
        "v_rest": v_rest,
        "par": pip3_par,
    }


# ============================================================
# 描画ヘルパー
# ============================================================

def plot_vessel(h, ax, title="", fontsize=12):
    """血管断面の描画（fill 付き）"""
    x = np.arange(len(h[0]))
    for i in range(4):
        ax.plot(x, h[i], color="red", linewidth=1)
    ax.fill_between(x, h[0], h[1], color="red", alpha=0.2)
    ax.fill_between(x, h[2], h[3], color="red", alpha=0.2)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.grid(True, alpha=0.3)

def save_vessel_gif(res_left, res_right, save_path, *, fps=12, ylim=(-5, 5)):
    """
    res_left/res_right の h_frames を左右に並べて GIF 保存する。
    """
    h_frames_left = res_left["h_frames"]
    h_frames_right = res_right["h_frames"]
    n_frames = min(len(h_frames_left), len(h_frames_right))
    if n_frames == 0:
        raise ValueError("h_frames is empty; cannot create GIF.")

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(len(h_frames_left[0][0]))

    def init_ax(ax, title):
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=10)

    init_ax(ax1, "Disease (vPI3K=1.274)")
    init_ax(ax2, "Normal (vPI3K=0.75)")

    # line objects (4 per axis)
    lines1 = [ax1.plot([], [], color="red", linewidth=1)[0] for _ in range(4)]
    lines2 = [ax2.plot([], [], color="red", linewidth=1)[0] for _ in range(4)]

    # fill objects (2 per axis: upper wall, lower wall)
    fills = {
        "ax1_upper": None,
        "ax1_lower": None,
        "ax2_upper": None,
        "ax2_lower": None,
    }

    def draw_frame(ax, h_frame, lines, fill_upper_key, fill_lower_key):
        # update lines
        for i in range(4):
            lines[i].set_data(x, h_frame[i])

        # refresh fills (simple & robust; 100 frames程度なら十分高速)
        if fills[fill_upper_key] is not None:
            fills[fill_upper_key].remove()
        if fills[fill_lower_key] is not None:
            fills[fill_lower_key].remove()
        fills[fill_upper_key] = ax.fill_between(x, h_frame[0], h_frame[1], color="red", alpha=0.2)
        fills[fill_lower_key] = ax.fill_between(x, h_frame[2], h_frame[3], color="red", alpha=0.2)

    # time label
    time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=10)

    def init():
        draw_frame(ax1, h_frames_left[0], lines1, "ax1_upper", "ax1_lower")
        draw_frame(ax2, h_frames_right[0], lines2, "ax2_upper", "ax2_lower")
        time_text.set_text("t = 0.0")
        artists = lines1 + lines2 + [time_text]
        return artists

    def update(k):
        draw_frame(ax1, h_frames_left[k], lines1, "ax1_upper", "ax1_lower")
        draw_frame(ax2, h_frames_right[k], lines2, "ax2_upper", "ax2_lower")
        time_text.set_text(f"frame {k+1}/{n_frames}")
        artists = lines1 + lines2 + [time_text]
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=False,
        interval=1000 / fps,
        repeat=False,
    )

    try:
        writer = animation.PillowWriter(fps=fps)
    except Exception as e:
        raise RuntimeError(
            "GIF 保存には Pillow が必要です。`pip install pillow` を入れてください。"
        ) from e

    # 文字列パスで渡す（Path のままだと保存に失敗することがある）
    path_str = str(save_path)
    ani.save(path_str, writer=writer)
    plt.close(fig)


# ============================================================
# メイン: 疾患 vs 正常の比較
# ============================================================

if __name__ == "__main__":

    # --- 共通パラメータ ---
    par_base = dict(
        Ptot=1.000, vPI3K=1.274, KPI3K=0.010, vPTEN=6.810,
        KPTEN=0.100, k_leak=5.667, alpha=5.000, KA=0.300,
        hill_n=4.021, kon=2.000, koff=18.000, k_on0_base=7.143,
        Vtot=0.984, gamma=13.571, Ku_k_on0=0.347, tau_u=0.010,
    )

    T_sim = 50
    sigma_c = 50.0      # 結合強度（5倍）
    seed = 42

    # --- 疾患状態: vPI3K = 1.274（興奮あり） ---
    par_disease = dict(par_base)

    # --- 正常状態: vPI3K = 0.75（興奮なし） ---
    par_normal = dict(par_base)
    par_normal["vPI3K"] = 0.75

    print("=" * 60)
    print("疾患状態 (vPI3K = 1.274, 興奮あり)")
    print("=" * 60)
    res_disease = CoupledVesselPIP3Model(
        pip3_par=par_disease, dh=0.001, sigma_c=sigma_c, u_threshold=0.4, T=T_sim, seed=seed,
    )

    print()
    print("=" * 60)
    print("正常状態 (vPI3K = 0.75, 興奮なし)")
    print("=" * 60)
    res_normal = CoupledVesselPIP3Model(
        pip3_par=par_normal, dh=0.001, sigma_c=sigma_c, u_threshold=0.4, T=T_sim, seed=seed,
    )

    # ============================================================
    # 比較プロット: 3 行 × 2 列（Word貼り付け用の文字サイズ）
    # ============================================================
    fs_label = 20   # 軸ラベル・サブタイトル
    fs_tick = 18    # 目盛り
    fs_legend = 16  # 凡例
    fs_panel = 20   # パネルラベル (a)(b)(c)
    fs_suptitle = 22  # 全体タイトル

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 0.8, 0.8],
                  hspace=0.40, wspace=0.25)

    # ---- Row 1: 血管断面（最終時刻） ----
    ax_v1 = fig.add_subplot(gs[0, 0])
    ax_v2 = fig.add_subplot(gs[0, 1], sharey=ax_v1)
    plot_vessel(res_disease["h_frames"][-1], ax_v1,
                title=f"Disease (vPI3K=1.274): vessel at T={T_sim}", fontsize=fs_label)
    plot_vessel(res_normal["h_frames"][-1], ax_v2,
                title=f"Normal (vPI3K=0.75): vessel at T={T_sim}", fontsize=fs_label)
    ax_v1.text(-0.12, 1.02, "(a)", transform=ax_v1.transAxes,
               fontsize=fs_panel, fontweight="bold", va="bottom")

    # ---- Row 2: PIP3 u(t) 時系列（界面1, 中央点） ----
    ax_u1 = fig.add_subplot(gs[1, 0])
    ax_u2 = fig.add_subplot(gs[1, 1], sharey=ax_u1)

    ax_u1.plot(res_disease["t_trace"], res_disease["u_trace"],
               color="tab:blue", linewidth=0.3, alpha=0.8)
    ax_u1.axhline(res_disease["u_rest"], color="gray", ls=":", alpha=0.5,
                  label=f"u* = {res_disease['u_rest']:.4f}")
    ax_u1.set_xlabel("Time", fontsize=fs_label)
    ax_u1.set_ylabel("u (PIP3)", fontsize=fs_label)
    ax_u1.set_title("Disease: PIP3 at h₁, x=mid", fontsize=fs_label)
    ax_u1.tick_params(axis="both", labelsize=fs_tick)
    ax_u1.legend(fontsize=fs_legend)
    ax_u1.grid(True, alpha=0.3)

    ax_u2.plot(res_normal["t_trace"], res_normal["u_trace"],
               color="tab:blue", linewidth=0.3, alpha=0.8)
    ax_u2.axhline(res_normal["u_rest"], color="gray", ls=":", alpha=0.5,
                  label=f"u* = {res_normal['u_rest']:.4f}")
    ax_u2.set_xlabel("Time", fontsize=fs_label)
    ax_u2.set_ylabel("")
    ax_u2.set_title("Normal: PIP3 at h₁, x=mid", fontsize=fs_label)
    ax_u2.tick_params(axis="both", labelsize=fs_tick)
    ax_u2.legend(fontsize=fs_legend)
    ax_u2.grid(True, alpha=0.3)
    ax_u1.text(-0.12, 1.02, "(b)", transform=ax_u1.transAxes,
               fontsize=fs_panel, fontweight="bold", va="bottom")

    # ---- Row 3: h1(t) 時系列（界面1, 中央点） ----
    ax_h1 = fig.add_subplot(gs[2, 0])
    ax_h2 = fig.add_subplot(gs[2, 1], sharey=ax_h1)

    ax_h1.plot(res_disease["t_trace"], res_disease["h_trace"],
               color="tab:red", linewidth=0.5, alpha=0.8)
    ax_h1.set_xlabel("Time", fontsize=fs_label)
    ax_h1.set_ylabel("h₁(x_mid, t)", fontsize=fs_label)
    ax_h1.set_title("Disease: h₁ position", fontsize=fs_label)
    ax_h1.tick_params(axis="both", labelsize=fs_tick)
    ax_h1.grid(True, alpha=0.3)

    ax_h2.plot(res_normal["t_trace"], res_normal["h_trace"],
               color="tab:red", linewidth=0.5, alpha=0.8)
    ax_h2.set_xlabel("Time", fontsize=fs_label)
    ax_h2.set_ylabel("")
    ax_h2.set_title("Normal: h₁ position", fontsize=fs_label)
    ax_h2.tick_params(axis="both", labelsize=fs_tick)
    ax_h2.grid(True, alpha=0.3)
    ax_h1.text(-0.12, 1.02, "(c)", transform=ax_h1.transAxes,
               fontsize=fs_panel, fontweight="bold", va="bottom")

    fig.suptitle(
        "PIP3–PTEN driven vessel deformation model\n"
        f"Disease (vPI3K=1.274, excitable) vs Normal (vPI3K=0.75, non-excitable)  "
        f"[σ_c={sigma_c}, T={T_sim}]",
        fontsize=fs_suptitle, y=0.99,
    )
    fig.subplots_adjust(top=0.90, hspace=0.45, wspace=0.25)

    # --- 保存 ---
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "coupled_vessel_pip3_comparison2.jpg"
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"\n図を保存しました: {save_path}")

    # --- GIF ムービー保存（血管壁の時間発展） ---
    gif_path = results_dir / "coupled_vessel_pip3_vessel_movie2.gif"
    try:
        save_vessel_gif(res_disease, res_normal, gif_path, fps=12, ylim=(-5, 5))
        if gif_path.exists():
            print(f"GIF ムービーを保存しました: {gif_path.absolute()}")
        else:
            print(f"警告: GIF の保存先を確認してください: {gif_path.absolute()}")
    except Exception as e:
        print(f"GIF 保存エラー: {e}")
        import traceback
        traceback.print_exc()
    plt.show()
