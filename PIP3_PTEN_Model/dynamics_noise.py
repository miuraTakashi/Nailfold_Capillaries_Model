"""
PIP3–PTEN 興奮系 — ノイズ駆動の確率的ダイナミクス

安定固定点（静止状態）近傍にホワイトノイズ η を加え、
ノイズが閾値を時々超えることで自発的な興奮（発火）が起きる様子を
Euler–Maruyama 法でシミュレーションする。

  du = f(u,v) dt  +  sigma_u dW_u
  dv = g(u,v) dt  +  sigma_v dW_v

出力:
  上段: u(t), v(t) の時系列（発火イベントが見える）
  下段: (u, v) 相平面上の軌道（ヌルクライン付き）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve

# ============================================================
# パラメータ（nullcline.py / dynamics.py と共通）
# ============================================================
par = dict(
    Ptot=1.000,
    vPI3K=1.274,
    KPI3K=0.010,
    vPTEN=6.810,
    KPTEN=0.100,
    k_leak=5.667,
    alpha=5.000,
    KA=0.300,
    hill_n=4.021,
    kon=2.000,
    koff=18.000,
    k_on0_base=7.143,
    Vtot=0.984,
    gamma=13.571,
    Ku_k_on0=0.347,
    tau_u=0.010,
)

# ============================================================
# ノイズ強度（ここを調整して発火頻度を変える）
# ============================================================
sigma_u = 3.0    # u（PIP3）へのノイズ強度
sigma_v = 0.0    # v（PTEN）へのノイズ強度（0 なら u のみにノイズ）

# ============================================================
# モデル関数（dynamics.py と同一）
# ============================================================

def A(u, par):
    alpha, KA, n = par["alpha"], par["KA"], par["hill_n"]
    return 1.0 + alpha * (u**n) / (KA**n + u**n)


def k_on0_func(u, par):
    k_on0_base, gamma, Ku_k_on0 = par["k_on0_base"], par["gamma"], par["Ku_k_on0"]
    return k_on0_base * (1.0 + gamma * u / (Ku_k_on0 + u))


def du_dt(u, v, par):
    Ptot = par["Ptot"]
    p2 = Ptot - u
    vPI3K, KPI3K = par["vPI3K"], par["KPI3K"]
    vPTEN, KPTEN = par["vPTEN"], par["KPTEN"]
    k_leak = par["k_leak"]
    tau_u = par["tau_u"]
    return (1.0 / tau_u) * (
        vPI3K * A(u, par) * p2 / (KPI3K + p2)
        - vPTEN * v * u / (KPTEN + u)
        - k_leak * u
    )


def dv_dt(u, v, par):
    Ptot, Vtot = par["Ptot"], par["Vtot"]
    p2 = Ptot - u
    kon, koff = par["kon"], par["koff"]
    k_on0_u = k_on0_func(u, par)
    k_on_total = k_on0_u + kon * p2
    return k_on_total * (Vtot - v) - koff * v


# ============================================================
# ヌルクライン計算
# ============================================================

def compute_nullclines(par, N=500):
    Ptot, Vtot = par["Ptot"], par["Vtot"]
    u_max = max(0.02, Ptot - 1e-6)
    u = np.linspace(0.02, u_max, N)
    p2 = Ptot - u

    vPI3K, KPI3K = par["vPI3K"], par["KPI3K"]
    vPTEN, KPTEN = par["vPTEN"], par["KPTEN"]
    k_leak = par["k_leak"]
    prod = vPI3K * A(u, par) * p2 / (KPI3K + p2)
    denom = vPTEN * (u / (KPTEN + u))
    numer = prod - k_leak * u
    v_u = np.where(numer > 0, numer / denom, np.nan)

    kon, koff = par["kon"], par["koff"]
    k_on0_u = k_on0_func(u, par)
    k_on_total = k_on0_u + kon * p2
    v_v = Vtot * k_on_total / (koff + k_on_total)

    return u, v_u, v_v


# ============================================================
# 固定点探索
# ============================================================

def find_fixed_points(par):
    u_nc, v_u_nc, v_v_nc = compute_nullclines(par, N=2000)
    diff = v_u_nc - v_v_nc
    valid = np.isfinite(diff)
    s0 = np.sign(diff[:-1])
    s1 = np.sign(diff[1:])
    idx = np.where(valid[:-1] & valid[1:] & (s0 != s1))[0]

    fps = []
    for i in idx:
        u0_g = u_nc[i] - diff[i] * (u_nc[i + 1] - u_nc[i]) / (diff[i + 1] - diff[i])
        v0_g = np.interp(u0_g, u_nc, v_v_nc)
        def residual(y):
            return [du_dt(y[0], y[1], par), dv_dt(y[0], y[1], par)]
        sol_fp, info, ier, msg = fsolve(residual, [u0_g, v0_g], full_output=True)
        if ier == 1:
            fps.append(sol_fp)

    fps.sort(key=lambda p: p[0])
    return fps


# ============================================================
# Euler–Maruyama 法による SDE シミュレーション
# ============================================================

def euler_maruyama(u0, v0, par, sigma_u, sigma_v,
                   T=50.0, dt=1e-4, seed=None):
    """
    du = f(u,v) dt + sigma_u * dW_u
    dv = g(u,v) dt + sigma_v * dW_v

    を Euler–Maruyama 法で解く。
    u, v は [0, Ptot], [0, Vtot] に反射壁でクリップする。

    Parameters
    ----------
    T     : 総シミュレーション時間
    dt    : 時間刻み幅
    seed  : 乱数シード（再現性のため）

    Returns
    -------
    t_arr, u_arr, v_arr : 時系列（間引き済み）
    """
    rng = np.random.default_rng(seed)

    Ptot = par["Ptot"]
    Vtot = par["Vtot"]

    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    # メモリ節約のため記録は間引く（every_k ステップごとに保存）
    every_k = max(1, int(1e-3 / dt))   # 約 0.001 ごとに記録
    n_save = n_steps // every_k + 1

    t_arr = np.empty(n_save)
    u_arr = np.empty(n_save)
    v_arr = np.empty(n_save)

    u, v = u0, v0
    save_idx = 0

    for step in range(n_steps):
        # 記録
        if step % every_k == 0:
            t_arr[save_idx] = step * dt
            u_arr[save_idx] = u
            v_arr[save_idx] = v
            save_idx += 1

        # 決定論的項
        fu = du_dt(u, v, par)
        fv = dv_dt(u, v, par)

        # ノイズ項（ガウシアンホワイトノイズ）
        dW_u = rng.standard_normal()
        dW_v = rng.standard_normal()

        # Euler–Maruyama 更新
        u += fu * dt + sigma_u * sqrt_dt * dW_u
        v += fv * dt + sigma_v * sqrt_dt * dW_v

        # 反射壁（濃度は物理的に非負かつ保存量以下）
        u = np.clip(u, 0.0, Ptot)
        v = np.clip(v, 0.0, Vtot)

    # 最終ステップ記録
    if save_idx < n_save:
        t_arr[save_idx] = n_steps * dt
        u_arr[save_idx] = u
        v_arr[save_idx] = v
        save_idx += 1

    return t_arr[:save_idx], u_arr[:save_idx], v_arr[:save_idx]


# ============================================================
# 描画ヘルパー（1条件分の時系列＋相平面を描画）
# ============================================================

def run_and_plot(par_local, ax_ts, ax_pp, sigma_u, sigma_v,
                 T_sim=50.0, dt=1e-4, seed=42, label_suffix=""):
    """
    1 条件分（パラメータ par_local）のシミュレーション＋描画を行う。
    ax_ts : 時系列用 Axes
    ax_pp : 相平面用 Axes
    """
    # ---- 固定点 ----
    fps = find_fixed_points(par_local)
    vPI3K_val = par_local["vPI3K"]
    print(f"\n--- vPI3K = {vPI3K_val} ---")
    print(f"  固定点の数: {len(fps)}")
    for i, fp in enumerate(fps):
        print(f"    FP{i}: u = {fp[0]:.6f}, v = {fp[1]:.6f}")

    u_rest, v_rest = fps[0]
    print(f"  安定固定点（静止状態）: u* = {u_rest:.6f}, v* = {v_rest:.6f}")

    # ---- SDE シミュレーション ----
    print(f"  Euler–Maruyama: T = {T_sim}, dt = {dt}, steps = {int(T_sim/dt)}")
    print("  計算中...")

    t, u_t, v_t = euler_maruyama(u_rest, v_rest, par_local,
                                  sigma_u, sigma_v,
                                  T=T_sim, dt=dt, seed=seed)
    print(f"  完了. 記録点数: {len(t)}")
    print(f"  u range: [{u_t.min():.4f}, {u_t.max():.4f}]")
    print(f"  v range: [{v_t.min():.4f}, {v_t.max():.4f}]")

    # ---- ヌルクライン ----
    u_nc, v_u_nc, v_v_nc = compute_nullclines(par_local)

    # ====== 時系列プロット ======
    ax_ts.plot(t, u_t, color="tab:blue", linewidth=0.4, alpha=0.8, label="u (PIP3)")
    ax_ts.plot(t, v_t, color="tab:orange", linewidth=0.4, alpha=0.8, label="v (membrane PTEN)")
    ax_ts.axhline(u_rest, color="tab:blue", ls=":", alpha=0.5, linewidth=1,
                  label=f"u* = {u_rest:.4f}")
    ax_ts.axhline(v_rest, color="tab:orange", ls=":", alpha=0.5, linewidth=1,
                  label=f"v* = {v_rest:.4f}")
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Concentration")
    ax_ts.set_title(f"vPI3K = {vPI3K_val}  "
                    f"(σ_u = {sigma_u}, σ_v = {sigma_v}, seed = {seed})")
    ax_ts.legend(fontsize=7, loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    # ====== 相平面プロット ======
    Ptot, Vtot = par_local["Ptot"], par_local["Vtot"]
    Ng = 20
    uu = np.linspace(0.0, Ptot, Ng)
    vv = np.linspace(0.0, Vtot, Ng)
    UU, VV = np.meshgrid(uu, vv)
    DU = du_dt(UU, VV, par_local)
    DV = dv_dt(UU, VV, par_local)
    speed = np.hypot(DU, DV)
    eps = 1e-12
    DU_n = DU / (speed + eps)
    DV_n = DV / (speed + eps)
    ax_pp.quiver(UU, VV, DU_n, DV_n, color="0.85", scale=25, width=0.003, zorder=0)

    ax_pp.plot(u_nc, v_u_nc, label="u-nullcline (du/dt=0)", color="tab:green",
               linewidth=2, zorder=2)
    ax_pp.plot(u_nc, v_v_nc, label="v-nullcline (dv/dt=0)", color="tab:red",
               linewidth=2, zorder=2)

    for fp in fps:
        ax_pp.plot(fp[0], fp[1], "ko", markersize=7, zorder=5)
    ax_pp.plot([], [], "ko", markersize=7, label="fixed points")

    ax_pp.plot(u_t, v_t, color="tab:blue", linewidth=0.2, alpha=0.3, zorder=3,
               label="stochastic trajectory")
    ax_pp.plot(u_rest, v_rest, "o", color="tab:cyan", markersize=10, zorder=4,
               markeredgecolor="k", markeredgewidth=0.8, label="start (rest state)")

    ax_pp.set_xlabel("u (PIP3)")
    ax_pp.set_ylabel("v (membrane-bound PTEN)")
    ax_pp.set_title(f"Phase plane — vPI3K = {vPI3K_val}")
    ax_pp.set_xlim(0.0, Ptot)
    ax_pp.set_ylim(0.0, Vtot)
    ax_pp.set_aspect("equal")
    ax_pp.legend(fontsize=7, loc="upper right")
    ax_pp.grid(True, alpha=0.3)


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":

    T_sim = 50.0     # 総時間（長めに取って複数回の発火を観察）
    dt = 1e-4        # 時間刻み（tau_u=0.01 に対して十分小さく）
    seed = 42        # 乱数シード（再現性。None にするとランダム）

    # ---- 2 条件のパラメータを用意 ----
    par1 = dict(par)                              # オリジナル vPI3K = 1.274（興奮あり）
    par2 = dict(par); par2["vPI3K"] = 0.75        # vPI3K = 0.75（興奮なし）

    print(f"ノイズ強度: sigma_u = {sigma_u}, sigma_v = {sigma_v}")

    # ---- Figure: 2 行 × 2 列 ----
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2],
                  hspace=0.35, wspace=0.30)

    ax_ts1 = fig.add_subplot(gs[0, 0])                    # 左上: 時系列 (vPI3K=1.274)
    ax_ts2 = fig.add_subplot(gs[0, 1], sharey=ax_ts1)     # 右上: 時系列 (vPI3K=0.75) — 縦軸共有
    ax_pp1 = fig.add_subplot(gs[1, 0])                    # 左下: 相平面 (vPI3K=1.274)
    ax_pp2 = fig.add_subplot(gs[1, 1], sharey=ax_pp1)     # 右下: 相平面 (vPI3K=0.75) — 縦軸共有

    # ---- 左列: vPI3K = 1.274（オリジナル、興奮あり） ----
    run_and_plot(par1, ax_ts1, ax_pp1, sigma_u, sigma_v,
                 T_sim=T_sim, dt=dt, seed=seed)

    # ---- 右列: vPI3K = 0.75（興奮なし） ----
    run_and_plot(par2, ax_ts2, ax_pp2, sigma_u, sigma_v,
                 T_sim=T_sim, dt=dt, seed=seed)

    # 右列の縦軸ラベルを非表示にして見やすくする
    ax_ts2.set_ylabel("")
    ax_pp2.set_ylabel("")

    plt.suptitle("PIP3–PTEN excitable system with stochastic noise — "
                 "comparison of vPI3K values",
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
