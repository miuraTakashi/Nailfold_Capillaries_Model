"""
PIP3–PTEN 興奮系ダイナミクスシミュレーション

nullcline.py で定義された du/dt, dv/dt を用い、
u(PIP3), v(膜結合PTEN) の初期値を与えて時間発展を数値的に解く。

安定固定点（静止状態）を数値的に求め、そこから
  (1) 閾値以下の摂動 → 静止状態へ直接戻る
  (2) 閾値以上の摂動 → 大きな興奮応答後に戻る
の 2 ケースを並べて比較する。

出力:
  上段左: 閾値以下の時系列  u(t), v(t)
  上段右: 閾値以上の時系列  u(t), v(t)
  下段  : 相平面上の 2 軌道（ヌルクライン付き）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ============================================================
# パラメータ（nullcline.py と共通）
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
# モデル関数（nullcline.py と同一）
# ============================================================

def A(u, par):
    """PIP3 側の正帰還（PI3K 活性化の実効増幅）"""
    alpha, KA, n = par["alpha"], par["KA"], par["hill_n"]
    return 1.0 + alpha * (u**n) / (KA**n + u**n)


def k_on0_func(u, par):
    """PIP3 依存の基礎結合速度"""
    k_on0_base, gamma, Ku_k_on0 = par["k_on0_base"], par["gamma"], par["Ku_k_on0"]
    return k_on0_base * (1.0 + gamma * u / (Ku_k_on0 + u))


def du_dt(u, v, par):
    """du/dt"""
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
    """dv/dt"""
    Ptot, Vtot = par["Ptot"], par["Vtot"]
    p2 = Ptot - u
    kon, koff = par["kon"], par["koff"]
    k_on0_u = k_on0_func(u, par)
    k_on_total = k_on0_u + kon * p2
    return k_on_total * (Vtot - v) - koff * v


# ============================================================
# ODE 系（solve_ivp 用）
# ============================================================

def ode_system(t, y, par):
    """solve_ivp に渡す右辺関数  y = [u, v]"""
    u, v = y
    return [du_dt(u, v, par), dv_dt(u, v, par)]


# ============================================================
# ヌルクライン計算（相平面プロット用）
# ============================================================

def compute_nullclines(par, N=500):
    """u-nullcline v=F(u) と v-nullcline v=G(u) を返す"""
    Ptot, Vtot = par["Ptot"], par["Vtot"]
    u_max = max(0.02, Ptot - 1e-6)
    u = np.linspace(0.02, u_max, N)
    p2 = Ptot - u

    # u-nullcline
    vPI3K, KPI3K = par["vPI3K"], par["KPI3K"]
    vPTEN, KPTEN = par["vPTEN"], par["KPTEN"]
    k_leak = par["k_leak"]
    prod = vPI3K * A(u, par) * p2 / (KPI3K + p2)
    denom = vPTEN * (u / (KPTEN + u))
    numer = prod - k_leak * u
    v_u = np.where(numer > 0, numer / denom, np.nan)

    # v-nullcline
    kon, koff = par["kon"], par["koff"]
    k_on0_u = k_on0_func(u, par)
    k_on_total = k_on0_u + kon * p2
    v_v = Vtot * k_on_total / (koff + k_on_total)

    return u, v_u, v_v


# ============================================================
# 固定点探索
# ============================================================

def find_fixed_points(par, u_guesses=None):
    """
    ヌルクライン交点を初期推定として fsolve で固定点を求める。
    安定固定点（最小 u）と不安定固定点（中間 u）を返す。
    """
    # ヌルクライン交点から初期推定を得る
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
        # fsolve で精密化
        def residual(y):
            return [du_dt(y[0], y[1], par), dv_dt(y[0], y[1], par)]
        sol_fp, info, ier, msg = fsolve(residual, [u0_g, v0_g], full_output=True)
        if ier == 1:  # 収束した場合のみ
            fps.append(sol_fp)

    # u の昇順にソート
    fps.sort(key=lambda p: p[0])
    return fps


def simulate(u0, v0, par, t_span=(0.0, 5.0), n_eval=5000):
    """初期値 (u0, v0) から ODE を数値積分する（stiff 系なので Radau を使用）"""
    t_eval = np.linspace(*t_span, n_eval)
    sol = solve_ivp(
        ode_system, t_span, [u0, v0],
        args=(par,),
        t_eval=t_eval,
        method="Radau",        # tau_u が小さく stiff なので陰的解法を使用
        rtol=1e-8, atol=1e-10,
    )
    return sol


# ============================================================
# シミュレーション実行
# ============================================================
if __name__ == "__main__":

    # ---- 固定点を探索 ----
    fps = find_fixed_points(par)
    print(f"固定点の数: {len(fps)}")
    for i, fp in enumerate(fps):
        print(f"  FP{i}: u = {fp[0]:.6f}, v = {fp[1]:.6f}")

    # 安定固定点（最小 u = 静止状態）
    u_rest, v_rest = fps[0]
    print(f"\n安定固定点（静止状態）: u* = {u_rest:.6f}, v* = {v_rest:.6f}")

    # 興奮系では、不安定固定点（中間 u）が存在すれば閾値の目安になる
    if len(fps) >= 2:
        u_thresh, v_thresh = fps[1]
        print(f"不安定固定点（閾値の目安）: u_thresh = {u_thresh:.6f}, v_thresh = {v_thresh:.6f}")
        # 閾値の目安: 安定固定点と不安定固定点の中間 u 付近
        du_half = (u_thresh - u_rest)
    else:
        # 固定点が 1 つしかない場合は手動で摂動量を設定
        du_half = 0.1

    # ---- 2 つの初期条件 ----
    # (1) 閾値以下: 固定点から u を少しだけ増やす
    delta_sub = du_half * 0.5
    u0_sub = u_rest + delta_sub
    v0_sub = v_rest

    # (2) 閾値以上: 閾値を明確に超える初期値（手動指定）
    u0_sup = 0.3
    v0_sup = 0.5

    print(f"\n--- 閾値以下 (sub-threshold) ---")
    print(f"  IC: u0 = {u0_sub:.6f}, v0 = {v0_sub:.6f}  (Δu = {delta_sub:.6f})")
    print(f"--- 閾値以上 (super-threshold) ---")
    print(f"  IC: u0 = {u0_sup:.6f}, v0 = {v0_sup:.6f}")

    # ---- 時間範囲 ----
    t_span = (0.0, 5.0)

    # ---- 数値積分（2 ケース） ----
    sol_sub = simulate(u0_sub, v0_sub, par, t_span)
    sol_sup = simulate(u0_sup, v0_sup, par, t_span)

    # ---- ヌルクライン ----
    u_nc, v_u_nc, v_v_nc = compute_nullclines(par)

    # ============================================================
    # プロット（上段: 時系列 x2, 下段: 相平面）
    # ============================================================
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.3], hspace=0.30, wspace=0.30)

    # 色の定義
    color_sub = "tab:cyan"      # 閾値以下
    color_sup = "tab:purple"    # 閾値以上

    # ------ 上段左: 閾値以下の時系列 ------
    ax_sub = fig.add_subplot(gs[0, 0])
    ax_sub.plot(sol_sub.t, sol_sub.y[0], label="u (PIP3)", color="tab:blue", linewidth=1.5)
    ax_sub.plot(sol_sub.t, sol_sub.y[1], label="v (membrane PTEN)", color="tab:orange", linewidth=1.5)
    ax_sub.axhline(u_rest, color="tab:blue", ls=":", alpha=0.4, label=f"u* = {u_rest:.4f}")
    ax_sub.axhline(v_rest, color="tab:orange", ls=":", alpha=0.4, label=f"v* = {v_rest:.4f}")
    ax_sub.set_xlabel("Time")
    ax_sub.set_ylabel("Concentration")
    ax_sub.set_title(f"Sub-threshold  (Δu = {delta_sub:.4f})")
    ax_sub.legend(fontsize=7, loc="upper right")
    ax_sub.grid(True, alpha=0.3)

    # ------ 上段右: 閾値以上の時系列 ------
    ax_sup = fig.add_subplot(gs[0, 1])
    ax_sup.plot(sol_sup.t, sol_sup.y[0], label="u (PIP3)", color="tab:blue", linewidth=1.5)
    ax_sup.plot(sol_sup.t, sol_sup.y[1], label="v (membrane PTEN)", color="tab:orange", linewidth=1.5)
    ax_sup.axhline(u_rest, color="tab:blue", ls=":", alpha=0.4, label=f"u* = {u_rest:.4f}")
    ax_sup.axhline(v_rest, color="tab:orange", ls=":", alpha=0.4, label=f"v* = {v_rest:.4f}")
    ax_sup.set_xlabel("Time")
    ax_sup.set_ylabel("Concentration")
    ax_sup.set_title(f"Super-threshold  (u0={u0_sup}, v0={v0_sup})")
    ax_sup.legend(fontsize=7, loc="upper right")
    ax_sup.grid(True, alpha=0.3)

    # ------ 下段: 相平面（2 軌道を重ねて表示） ------
    ax_ph = fig.add_subplot(gs[1, :])

    # ベクトル場
    Ptot, Vtot = par["Ptot"], par["Vtot"]
    Ng = 20
    uu = np.linspace(0.0, Ptot, Ng)
    vv = np.linspace(0.0, Vtot, Ng)
    UU, VV = np.meshgrid(uu, vv)
    DU = du_dt(UU, VV, par)
    DV = dv_dt(UU, VV, par)
    speed = np.hypot(DU, DV)
    eps = 1e-12
    DU_n = DU / (speed + eps)
    DV_n = DV / (speed + eps)
    ax_ph.quiver(UU, VV, DU_n, DV_n, color="0.85", scale=25, width=0.003, zorder=0)

    # ヌルクライン
    ax_ph.plot(u_nc, v_u_nc, label="u-nullcline (du/dt=0)", color="tab:green",
               linewidth=2, zorder=2)
    ax_ph.plot(u_nc, v_v_nc, label="v-nullcline (dv/dt=0)", color="tab:red",
               linewidth=2, zorder=2)

    # 固定点
    for fp in fps:
        ax_ph.plot(fp[0], fp[1], "ko", markersize=7, zorder=5)
    ax_ph.plot(u_rest, v_rest, "ko", markersize=7, label="fixed points")

    # 閾値以下の軌道
    ax_ph.plot(sol_sub.y[0], sol_sub.y[1], color=color_sub, linewidth=1.5,
               alpha=0.9, zorder=3, label=f"sub-threshold (Δu={delta_sub:.4f})")
    ax_ph.plot(u0_sub, v0_sub, "o", color=color_sub, markersize=8, zorder=4,
               markeredgecolor="k", markeredgewidth=0.5)

    # 閾値以上の軌道
    ax_ph.plot(sol_sup.y[0], sol_sup.y[1], color=color_sup, linewidth=1.5,
               alpha=0.9, zorder=3, label=f"super-threshold ({u0_sup}, {v0_sup})")
    ax_ph.plot(u0_sup, v0_sup, "o", color=color_sup, markersize=8, zorder=4,
               markeredgecolor="k", markeredgewidth=0.5)

    ax_ph.set_xlabel("u (PIP3)")
    ax_ph.set_ylabel("v (membrane-bound PTEN)")
    ax_ph.set_title("Phase plane — sub- vs super-threshold perturbation")
    ax_ph.set_xlim(0.0, Ptot)
    ax_ph.set_ylim(0.0, Vtot)
    ax_ph.set_aspect("equal")
    ax_ph.legend(fontsize=8, loc="upper right")
    ax_ph.grid(True, alpha=0.3)

    plt.suptitle("PIP3–PTEN excitable dynamics: threshold comparison", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
