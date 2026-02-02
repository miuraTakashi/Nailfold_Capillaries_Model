import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ----------------------------
# PIP3–PTEN “生化学型”興奮系（空間一様：ODE）でヌルクラインを描く（形1：PTEN膜結合に基礎付着 k_on0 を追加）
#
# u := PIP3（膜上濃度）, v := 膜結合PTEN
# p2 = Ptot - u
#
# du/dt =  (1/tau_u) * [vPI3K * A(u) * p2/(KPI3K + p2)  -  vPTEN * v * u/(KPTEN+u) - k_leak * u]
#          ↑時間スケールパラメータ（tau_uが小さいほどuの変化が速い）
#
# dv/dt =  (k_on0(u) + kon*p2)*(Vtot - v)  -  koff*v
#          ↑PIP3依存の基礎結合速度増加（仮説的メカニズム）
#          k_on0(u) = k_on0_base * (1 + gamma*u/(Ku_k_on0+u))

#
# A(u) = 1 + alpha * u^n/(KA^n + u^n)   （PIP3側の正帰還）
#
# 【Remark: 仮定しているメカニズム】
# PIP3依存の基礎結合速度（k_on0）増加は、以下のような仮説的メカニズムに基づいています：
# 1. PIP3が増えると、膜の構造や電荷が変化する可能性
# 2. PIP3が増えると、PIP2非依存の基礎的PTEN膜結合速度が増加する
# 3. これにより、k_on_total = k_on0(u) + kon*p2 がuの増加に対してより強く反応し、
#    vのヌルクラインが正の傾きになる可能性がある
# 
# 注意：このメカニズムは文献では直接報告されていませんが、vのヌルクラインの
# 傾きを正にするための数学的に妥当な方法の一つです。
# 生物学的妥当性については、今後の実験的検証が必要です。
# ----------------------------

# パラメータ（例：必要に応じて調整）
par = dict(
    Ptot=1.000,     # PIP2+PIP3 総量（膜上）
    vPI3K=1.274,   # PI3K反応の最大速度係数
    KPI3K=0.010,   # PI3K側の飽和定数
    vPTEN=6.810,   # PTEN反応の最大速度係数
    KPTEN=0.100,   # PTEN側の飽和定数（Km相当の実効値）
    k_leak=5.667,  # PIP3の漏れ
    alpha=5.000,    # 正帰還の強さ
    KA=0.300,      # 正帰還の半飽和
    hill_n=4.021,     # 正帰還のHill係数
    kon=2.000,      # PTEN膜結合（PIP2依存）の速度係数
    koff=18.000,    # PTEN膜解離の速度係数
    k_on0_base=7.143,  # PIP2非依存の基礎的膜再付着の基礎値
    Vtot=0.984,     # PTEN総量（膜結合 + 細胞質プールの代表）
    gamma=13.571,    # PIP3依存の基礎結合速度増加の強さ（仮説的メカニズム）
    Ku_k_on0=0.347, # PIP3依存の基礎結合速度増加の半飽和定数
    tau_u=0.010,    # u（PIP3）の時間スケール（小さいほど変化が速い）
)

def A(u, par):
    """PIP3側の正帰還（PI3K活性化の実効増幅）"""
    alpha, KA, n = par["alpha"], par["KA"], par["hill_n"]
    return 1.0 + alpha * (u**n) / (KA**n + u**n)

def k_on0_func(u, par):
    """PIP3依存の基礎結合速度（仮説的メカニズム）"""
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
    # tau_uで時間スケールを調整（小さいほど変化が速い）
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
    # PIP3依存の基礎結合速度増加：PIP3が増えるとk_on0が増える（仮説的メカニズム）
    k_on0_u = k_on0_func(u, par)
    k_on_total = k_on0_u + kon * p2
    return k_on_total * (Vtot - v) - koff * v

def nullclines_shape1(par, N=500):
    """
    形5のヌルクライン（PIP3依存の基礎結合速度増加を含む）：
      u-nullcline: du/dt = 0 から v = F(u)
      v-nullcline: dv/dt = 0 から v = G(u)（k_on0(u)項を含む）
    """
    Ptot, Vtot = par["Ptot"], par["Vtot"]

    # u=0で式が発散/0割になりやすいので少し避ける
    # uの描画・計算範囲を 0.02 < u < 1.0 に設定
    # また u=Ptot では p2=Ptot-u=0 となり PI3K項が消えて漏れ項だけが残るため、
    # v=F(u) が大きく負に振れやすい。端点を少し避ける。
    u_max = max(0.02, Ptot - 1e-6)
    u = np.linspace(0.02, u_max, N)
    p2 = Ptot - u

    # ---- u-ヌルクライン：du/dt = 0 を v = F(u) として解く ----
    vPI3K, KPI3K = par["vPI3K"], par["KPI3K"]
    vPTEN, KPTEN = par["vPTEN"], par["KPTEN"]
    k_leak = par["k_leak"]

    prod = vPI3K * A(u, par) * p2 / (KPI3K + p2)         # PIP2→PIP3
    denom = vPTEN * (u / (KPTEN + u))                    # v*u/(K+u) の v以外
    # du/dt = prod - vPTEN*v*u/(KPTEN+u) - k_leak*u = 0 より
    # v = (prod - k_leak*u) / [vPTEN * u/(KPTEN+u)]
    numer = prod - k_leak * u
    v_u = np.where(numer > 0, numer / denom, np.nan)     # 物理的に意味の薄い負値は描かない

    # ---- v-ヌルクライン（形5）：dv/dt = 0 を v = G(u) として解く ----
    kon, koff = par["kon"], par["koff"]
    # 0 = (k_on0(u) + kon*p2)*(Vtot - v) - koff*v
    # k_on0(u) = k_on0_base * (1 + gamma*u/(Ku_k_on0+u))
    # => v = Vtot*(k_on0(u) + kon*p2)/(koff + k_on0(u) + kon*p2)
    k_on0_u = k_on0_func(u, par)
    k_on_total = k_on0_u + kon * p2
    v_v = Vtot * k_on_total / (koff + k_on_total)

    return u, v_u, v_v

def fixed_points_from_nullclines(u, v_u, v_v):
    """ヌルクライン交点（v_u - v_v = 0）の符号反転から近似固定点を得る"""
    diff = v_u - v_v
    # NaN が混ざっていても壊れないように、有効区間だけ見る
    valid = np.isfinite(diff)
    s0 = np.sign(diff[:-1])
    s1 = np.sign(diff[1:])
    idx = np.where(valid[:-1] & valid[1:] & (s0 != s1))[0]
    fps = []
    for i in idx:
        u0, u1 = u[i], u[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        # 線形補間で交点
        uc = u0 - d0 * (u1 - u0) / (d1 - d0)
        vc = np.interp(uc, u, v_v)
        fps.append((uc, vc))
    return fps

# ---- 実行（インタラクティブ版） ----
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.45)  # スライダー用のスペース確保（パラメータ追加により拡張）

eps = 1e-12

# スライダーの位置と範囲を定義（2列配置）
# 左列
slider_params_left = [
    # (name, label, valmin, valmax, valinit, ypos)
    ("vPI3K", "vPI3K", 0.0, 5.0, par["vPI3K"], 0.36),
    ("KPI3K", "KPI3K", 0.01, 1.0, par["KPI3K"], 0.33),
    ("vPTEN", "vPTEN", 0.0, 20.0, par["vPTEN"], 0.30),
    ("KPTEN", "KPTEN", 0.01, 1.0, par["KPTEN"], 0.27),
    ("k_leak", "k_leak", 0.0, 10.0, par["k_leak"], 0.24),
    ("alpha", "alpha", 0.0, 20.0, par["alpha"], 0.21),
    ("KA", "KA", 0.01, 1.0, par["KA"], 0.18),
    ("tau_u", "tau_u", 0.01, 10.0, par["tau_u"], 0.15),
]

# 右列
slider_params_right = [
    ("hill_n", "hill_n", 1.0, 10.0, par["hill_n"], 0.36),
    ("kon", "kon", 0.0, 10.0, par["kon"], 0.33),
    ("koff", "koff", 0.0, 50.0, par["koff"], 0.30),
    ("k_on0_base", "k_on0_base", 0.0, 20.0, par["k_on0_base"], 0.27),
    ("gamma", "gamma", 0.0, 100.0, par["gamma"], 0.24),
    ("Ku_k_on0", "Ku_k_on0", 0.01, 1.0, par["Ku_k_on0"], 0.21),
    ("Vtot", "Vtot", 0.1, 3.0, par["Vtot"], 0.18),
    ("Ptot", "Ptot", 0.1, 3.0, par["Ptot"], 0.15),
]

sliders = {}
# 左列のスライダー
for name, label, vmin, vmax, vinit, ypos in slider_params_left:
    ax_slider = plt.axes([0.10, ypos, 0.35, 0.02])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit, valfmt='%.3f')
    sliders[name] = slider

# 右列のスライダー
for name, label, vmin, vmax, vinit, ypos in slider_params_right:
    ax_slider = plt.axes([0.55, ypos, 0.35, 0.02])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit, valfmt='%.3f')
    sliders[name] = slider

# 更新関数
def update(val):
    # パラメータを更新
    for name, slider in sliders.items():
        par[name] = slider.val
    
    # ヌルクラインを再計算
    u_new, v_u_new, v_v_new = nullclines_shape1(par)
    fps_new = fixed_points_from_nullclines(u_new, v_u_new, v_v_new)
    
    # ベクトル場を再計算（高速化のためグリッドを粗く、quiverを使用）
    Ptot_new = par["Ptot"]
    Vtot_new = par["Vtot"]
    # ベクトル場のグリッドを粗くして高速化（15x15に減らす）
    uu_new = np.linspace(0.0, Ptot_new, 15)
    vv_new = np.linspace(0.0, Vtot_new, 15)
    UU_new, VV_new = np.meshgrid(uu_new, vv_new)
    DU_new = du_dt(UU_new, VV_new, par)
    DV_new = dv_dt(UU_new, VV_new, par)
    speed_new = np.hypot(DU_new, DV_new)
    # 正規化（方向のみ表示）
    DU_n_new = DU_new / (speed_new + eps)
    DV_n_new = DV_new / (speed_new + eps)
    
    # プロットを更新
    ax.clear()
    # streamplotの代わりにquiverを使用（より高速）
    ax.quiver(UU_new, VV_new, DU_n_new, DV_n_new, color="0.75", scale=20, width=0.003, zorder=0)
    ax.plot(u_new, v_u_new, label="u-nullcline: du/dt = 0  (v = F(u))", zorder=2, linewidth=2)
    ax.plot(u_new, v_v_new, label="v-nullcline (shape5): dv/dt = 0  (v = G(u))", zorder=2, linewidth=2)
    
    if fps_new:
        ax.scatter([p[0] for p in fps_new], [p[1] for p in fps_new], c='red', s=100, marker='o', label="fixed points (approx)", zorder=3)
    
    ax.set_xlabel("u = PIP3")
    ax.set_ylabel("v = membrane-bound PTEN")
    ax.set_title("Nullclines + vector field (Shape 5: PIP3-dependent basal binding rate increase)")
    ax.set_xlim(0.0, Ptot_new)
    ax.set_ylim(0.0, Vtot_new)
    ax.set_aspect('equal')  # アスペクト比を1:1に設定
    ax.legend()
    fig.canvas.draw_idle()

# 各スライダーに更新関数を接続
for slider in sliders.values():
    slider.on_changed(update)

# 初期プロット
update(None)

plt.show()
