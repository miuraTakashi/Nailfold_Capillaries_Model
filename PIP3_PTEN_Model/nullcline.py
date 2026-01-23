import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# PIP3–PTEN “生化学型”興奮系（空間一様：ODE）でヌルクラインを描く（形1：PTEN膜結合に基礎付着 k_on0 を追加）
#
# u := PIP3（膜上濃度）, v := 膜結合PTEN
# p2 = Ptot - u
#
# du/dt =  vPI3K * A(u) * p2/(KPI3K + p2)  -  vPTEN * v * u/(KPTEN+u)
#
# dv/dt =  (k_on0 + kon*p2)*(Vtot - v)  -  koff*v
#
# A(u) = 1 + alpha * u^n/(KA^n + u^n)   （PIP3側の正帰還）
# ----------------------------

# パラメータ（例：必要に応じて調整）
par = dict(
    Ptot=1.0,     # PIP2+PIP3 総量（膜上）
    Vtot=1.0,     # PTEN 総量（膜結合 + 細胞質プールの代表）
    vPI3K=0.20,   # PI3K反応の最大速度係数
    KPI3K=0.20,   # PI3K側の飽和定数
    vPTEN=5.00,   # PTEN反応の最大速度係数
    KPTEN=0.10,   # PTEN側の飽和定数（Km相当の実効値）
    alpha=5.0,    # 正帰還の強さ
    KA=0.30,      # 正帰還の半飽和
    hill_n=4,     # 正帰還のHill係数
    kon=2.0,      # PTEN膜結合（PIP2依存）の速度係数
    koff=18.0,    # PTEN膜解離の速度係数
    k_on0=0.10,   # ★追加：PIP2非依存の基礎的膜再付着（形1）
)

def A(u, par):
    """PIP3側の正帰還（PI3K活性化の実効増幅）"""
    alpha, KA, n = par["alpha"], par["KA"], par["hill_n"]
    return 1.0 + alpha * (u**n) / (KA**n + u**n)

def nullclines_shape1(par, N=3000):
    """
    形1のヌルクライン：
      u-nullcline: du/dt = 0 から v = F(u)
      v-nullcline: dv/dt = 0 から v = G(u)（k_on0を含む）
    """
    Ptot, Vtot = par["Ptot"], par["Vtot"]

    # u=0で式が発散/0割になりやすいので少し避ける
    # uの描画・計算範囲を 0.02 < u < 1.0 に設定
    u = np.linspace(0.02, 1.0, N)
    p2 = Ptot - u

    # ---- u-ヌルクライン：du/dt = 0 を v = F(u) として解く ----
    vPI3K, KPI3K = par["vPI3K"], par["KPI3K"]
    vPTEN, KPTEN = par["vPTEN"], par["KPTEN"]

    prod = vPI3K * A(u, par) * p2 / (KPI3K + p2)         # PIP2→PIP3
    denom = vPTEN * (u / (KPTEN + u))                    # v*u/(K+u) の v以外
    v_u = prod / denom                                    # v = prod/denom

    # ---- v-ヌルクライン（形1）：dv/dt = 0 を v = G(u) として解く ----
    kon, koff, k_on0 = par["kon"], par["koff"], par["k_on0"]
    # 0 = (k_on0 + kon*p2)*(Vtot - v) - koff*v
    # => v = Vtot*(k_on0 + kon*p2)/(koff + k_on0 + kon*p2)
    v_v = Vtot * (k_on0 + kon * p2) / (koff + k_on0 + kon * p2)

    return u, v_u, v_v

def fixed_points_from_nullclines(u, v_u, v_v):
    """ヌルクライン交点（v_u - v_v = 0）の符号反転から近似固定点を得る"""
    diff = v_u - v_v
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    fps = []
    for i in idx:
        u0, u1 = u[i], u[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        # 線形補間で交点
        uc = u0 - d0 * (u1 - u0) / (d1 - d0)
        vc = np.interp(uc, u, v_v)
        fps.append((uc, vc))
    return fps

# ---- 実行 ----
u, v_u, v_v = nullclines_shape1(par)
fps = fixed_points_from_nullclines(u, v_u, v_v)

plt.figure()
plt.plot(u, v_u, label="u-nullcline: du/dt = 0  (v = F(u))")
plt.plot(u, v_v, label="v-nullcline (shape1): dv/dt = 0  (v = G(u))")

if fps:
    plt.plot([p[0] for p in fps], [p[1] for p in fps], "o", label="fixed points (approx)")

plt.xlabel("u = PIP3")
plt.ylabel("v = membrane-bound PTEN")
plt.title("Nullclines (Shape 1: basal PTEN reattachment k_on0 added)")
plt.legend()
plt.tight_layout()
plt.show()

print("Fixed points (u, v) ~")
for (uc, vc) in fps:
    print(f"  ({uc:.6f}, {vc:.6f})")

# 右端のvがどれだけ持ち上がったか（u->Ptotでの極限値）
Vtot, koff, k_on0 = par["Vtot"], par["koff"], par["k_on0"]
v_right_limit = Vtot * (k_on0) / (koff + k_on0)
print(f"\nRight-end v-limit (u->Ptot): Vtot*k_on0/(koff+k_on0) = {v_right_limit:.6e}")
