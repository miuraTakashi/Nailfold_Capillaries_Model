"""
PIP3 (u) fast, RASA1-like feedback (w) intermediate, PTEN (v) slow-constant model.

Assumption:
  - v (membrane PTEN) is much slower than u and w over the analysis window.
  - Therefore v is treated as a constant parameter v_const and eliminated
    from the dynamic state variables.

Reduced system (2D):
  du/dt = (1/tau_u) * [ PI3K_prod(u) * inhibition_w(w) - PTEN_deg(u, v_const) - k_leak*u ]
  dw/dt = (w_inf(u) - w) / tau_w

This file is derived from dynamics_rasgap.py with v fixed.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


par = dict(
    # --- original PIP3-PTEN parameters ---
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
    delta_off=0.0,
    Ku_k_off=0.4,
    tau_u=0.010,
    # --- RASA1-like feedback (w) ---
    beta_w=3.0,
    tau_w=0.5,
    w_max=1.0,
    Kw=0.25,
    hill_m=4.0,
    # --- slow PTEN is fixed in this reduced model ---
    v_const=0.80,
)


def A(u, p):
    alpha, KA, n = p["alpha"], p["KA"], p["hill_n"]
    return 1.0 + alpha * (u**n) / (KA**n + u**n)


def inhibition_w(w, p):
    return 1.0 / (1.0 + p["beta_w"] * np.maximum(w, 0.0))


def w_inf(u, p):
    w_max, Kw, m = p["w_max"], p["Kw"], p["hill_m"]
    u = np.maximum(u, 0.0)
    return w_max * (u**m) / (Kw**m + u**m)


def du_dt(u, w, p):
    Ptot = p["Ptot"]
    p2 = Ptot - u
    vPI3K, KPI3K = p["vPI3K"], p["KPI3K"]
    vPTEN, KPTEN = p["vPTEN"], p["KPTEN"]
    k_leak = p["k_leak"]
    tau_u = p["tau_u"]
    v_const = p["v_const"]
    prod = vPI3K * A(u, p) * p2 / (KPI3K + p2)
    deg = vPTEN * v_const * u / (KPTEN + u)
    return (1.0 / tau_u) * (prod * inhibition_w(w, p) - deg - k_leak * u)


def dw_dt(u, w, p):
    return (w_inf(u, p) - w) / p["tau_w"]


def ode_system(t, y, p):
    u, w = y
    return [du_dt(u, w, p), dw_dt(u, w, p)]


def find_fixed_point(p, guess=(0.05, 0.0)):
    def residual(y):
        u, w = y
        return [du_dt(u, w, p), dw_dt(u, w, p)]

    sol, info, ier, msg = fsolve(residual, guess, full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(f"fsolve failed: {msg}")
    return sol


def simulate(y0, p, t_span=(0.0, 6.0), n_eval=3000):
    t_eval = np.linspace(*t_span, n_eval)
    return solve_ivp(
        ode_system,
        t_span,
        y0,
        args=(p,),
        t_eval=t_eval,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )


def nullclines_uw(p, n=700):
    """
    Compute u- and w-nullclines in (u,w):
      - du/dt = 0 => w = F(u)
      - dw/dt = 0 => w = w_inf(u)
    """
    Ptot, w_max = p["Ptot"], p["w_max"]
    u = np.linspace(0.0, Ptot * 0.999, n)

    vPI3K, KPI3K = p["vPI3K"], p["KPI3K"]
    vPTEN, KPTEN = p["vPTEN"], p["KPTEN"]
    k_leak = p["k_leak"]
    v_const = p["v_const"]
    beta = p["beta_w"]

    p2 = Ptot - u
    prod = vPI3K * A(u, p) * p2 / (KPI3K + p2)
    deg = vPTEN * v_const * u / (KPTEN + u)
    rhs = deg + k_leak * u

    w_u = np.full_like(u, np.nan)
    valid = (prod > 0.0) & (rhs > 0.0)
    inhib = np.full_like(u, np.nan)
    inhib[valid] = rhs[valid] / prod[valid]
    ok = valid & (inhib > 0.0) & (inhib <= 1.0)
    w_u[ok] = (1.0 / inhib[ok] - 1.0) / beta
    w_u[(w_u < 0.0) | (w_u > w_max)] = np.nan

    w_w = w_inf(u, p)
    return u, w_u, w_w


def _peak_u_for_du(du, u_rest, w_rest, p, t_span):
    y0 = [min(p["Ptot"] * 0.999, max(0.0, float(u_rest) + float(du))), float(w_rest)]
    sol = simulate(y0, p, t_span=t_span, n_eval=2500)
    if not sol.success:
        return np.nan
    return float(np.nanmax(sol.y[0]))


def choose_sub_super_du(u_rest, w_rest, p, *, t_span=(0.0, 8.0)):
    """
    Auto-select sub/super perturbations around an empirical threshold.
    Threshold criterion: peak(u) exceeds u_rest + 0.12.
    """
    du_scan = np.linspace(0.002, 0.20, 40)
    peaks = np.array([_peak_u_for_du(du, u_rest, w_rest, p, t_span) for du in du_scan], dtype=float)
    fire_level = float(u_rest + 0.12)
    fired = np.where(peaks > fire_level)[0]

    if len(fired) == 0:
        # fallback when no clear threshold-like jump is found
        return 0.01, 0.08, peaks, du_scan, fire_level

    k = int(fired[0])
    du_super = float(du_scan[k])
    du_sub = float(du_scan[max(0, k - 1)])
    return du_sub, du_super, peaks, du_scan, fire_level


if __name__ == "__main__":
    # Fixed PTEN level (slow variable treated as constant)
    u_rest, w_rest = find_fixed_point(par, guess=(0.05, 0.02))
    print(f"rest (v fixed): u*={u_rest:.6f}, w*={w_rest:.6f}, v_const={par['v_const']:.6f}")

    # Two initial conditions in u around rest (sub/super), auto-picked near threshold
    t_span = (0.0, 8.0)
    du_small, du_large, peaks, du_scan, fire_level = choose_sub_super_du(
        u_rest, w_rest, par, t_span=t_span
    )
    print(
        "auto threshold search: "
        f"du_sub={du_small:.4f}, du_super={du_large:.4f}, fire_level(u_peak)={fire_level:.4f}"
    )
    y0_sub = [max(0.0, float(u_rest) + du_small), float(w_rest)]
    y0_sup = [min(par["Ptot"] * 0.98, float(u_rest) + du_large), float(w_rest)]

    sol_sub = simulate(y0_sub, par, t_span=t_span)
    sol_sup = simulate(y0_sup, par, t_span=t_span)

    u_nc, w_du0, w_dw0 = nullclines_uw(par)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={"hspace": 0.35, "wspace": 0.28})
    ax_ts1, ax_ts2 = axes[0, 0], axes[0, 1]
    ax_uw, ax_ww = axes[1, 0], axes[1, 1]

    for ax, sol, title in [
        (ax_ts1, sol_sub, f"Sub perturbation (Du={du_small})"),
        (ax_ts2, sol_sup, f"Super perturbation (Du={du_large})"),
    ]:
        t = sol.t
        u, w = sol.y
        ax.plot(t, u, color="tab:blue", lw=1.8, label="u (PIP3)")
        ax.plot(t, w, color="tab:purple", lw=1.8, label="w (RASA1-like)")
        ax.axhline(u_rest, color="tab:blue", ls=":", alpha=0.4, lw=1)
        ax.axhline(w_rest, color="tab:purple", ls=":", alpha=0.4, lw=1)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Level")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    ax_uw.plot(sol_sub.y[0], sol_sub.y[1], color="tab:cyan", lw=2, label="sub")
    ax_uw.plot(sol_sup.y[0], sol_sup.y[1], color="tab:pink", lw=2, label="super")
    ax_uw.plot([u_rest], [w_rest], "ko", ms=6, label="rest")
    ax_uw.plot(u_nc, w_du0, color="tab:red", lw=2.0, label="du/dt = 0")
    ax_uw.plot(u_nc, w_dw0, color="tab:green", lw=2.0, label="dw/dt = 0")
    # Vector field (normalized) in (u,w)
    u_min, u_max = 0.0, float(par["Ptot"] * 0.999)
    w_min, w_max = 0.0, float(par["w_max"])
    u_vf = np.linspace(u_min, u_max, 17)
    w_vf = np.linspace(w_min, w_max, 17)
    UU, WW = np.meshgrid(u_vf, w_vf, indexing="xy")
    DU = du_dt(UU, WW, par)
    DW = dw_dt(UU, WW, par)
    speed = np.hypot(DU, DW)
    eps = 1e-12
    ax_uw.quiver(
        UU,
        WW,
        DU / (speed + eps),
        DW / (speed + eps),
        color="0.82",
        scale=26,
        width=0.003,
        zorder=0,
    )
    ax_uw.set_xlabel("u (PIP3)")
    ax_uw.set_ylabel("w (RASA1-like)")
    ax_uw.set_title("Phase plane: (u,w) with nullclines")
    ax_uw.set_xlim(u_min, u_max)
    ax_uw.set_ylim(w_min, w_max)
    ax_uw.grid(True, alpha=0.3)
    ax_uw.legend(fontsize=8, loc="upper right")

    ax_ww.plot(sol_sub.t, sol_sub.y[1], color="tab:cyan", lw=2, label="w sub")
    ax_ww.plot(sol_sup.t, sol_sup.y[1], color="tab:pink", lw=2, label="w super")
    ax_ww.axhline(w_rest, color="k", ls=":", lw=1, alpha=0.5, label="w rest")
    ax_ww.set_xlabel("Time")
    ax_ww.set_ylabel("w (RASA1-like)")
    ax_ww.set_title("Slow/intermediate variable trajectory")
    ax_ww.grid(True, alpha=0.3)
    ax_ww.legend(fontsize=8, loc="upper right")
    # show du-to-peak map to confirm threshold-like transition
    ax_ww2 = ax_ww.twinx()
    ax_ww2.plot(du_scan, peaks, color="0.4", lw=1.2, alpha=0.85, label="peak(u) vs Du")
    ax_ww2.axhline(fire_level, color="0.45", ls="--", lw=1.0, alpha=0.8)
    ax_ww2.set_ylabel("peak(u) from Du scan", color="0.35")
    ax_ww2.tick_params(axis="y", colors="0.35")

    fig.suptitle("Reduced model: fast u, intermediate w, slow PTEN fixed (v=v_const)", fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.93)
    plt.show()

