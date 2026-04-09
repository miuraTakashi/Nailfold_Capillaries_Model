"""
Variant of RASA1_PTEN_with_eliminated_PIP3.py with swapped time scales:
  tau_w <-> tau_v
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from RASA1_PTEN_with_eliminated_PIP3 import (
    par as base_par,
    find_fixed_point,
    u_hat,
    simulate,
    compute_field_and_nullclines,
)


par = dict(base_par)
par["tau_w"], par["tau_v"] = base_par["tau_v"], base_par["tau_w"]


if __name__ == "__main__":
    w_rest, v_rest = find_fixed_point(par, guess=(0.05, 0.25))
    u_rest = u_hat(float(v_rest), float(w_rest), par)
    print(
        "rest (u eliminated, swapped taus): "
        f"u_hat={u_rest:.6f}, w*={w_rest:.6f}, v*={v_rest:.6f}, "
        f"tau_w={par['tau_w']:.3f}, tau_v={par['tau_v']:.3f}"
    )

    # perturb v downward (PTEN depletion-like input)
    dv_small = 0.01
    dv_large = 0.06
    y0_sub = (float(w_rest), max(0.0, float(v_rest) - dv_small))
    y0_sup = (float(w_rest), max(0.0, float(v_rest) - dv_large))

    t_span = (0.0, 8.0)
    sol_sub = simulate(y0_sub, par, t_span=t_span)
    sol_sup = simulate(y0_sup, par, t_span=t_span)

    def u_series(sol):
        ws, vs = sol.y
        return np.array([u_hat(float(v), float(w), par) for w, v in zip(ws, vs)])

    u_sub = u_series(sol_sub)
    u_sup = u_series(sol_sup)

    W, V, Fw, Fv = compute_field_and_nullclines(par, u_ref=u_rest)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={"hspace": 0.35, "wspace": 0.30})
    ax_ts1, ax_ts2 = axes[0, 0], axes[0, 1]
    ax_wv, ax_uv = axes[1, 0], axes[1, 1]

    for ax, sol, u_t, title in [
        (ax_ts1, sol_sub, u_sub, f"Sub perturbation (Dv=-{dv_small})"),
        (ax_ts2, sol_sup, u_sup, f"Super perturbation (Dv=-{dv_large})"),
    ]:
        t = sol.t
        w, v = sol.y
        ax.plot(t, u_t, color="tab:blue", lw=1.5, label="u_hat (PIP3, QSS)")
        ax.plot(t, w, color="tab:purple", lw=1.5, label="w (RASA1-like)")
        ax.plot(t, v, color="tab:orange", lw=1.5, label="v (PTEN)")
        ax.axhline(u_rest, color="tab:blue", ls=":", alpha=0.4, lw=1)
        ax.axhline(w_rest, color="tab:purple", ls=":", alpha=0.4, lw=1)
        ax.axhline(v_rest, color="tab:orange", ls=":", alpha=0.4, lw=1)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Level")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    y_min = float(min(np.nanmin(sol_sub.y), np.nanmin(sol_sup.y), np.nanmin(u_sub), np.nanmin(u_sup)))
    y_max = float(max(np.nanmax(sol_sub.y), np.nanmax(sol_sup.y), np.nanmax(u_sub), np.nanmax(u_sup)))
    pad = 0.05 * max(1e-6, (y_max - y_min))
    ax_ts1.set_ylim(y_min - pad, y_max + pad)
    ax_ts2.set_ylim(y_min - pad, y_max + pad)

    ax_wv.plot(sol_sub.y[0], sol_sub.y[1], color="tab:cyan", lw=2, label="sub")
    ax_wv.plot(sol_sup.y[0], sol_sup.y[1], color="tab:pink", lw=2, label="super")
    ax_wv.plot([w_rest], [v_rest], "ko", ms=6, label="rest")
    Fw_m = np.ma.masked_invalid(Fw)
    Fv_m = np.ma.masked_invalid(Fv)
    ax_wv.contour(W, V, Fw_m, levels=[0.0], colors=["tab:green"], linewidths=2.0)
    ax_wv.plot([], [], color="tab:green", lw=2.0, label="dw/dt = 0")
    ax_wv.contour(W, V, Fv_m, levels=[0.0], colors=["tab:red"], linewidths=2.0)
    ax_wv.plot([], [], color="tab:red", lw=2.0, label="dv/dt = 0")
    skip = (slice(None, None, 10), slice(None, None, 10))
    Sp = np.hypot(Fw[skip], Fv[skip])
    eps = 1e-12
    ax_wv.quiver(
        W[skip],
        V[skip],
        Fw[skip] / (Sp + eps),
        Fv[skip] / (Sp + eps),
        color="0.82",
        scale=25,
        width=0.003,
        zorder=0,
    )
    ax_wv.set_xlabel("w (RASA1-like)")
    ax_wv.set_ylabel("v (PTEN)")
    ax_wv.set_title("Phase plane: (w,v) with nullclines")
    ax_wv.set_xlim(0.0, 0.35)
    ax_wv.set_ylim(0.0, 0.35)
    ax_wv.grid(True, alpha=0.3)
    ax_wv.legend(fontsize=8, loc="upper right")

    ax_uv.plot(u_sub, sol_sub.y[1], color="tab:cyan", lw=2, label="sub")
    ax_uv.plot(u_sup, sol_sup.y[1], color="tab:pink", lw=2, label="super")
    ax_uv.plot([u_rest], [v_rest], "ko", ms=6, label="rest")
    ax_uv.set_xlabel("u_hat (PIP3, QSS)")
    ax_uv.set_ylabel("v (PTEN)")
    ax_uv.set_title("Projection: (u_hat, v)")
    ax_uv.grid(True, alpha=0.3)
    ax_uv.legend(fontsize=8, loc="upper right")

    fig.suptitle("Reduced model (swapped taus): fast PIP3 eliminated", fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.93)

    # ---- comparison plot: how dv/dt=0 moves when Vtot is changed ----
    fig_cmp, ax_cmp = plt.subplots(figsize=(6.8, 5.6))
    vtot_scales = [1.00, 0.85, 0.70]
    colors = ["tab:red", "tab:orange", "tab:brown"]

    for s, c in zip(vtot_scales, colors):
        p_i = dict(par)
        p_i["Vtot"] = float(par["Vtot"] * s)
        try:
            w_i, v_i = find_fixed_point(p_i, guess=(float(w_rest), float(v_rest)))
            u_i = u_hat(float(v_i), float(w_i), p_i)
        except Exception:
            # fallback if fixed-point solve is sensitive for this Vtot
            u_i = float(u_rest)
        W_i, V_i, _, Fv_i = compute_field_and_nullclines(p_i, u_ref=u_i)
        Fv_i_m = np.ma.masked_invalid(Fv_i)
        cs = ax_cmp.contour(W_i, V_i, Fv_i_m, levels=[0.0], colors=[c], linewidths=2.2)
        if len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
            ax_cmp.plot([], [], color=c, lw=2.2, label=f"dv/dt = 0 (Vtot x {s:.2f})")

    ax_cmp.set_xlabel("w (RASA1-like)")
    ax_cmp.set_ylabel("v (PTEN)")
    ax_cmp.set_title("Vtot sweep (swapped taus): dv/dt = 0 comparison")
    ax_cmp.set_xlim(0.0, 0.35)
    ax_cmp.set_ylim(0.0, 0.35)
    ax_cmp.grid(True, alpha=0.3)
    ax_cmp.legend(loc="upper right", fontsize=9)

    plt.show()

