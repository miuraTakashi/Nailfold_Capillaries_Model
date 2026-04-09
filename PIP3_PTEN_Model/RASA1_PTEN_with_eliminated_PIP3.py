"""
Reduced 2-variable model:
  PIP3 (u) = fast  -> eliminated by quasi-steady assumption du/dt = 0
  RASA1-like (w) = intermediate
  PTEN membrane (v) = slow

State variables:
  y = [w, v]

Dynamics:
  u_hat(v, w) solves du_dt(u, v, w) = 0 in u in [0, Ptot]
  dw/dt = (w_inf(u_hat) - w) / tau_w
  dv/dt = (dv_raw(u_hat, v)) / tau_v

Note:
  tau_v > tau_w gives "v slower than w".
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
    tau_w=0.3,   # intermediate (faster than before)
    w_max=1.0,
    Kw=0.25,
    hill_m=4.0,
    # --- PTEN slow scale ---
    tau_v=6.0,   # slow (larger separation from tau_w)
)


def A(u, p):
    alpha, KA, n = p["alpha"], p["KA"], p["hill_n"]
    return 1.0 + alpha * (u**n) / (KA**n + u**n)


def k_on0_func(u, p):
    k_on0_base, gamma, Ku_k_on0 = p["k_on0_base"], p["gamma"], p["Ku_k_on0"]
    return k_on0_base / (1.0 + gamma * u / (Ku_k_on0 + u))


def k_off_func(u, p):
    koff = p["koff"]
    delta_off = p.get("delta_off", 0.0)
    Ku_k_off = p.get("Ku_k_off", 1.0)
    if delta_off <= 0.0 or Ku_k_off <= 0.0:
        return koff
    return koff * (1.0 + delta_off * u / (Ku_k_off + u))


def inhibition_w(w, p):
    return 1.0 / (1.0 + p["beta_w"] * np.maximum(w, 0.0))


def w_inf(u, p):
    w_max, Kw, m = p["w_max"], p["Kw"], p["hill_m"]
    u = np.maximum(u, 0.0)
    return w_max * (u**m) / (Kw**m + u**m)


def du_dt(u, v, w, p):
    Ptot = p["Ptot"]
    p2 = Ptot - u
    vPI3K, KPI3K = p["vPI3K"], p["KPI3K"]
    vPTEN, KPTEN = p["vPTEN"], p["KPTEN"]
    k_leak = p["k_leak"]
    tau_u = p["tau_u"]
    prod = vPI3K * A(u, p) * p2 / (KPI3K + p2)
    deg = vPTEN * v * u / (KPTEN + u)
    return (1.0 / tau_u) * (prod * inhibition_w(w, p) - deg - k_leak * u)


def dv_raw(u, v, p):
    Ptot, Vtot = p["Ptot"], p["Vtot"]
    p2 = Ptot - u
    kon = p["kon"]
    k_on_total = k_on0_func(u, p) + kon * p2
    k_off_u = k_off_func(u, p)
    return k_on_total * (Vtot - v) - k_off_u * v


def _bisect_root(fun, a, b, max_iter=80, tol=1e-10):
    fa = fun(a)
    fb = fun(b)
    if not np.isfinite(fa) or not np.isfinite(fb):
        raise ValueError("Non-finite bracket values")
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("No sign change in bracket")
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = fun(mid)
        if not np.isfinite(fmid):
            break
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def u_hat(v, w, p, u_guess=None, root_tol=1e-4):
    Ptot = float(p["Ptot"])

    def f(u):
        return du_dt(u, v, w, p)

    us = np.linspace(0.0, Ptot, 240)
    fs = np.array([f(float(ui)) for ui in us])
    finite = np.isfinite(fs)
    us = us[finite]
    fs = fs[finite]
    if len(us) < 2:
        raise RuntimeError("u_hat: insufficient finite samples")

    idx0 = np.where(fs == 0.0)[0]
    if len(idx0) > 0:
        return float(us[int(idx0[0])])

    s = np.sign(fs)
    flips = np.where(s[:-1] * s[1:] < 0)[0]
    if len(flips) == 0:
        if u_guess is None:
            return float(us[int(np.argmin(np.abs(fs)))])
        return float(np.clip(u_guess, 0.0, Ptot))

    roots = []
    for k in flips:
        a = float(us[k])
        b = float(us[k + 1])
        u0 = float(_bisect_root(f, a, b))
        if abs(f(u0)) < root_tol:
            du = max(1e-6, 1e-6 * Ptot)
            uL = max(0.0, u0 - du)
            uR = min(Ptot, u0 + du)
            dfdu = (f(uR) - f(uL)) / (uR - uL) if uR > uL else np.nan
            stable = np.isfinite(dfdu) and (dfdu < 0.0)
            roots.append((u0, stable))

    if not roots:
        return float(us[int(np.argmin(np.abs(fs)))])

    stable_roots = [r for r in roots if r[1]]
    candidates = stable_roots if stable_roots else roots
    if u_guess is None:
        return float(sorted(candidates, key=lambda t: t[0])[0][0])
    return float(sorted(candidates, key=lambda t: abs(t[0] - u_guess))[0][0])


def ode_system(t, y, p):
    w, v = float(y[0]), float(y[1])
    u = u_hat(v, w, p)
    if not np.isfinite(u):
        return [float("nan"), float("nan")]
    dw = (float(w_inf(u, p)) - w) / float(p["tau_w"])
    dv = dv_raw(u, v, p) / float(p["tau_v"])
    return [dw, dv]


def find_fixed_point(p, guess=(0.05, 0.25)):
    def residual(y):
        w, v = float(y[0]), float(y[1])
        u = u_hat(v, w, p)
        if not np.isfinite(u):
            return [1e6, 1e6]
        return [
            (float(w_inf(u, p)) - w) / float(p["tau_w"]),
            dv_raw(u, v, p) / float(p["tau_v"]),
        ]

    sol, info, ier, msg = fsolve(residual, guess, full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(f"fsolve failed: {msg}")
    return sol


def simulate(y0, p, t_span=(0.0, 10.0), n_eval=4000):
    t_eval = np.linspace(*t_span, n_eval)
    return solve_ivp(
        lambda t, y: ode_system(t, y, p),
        t_span,
        list(y0),
        t_eval=t_eval,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )


def compute_field_and_nullclines(
    p,
    *,
    u_ref,
    w_min=0.0,
    w_max=0.35,
    v_min=0.0,
    v_max=0.35,
    n=180,
):
    w_grid = np.linspace(w_min, w_max, n)
    v_grid = np.linspace(v_min, v_max, n)
    W, V = np.meshgrid(w_grid, v_grid, indexing="xy")
    Fw = np.empty_like(W, dtype=float)
    Fv = np.empty_like(W, dtype=float)

    # Use a fixed reference branch (closest to u_ref) at every grid point.
    # This avoids row/scan-order artifacts that can break contours.
    for j in range(W.shape[0]):
        for i in range(W.shape[1]):
            w = float(W[j, i])
            v = float(V[j, i])
            u = u_hat(v, w, p, u_guess=float(u_ref))
            if np.isfinite(u):
                Fw[j, i] = (float(w_inf(u, p)) - w) / float(p["tau_w"])
                Fv[j, i] = dv_raw(u, v, p) / float(p["tau_v"])
            else:
                Fw[j, i] = float("nan")
                Fv[j, i] = float("nan")
    return W, V, Fw, Fv


def v_nullcline_w_of_v(
    p,
    *,
    v_values,
    w_min=0.0,
    w_max=0.35,
    n_w_scan=220,
    choose="low_w",
):
    """
    Numerically represent dv/dt=0 as w=f(v) on a chosen branch.
    For each v, solve dv_raw(u_hat(v,w), v)/tau_v = 0 in w in [w_min, w_max].

    choose:
      - "low_w": choose smallest-w root (default)
      - "high_w": choose largest-w root
    """
    w_scan = np.linspace(w_min, w_max, n_w_scan)
    v_out = []
    w_out = []

    for v in v_values:
        v = float(v)

        def psi(w):
            u = u_hat(v, float(w), p)
            if not np.isfinite(u):
                return np.nan
            return dv_raw(u, v, p) / float(p["tau_v"])

        vals = np.array([psi(w) for w in w_scan], dtype=float)
        finite = np.isfinite(vals)
        roots = []
        for k in range(len(w_scan) - 1):
            if not (finite[k] and finite[k + 1]):
                continue
            y0 = vals[k]
            y1 = vals[k + 1]
            if y0 == 0.0:
                roots.append(float(w_scan[k]))
                continue
            if y0 * y1 < 0.0:
                a = float(w_scan[k])
                b = float(w_scan[k + 1])
                roots.append(float(_bisect_root(psi, a, b, tol=1e-10)))

        if not roots:
            continue
        w_sel = min(roots) if choose == "low_w" else max(roots)
        v_out.append(v)
        w_out.append(float(w_sel))

    return np.array(v_out), np.array(w_out)


if __name__ == "__main__":
    w_rest, v_rest = find_fixed_point(par, guess=(0.05, 0.25))
    u_rest = u_hat(float(v_rest), float(w_rest), par)
    print(
        "rest (u eliminated): "
        f"u_hat={u_rest:.6f}, w*={w_rest:.6f}, v*={v_rest:.6f}, tau_v={par['tau_v']:.3f}"
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
    v_line = np.linspace(0.0, 0.35, 260)
    v_nv, w_nv = v_nullcline_w_of_v(par, v_values=v_line, w_min=0.0, w_max=0.35, choose="low_w")

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
        ax.plot(t, v, color="tab:orange", lw=1.5, label="v (PTEN, slow)")
        ax.axhline(u_rest, color="tab:blue", ls=":", alpha=0.4, lw=1)
        ax.axhline(w_rest, color="tab:purple", ls=":", alpha=0.4, lw=1)
        ax.axhline(v_rest, color="tab:orange", ls=":", alpha=0.4, lw=1)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Level")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    # Match y-scale between sub/super time-series panels for fair comparison
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
    if len(v_nv) > 1:
        # explicit-style representation of dv/dt=0 as w=f(v) on selected branch
        ax_wv.plot(w_nv, v_nv, color="tab:red", lw=2.8, ls="--", label="dv/dt = 0 as w=f(v)")
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
    ax_wv.set_ylabel("v (PTEN, slow)")
    ax_wv.set_title("Phase plane: (w,v) with nullclines")
    ax_wv.set_xlim(0.0, 0.35)
    ax_wv.set_ylim(0.0, 0.35)
    ax_wv.grid(True, alpha=0.3)
    ax_wv.legend(fontsize=8, loc="upper right")

    ax_uv.plot(u_sub, sol_sub.y[1], color="tab:cyan", lw=2, label="sub")
    ax_uv.plot(u_sup, sol_sup.y[1], color="tab:pink", lw=2, label="super")
    ax_uv.plot([u_rest], [v_rest], "ko", ms=6, label="rest")
    ax_uv.set_xlabel("u_hat (PIP3, QSS)")
    ax_uv.set_ylabel("v (PTEN, slow)")
    ax_uv.set_title("Projection: (u_hat, v)")
    ax_uv.grid(True, alpha=0.3)
    ax_uv.legend(fontsize=8, loc="upper right")

    fig.suptitle("Reduced model: fast PIP3 eliminated, intermediate w and slow PTEN", fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.93)
    plt.show()

