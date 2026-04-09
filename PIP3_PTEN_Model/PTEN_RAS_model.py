"""
PTEN–RAS model obtained by eliminating fast PIP3 (u)

Starting point:
  PIP3–PTEN + RasGAP(w) model in dynamics_rasgap.py with variables
    u = PIP3 (fast)
    v = membrane PTEN
    w = RasGAP-like slow negative feedback (RAS module variable)

Fast elimination (quasi-steady-state):
  Assume u relaxes much faster than v,w, and set
      du/dt(u, v, w) = 0
  to obtain u = u_hat(v, w).

Reduced 2D system:
    dv/dt = dv_dt(u_hat(v,w), v)
    dw/dt = (w_inf(u_hat(v,w)) - w)/tau_w

Notes:
  - This reduction is only valid when u is fast and has a unique stable
    quasi-steady branch for the (v,w) region explored.
  - u_hat is found by a bracketed root search in u ∈ [0, Ptot].
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


par = dict(
    # --- original PIP3–PTEN parameters ---
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
    # --- RasGAP-like feedback (w) ---
    beta_w=3.0,
    tau_w=0.5,
    w_max=1.0,
    Kw=0.25,
    hill_m=4.0,
)


def A(u: float | np.ndarray, par: dict) -> float | np.ndarray:
    alpha, KA, n = par["alpha"], par["KA"], par["hill_n"]
    return 1.0 + alpha * (u**n) / (KA**n + u**n)


def k_on0_func(u: float | np.ndarray, par: dict) -> float | np.ndarray:
    k_on0_base, gamma, Ku_k_on0 = par["k_on0_base"], par["gamma"], par["Ku_k_on0"]
    return k_on0_base / (1.0 + gamma * u / (Ku_k_on0 + u))


def k_off_func(u: float | np.ndarray, par: dict) -> float | np.ndarray:
    koff = par["koff"]
    delta_off = par.get("delta_off", 0.0)
    Ku_k_off = par.get("Ku_k_off", 1.0)
    if delta_off <= 0.0 or Ku_k_off <= 0.0:
        return koff
    return koff * (1.0 + delta_off * u / (Ku_k_off + u))


def inhibition_w(w: float | np.ndarray, par: dict) -> float | np.ndarray:
    return 1.0 / (1.0 + par["beta_w"] * np.maximum(w, 0.0))


def w_inf(u: float | np.ndarray, par: dict) -> float | np.ndarray:
    w_max, Kw, m = par["w_max"], par["Kw"], par["hill_m"]
    u = np.maximum(u, 0.0)
    return w_max * (u**m) / (Kw**m + u**m)


def du_dt(u: float, v: float, w: float, par: dict) -> float:
    Ptot = par["Ptot"]
    p2 = Ptot - u
    vPI3K, KPI3K = par["vPI3K"], par["KPI3K"]
    vPTEN, KPTEN = par["vPTEN"], par["KPTEN"]
    k_leak = par["k_leak"]
    tau_u = par["tau_u"]
    prod = vPI3K * float(A(u, par)) * p2 / (KPI3K + p2)
    deg = vPTEN * v * u / (KPTEN + u)
    return (1.0 / tau_u) * (prod * float(inhibition_w(w, par)) - deg - k_leak * u)


def dv_dt(u: float, v: float, par: dict) -> float:
    Ptot, Vtot = par["Ptot"], par["Vtot"]
    p2 = Ptot - u
    kon = par["kon"]
    k_on_total = float(k_on0_func(u, par)) + kon * p2
    k_off_u = float(k_off_func(u, par))
    return k_on_total * (Vtot - v) - k_off_u * v


def _bisect_root(fun, a: float, b: float, *, max_iter: int = 80, tol: float = 1e-10) -> float:
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


def u_hat(
    v: float,
    w: float,
    par: dict,
    *,
    u_guess: float | None = None,
    root_tol: float = 1e-4,
) -> float:
    """
    Solve du_dt(u, v, w)=0 for u in [0, Ptot].
    Uses a coarse scan + bisection to be robust.
    """
    Ptot = float(par["Ptot"])

    def f(u):
        return du_dt(u, v, w, par)

    # coarse scan for sign changes (may have multiple roots)
    n_scan = 240
    us = np.linspace(0.0, Ptot, n_scan)
    fs = np.array([f(float(ui)) for ui in us])

    finite = np.isfinite(fs)
    us = us[finite]
    fs = fs[finite]
    if len(us) < 2:
        raise RuntimeError("u_hat: insufficient finite samples")

    # if an exact root sampled
    idx0 = np.where(fs == 0.0)[0]
    if len(idx0) > 0:
        return float(us[int(idx0[0])])

    s = np.sign(fs)
    flips = np.where(s[:-1] * s[1:] < 0)[0]
    if len(flips) == 0:
        # This should be rare for this model (typically f(0)>0 and f(Ptot)<0).
        # Keep a fallback for robustness.
        if u_guess is None:
            u_guess = float(us[int(np.argmin(np.abs(fs)))])

        def g(u_arr):
            u = float(np.clip(u_arr[0], 0.0, Ptot))
            return [f(u)]

        sol, info, ier, msg = fsolve(g, [u_guess], full_output=True, xtol=1e-12)
        u_sol = float(np.clip(sol[0], 0.0, Ptot))
        return u_sol

    roots = []
    # find all roots by bisection on each sign-change interval
    for k in flips:
        a = float(us[k])
        b = float(us[k + 1])
        u0 = float(_bisect_root(f, a, b))
        if abs(f(u0)) < root_tol:
            # stability of fast u dynamics: df/du < 0 means stable
            du = max(1e-6, 1e-6 * Ptot)
            uL = max(0.0, u0 - du)
            uR = min(Ptot, u0 + du)
            dfdu = (f(uR) - f(uL)) / (uR - uL) if uR > uL else np.nan
            stable = np.isfinite(dfdu) and (dfdu < 0.0)
            roots.append((u0, stable, dfdu))

    if not roots:
        # fallback: pick minimal |f| sample
        return float(us[int(np.argmin(np.abs(fs)))])

    stable_roots = [r for r in roots if r[1]]
    candidates = stable_roots if stable_roots else roots

    if u_guess is None:
        # choose smallest u by default (low-PIP3 branch)
        return float(sorted(candidates, key=lambda t: t[0])[0][0])

    # choose closest to u_guess
    return float(sorted(candidates, key=lambda t: abs(t[0] - u_guess))[0][0])


def ode_system(t: float, y: np.ndarray, par: dict) -> list[float]:
    v, w = float(y[0]), float(y[1])
    u = u_hat(v, w, par)
    if not np.isfinite(u):
        return [float("nan"), float("nan")]
    return [dv_dt(u, v, par), (float(w_inf(u, par)) - w) / float(par["tau_w"])]


def find_fixed_point(par: dict, guess: tuple[float, float] = (0.2, 0.05)) -> np.ndarray:
    def residual(y):
        v, w = float(y[0]), float(y[1])
        u = u_hat(v, w, par)
        if not np.isfinite(u):
            return [1e6, 1e6]
        return [
            dv_dt(u, v, par),
            (float(w_inf(u, par)) - w) / float(par["tau_w"]),
        ]

    sol, info, ier, msg = fsolve(residual, guess, full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(f"fsolve failed: {msg}")
    return sol


def simulate(y0: tuple[float, float], par: dict, t_span=(0.0, 20.0), n_eval: int = 5000):
    t_eval = np.linspace(*t_span, n_eval)
    return solve_ivp(
        lambda t, y: ode_system(t, y, par),
        t_span,
        list(y0),
        t_eval=t_eval,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )


def compute_nullclines_vw(
    par: dict,
    *,
    v_grid: np.ndarray | None = None,
    w_grid: np.ndarray | None = None,
    n_v: int = 220,
    n_w: int = 220,
    u_guess_init: float | None = None,
    warm_start: bool = False,
):
    """
    Compute nullclines in (v,w) plane:
      - dv/dt = 0  <=>  F(v,w) = dv_dt(u_hat(v,w), v) = 0
      - dw/dt = 0  <=>  w = w_inf(u_hat(v,w))

    Returns:
      V, W mesh
      F = dv/dt on mesh
      G = dw/dt on mesh
      U = u_hat(v,w) on mesh (for inspection)
    """
    Vtot = float(par["Vtot"])
    w_max = float(par["w_max"])

    if v_grid is None:
        v_grid = np.linspace(0.0, Vtot, n_v)
    if w_grid is None:
        w_grid = np.linspace(0.0, w_max, n_w)

    V, W = np.meshgrid(v_grid, w_grid, indexing="xy")
    F = np.empty_like(V, dtype=float)
    G = np.empty_like(V, dtype=float)
    U = np.empty_like(V, dtype=float)

    # NOTE:
    # u_hat(v,w) can have multiple roots; for nullclines we must use a
    # consistent branch selection rule. By default we DO NOT warm-start,
    # so u_hat is evaluated with its internal deterministic rule.
    u_guess = u_guess_init if warm_start else None
    for j in range(V.shape[0]):
        u_guess_row = u_guess
        for i in range(V.shape[1]):
            v = float(V[j, i])
            w = float(W[j, i])
            u = u_hat(v, w, par, u_guess=u_guess_row if warm_start else None)
            U[j, i] = u
            if np.isfinite(u):
                F[j, i] = dv_dt(u, v, par)
                G[j, i] = (float(w_inf(u, par)) - w) / float(par["tau_w"])
                if warm_start:
                    u_guess_row = u
            else:
                F[j, i] = float("nan")
                G[j, i] = float("nan")
        if warm_start:
            u_guess = u_guess_row

    return V, W, F, G, U


def _scan_brackets_1d(x: np.ndarray, y: np.ndarray):
    """Return list of index k such that y[k]*y[k+1] < 0 with finite values."""
    finite = np.isfinite(y)
    idx = []
    for k in range(len(x) - 1):
        if not (finite[k] and finite[k + 1]):
            continue
        y0 = y[k]
        y1 = y[k + 1]
        if y0 == 0.0:
            idx.append(k)
        elif y0 * y1 < 0:
            idx.append(k)
    return idx


def nullcline_dw_dt(par: dict, *, v_values: np.ndarray, n_u_scan: int = 600):
    """
    Compute dw/dt=0 nullcline in (v,w) using u as mediator.

    dw/dt=0  =>  w = w_inf(u)
    Together with the fast condition du/dt=0 along this nullcline:
        du_dt(u, v, w_inf(u)) = 0

    For each v, solve the scalar equation in u ∈ [0, Ptot], then set w=w_inf(u).
    This avoids discontinuities from u_hat(v,w) branch switching.
    """
    Ptot = float(par["Ptot"])
    u_scan = np.linspace(0.0, Ptot, n_u_scan)

    v_out = []
    w_out = []
    u_out = []

    for v in v_values:
        v = float(v)

        def phi(u):
            w = float(w_inf(float(u), par))
            return du_dt(float(u), v, w, par)

        vals = np.array([phi(u) for u in u_scan], dtype=float)
        brackets = _scan_brackets_1d(u_scan, vals)
        if not brackets:
            continue

        # choose smallest-u root (low-PIP3 branch)
        k = brackets[0]
        a = float(u_scan[k])
        b = float(u_scan[min(k + 1, len(u_scan) - 1)])
        if phi(a) == 0.0:
            u_root = a
        else:
            u_root = float(_bisect_root(phi, a, b, tol=1e-10))

        w_root = float(w_inf(u_root, par))
        v_out.append(v)
        w_out.append(w_root)
        u_out.append(float(u_root))

    return np.array(v_out), np.array(w_out), np.array(u_out)


def _v_null_from_dv0(u: float, par: dict) -> float:
    """dv/dt=0 solved for v as a function of u."""
    Ptot, Vtot = float(par["Ptot"]), float(par["Vtot"])
    p2 = Ptot - float(u)
    kon = float(par["kon"])
    k_on_total = float(k_on0_func(float(u), par)) + kon * p2
    k_off_u = float(k_off_func(float(u), par))
    return Vtot * k_on_total / (k_off_u + k_on_total)


def nullcline_dv_dt(par: dict, *, n_u: int = 800):
    """
    Compute dv/dt=0 nullcline in (v,w) using u as mediator.

    dv/dt=0 gives v = v_null(u).
    Along the fast manifold du/dt=0, solve for w explicitly from
        prod(u) * inhibition_w(w) = deg(u, v_null(u)) + k_leak*u
    with inhibition_w(w)=1/(1+beta_w*w).
    """
    Ptot = float(par["Ptot"])
    beta = float(par["beta_w"])
    w_max = float(par["w_max"])

    u_vals = np.linspace(0.0, Ptot, n_u)
    v_out = []
    w_out = []
    u_out = []

    for u in u_vals:
        u = float(u)
        v = float(_v_null_from_dv0(u, par))

        p2 = Ptot - u
        vPI3K, KPI3K = float(par["vPI3K"]), float(par["KPI3K"])
        vPTEN, KPTEN = float(par["vPTEN"]), float(par["KPTEN"])
        k_leak = float(par["k_leak"])

        prod = vPI3K * float(A(u, par)) * p2 / (KPI3K + p2)
        deg = vPTEN * v * u / (KPTEN + u)
        rhs = deg + k_leak * u

        if prod <= 0.0 or rhs <= 0.0:
            continue

        inhib = rhs / prod  # should be in (0,1]
        if inhib <= 0.0:
            continue
        if inhib > 1.0:
            # would require negative w; skip
            continue

        w = (1.0 / inhib - 1.0) / beta
        if 0.0 <= w <= w_max:
            v_out.append(v)
            w_out.append(float(w))
            u_out.append(float(u))

    return np.array(v_out), np.array(w_out), np.array(u_out)


if __name__ == "__main__":
    v_rest, w_rest = find_fixed_point(par, guess=(0.2, 0.05))
    u_rest = u_hat(float(v_rest), float(w_rest), par)
    print(f"rest (reduced): u_hat={u_rest:.6f}, v*={v_rest:.6f}, w*={w_rest:.6f}")

    # ICs: perturb v in the negative direction (PTEN depletion stimulus)
    dv_small = 0.01
    dv_large = 0.02
    y0_sub = (max(0.0, float(v_rest) - dv_small), float(w_rest))
    y0_sup = (max(0.0, float(v_rest) - dv_large), float(w_rest))

    t_span = (0.0, 2.0)
    sol_sub = simulate(y0_sub, par, t_span=t_span)
    sol_sup = simulate(y0_sup, par, t_span=t_span)

    # reconstruct u_hat(t) for plotting
    def u_series(sol):
        vs, ws = sol.y
        return np.array([u_hat(float(v), float(w), par) for v, w in zip(vs, ws)])

    u_sub = u_series(sol_sub)
    u_sup = u_series(sol_sup)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.35, wspace=0.30)
    ax_ts1 = fig.add_subplot(gs[0, 0])
    ax_ts2 = fig.add_subplot(gs[0, 1], sharey=ax_ts1)
    ax_vw = fig.add_subplot(gs[1, 0])
    ax_uw = fig.add_subplot(gs[1, 1])

    for ax, sol, u_t, title in [
        (ax_ts1, sol_sub, u_sub, f"Sub perturbation (Δv=-{dv_small})"),
        (ax_ts2, sol_sup, u_sup, f"Super perturbation (Δv=-{dv_large})"),
    ]:
        t = sol.t
        v, w = sol.y
        ax.plot(t, u_t, color="tab:blue", lw=1.5, label="u_hat (PIP3, QSS)")
        ax.plot(t, v, color="tab:orange", lw=1.5, label="v (membrane PTEN)")
        ax.plot(t, w, color="tab:purple", lw=1.5, label="w (RasGAP)")
        ax.axhline(u_rest, color="tab:blue", ls=":", alpha=0.4, lw=1)
        ax.axhline(v_rest, color="tab:orange", ls=":", alpha=0.4, lw=1)
        ax.axhline(w_rest, color="tab:purple", ls=":", alpha=0.4, lw=1)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Level")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    ax_vw.plot(sol_sub.y[0], sol_sub.y[1], color="tab:cyan", lw=2, label="sub")
    ax_vw.plot(sol_sup.y[0], sol_sup.y[1], color="tab:pink", lw=2, label="super")
    ax_vw.plot([v_rest], [w_rest], "ko", ms=6, label="rest")
    ax_vw.set_xlabel("v (membrane PTEN)")
    ax_vw.set_ylabel("w (RasGAP)")
    ax_vw.set_title("Phase plane: (v,w) with nullclines")
    ax_vw.grid(True, alpha=0.3)
    ax_vw.legend(fontsize=8, loc="upper right")

    # ---- nullclines in (v,w) ----
    # "Original policy": compute dv/dt, dw/dt on a 2D grid and draw zero-level contours.
    # NOTE: if u_hat(v,w) has multiple roots, the chosen branch (here: u_hat's internal rule)
    # can introduce sharp transitions; a sufficiently fine grid helps.
    v_min, v_max = 0.0, 0.3
    w_min, w_max = 0.0, 0.3
    v_grid = np.linspace(v_min, v_max, 256)
    w_grid = np.linspace(w_min, w_max, 256)
    V, W, F, G, U = compute_nullclines_vw(par, v_grid=v_grid, w_grid=w_grid, warm_start=False)
    ax_vw.contour(V, W, F, levels=[0.0], colors=["tab:red"], linewidths=2.2, linestyles="-")
    ax_vw.plot([], [], color="tab:red", lw=2.2, label="dv/dt = 0")
    ax_vw.contour(V, W, G, levels=[0.0], colors=["tab:green"], linewidths=2.2, linestyles="-")
    ax_vw.plot([], [], color="tab:green", lw=2.2, label="dw/dt = 0")

    # vector field (normalized)
    skip = (slice(None, None, 12), slice(None, None, 12))
    F_s = F[skip]
    G_s = G[skip]
    speed = np.hypot(F_s, G_s)
    eps = 1e-12
    ax_vw.quiver(
        V[skip],
        W[skip],
        F_s / (speed + eps),
        G_s / (speed + eps),
        color="0.85",
        scale=25,
        width=0.003,
        zorder=0,
    )

    ax_vw.set_xlim(v_min, v_max)
    ax_vw.set_ylim(w_min, w_max)

    # redraw legend with the added handles
    ax_vw.legend(fontsize=8, loc="upper right")

    ax_uw.plot(u_sub, sol_sub.y[1], color="tab:cyan", lw=2, label="sub")
    ax_uw.plot(u_sup, sol_sup.y[1], color="tab:pink", lw=2, label="super")
    ax_uw.plot([u_rest], [w_rest], "ko", ms=6, label="rest")
    ax_uw.set_xlabel("u_hat (PIP3, QSS)")
    ax_uw.set_ylabel("w (RasGAP)")
    ax_uw.set_title("Projection: (u_hat,w)")
    ax_uw.grid(True, alpha=0.3)
    ax_uw.legend(fontsize=8, loc="upper right")

    fig.suptitle("PTEN–RAS (RasGAP) reduced model by eliminating fast PIP3", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92)
    plt.show()

