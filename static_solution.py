# -*- coding: utf-8 -*-
"""
Static solution: quasi-static Newton-Raphson, u2 imposed by increasing
increments. At each increment we start from the previous solution (warm
start) and solve res_static(u1, u2) = 0.

Since the residual can have several roots (unstable branch of the
snap-back), the displacement-controlled NR follows the nearest stable
branch and "jumps" over the unstable zone: the resulting F-u2 curve thus
differs from the analytical solution around the peak.
"""

import numpy as np

import model as mdl
import parameters as prm


def solve(inc=prm.inc, u2_f=None):
    """
    Solve the static problem over `inc` increments from 0 to u2_f
    (defaults from parameters.py).

    Returns:
        u2_steps : (inc,) imposed displacements
        u1_sol   : (inc,) equilibrium solutions u1
        F_sol    : (inc,) corresponding force F2
        history  : list of (u2n, kappa, hist_u1, hist_res) per increment,
                   useful to plot the residual + the NR iterates
    """
    if u2_f is None:
        u2_f = mdl.u2_f

    u2_steps = np.linspace(0.0, u2_f, inc)
    u1_sol = np.empty(inc)
    F_sol = np.empty(inc)
    history = []

    u1_prev = 0.0
    kappa = 0.0  # damage history variable (max of u1 reached)
    for n, u2n in enumerate(u2_steps):
        residual = lambda u, u2n=u2n, kappa=kappa: mdl.res_static(u, u2n, kappa)
        tangent = lambda u, kappa=kappa: mdl.tangent_static(u, kappa)
        u1_n, ok, hist_x, hist_r = mdl.newton_raphson(residual, tangent, u1_prev)
        if not ok:
            # Beyond the limit point there is no root near the previous iterate:
            # the NR cycles. Restart from the fully damaged branch (u1 ~ u2),
            # which performs the snap to the other stable branch.
            u1_n, ok, hist_x2, hist_r2 = mdl.newton_raphson(residual, tangent, u2n)
            hist_x = np.concatenate([hist_x, hist_x2])
            hist_r = np.concatenate([hist_r, hist_r2])
            print(f"[Static] step {n} (u2={u2n:.3f} mm): branch jump, "
                  f"NR restarted from u1=u2 -> {'converged' if ok else 'FAILED'}")
        u1_sol[n] = u1_n
        F_sol[n] = mdl.k2 * (u2n - u1_n)
        history.append((u2n, kappa, hist_x, hist_r))
        u1_prev = u1_n
        kappa = max(kappa, u1_n)

    return u2_steps, u1_sol, F_sol, history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from analytical_solution import analytical_curve

    u2_steps, u1_sol, F_sol, history = solve()
    _, u2_ana, F_ana = analytical_curve()

    # -- Fig 1: static F-u2 vs analytical
    plt.figure(figsize=(8, 5))
    plt.plot(u2_ana, F_ana, 'r--', lw=2, label='Analytical (with snap-back)')
    plt.plot(u2_steps, F_sol, 'b-', lw=1.5, label='Static (Newton-Raphson)')
    plt.xlabel('u2 [mm]')
    plt.ylabel('Force F2 [N]')
    plt.title('Static vs analytical solution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 2: residual R(u1) per increment + NR iterates
    n_res_demo = 15
    u1_sweep = np.linspace(0.0, u1_sol.max() * 1.2, 400)
    colors = plt.cm.viridis(np.linspace(0, 1, n_res_demo))
    idx_demo = np.linspace(0, len(history) - 1, n_res_demo).astype(int)

    plt.figure(figsize=(8, 6))
    for color, i in zip(colors, idx_demo):
        u2n, kappa, hist_x, hist_r = history[i]
        plt.plot(u1_sweep, mdl.res_static(u1_sweep, u2n, kappa), color=color, lw=1.2, alpha=0.8)
        plt.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel('u1 [mm]')
    plt.ylabel('Static residual R(u1) [N]')
    plt.title('Static: residual(u1) per increment + NR iterates')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
