# -*- coding: utf-8 -*-
"""
Static solution: quasi-static Newton-Raphson, u2 imposed by increasing
increments. At each increment we start from the previous solution (warm
start) and solve res_static(u1, u2) = 0.

Beyond the limit point of the snap-back there is no root of the residual
near the previous solution. With fine increments the NR cycles around the
local maximum of the residual and does not converge; with coarse increments
an iterate may land close to that maximum, where the tangent is almost
zero, and the resulting huge Newton step may throw it beyond the softening
zone, onto the fully damaged branch (u1 ~ u2) where the residual is a
straight line: the NR then "jumps" by itself and converges. No trick is
used (no restart on the other branch, no arc-length): if the NR fails, the
quasi-static calculation simply stops at that increment, with a message,
and the remaining entries of the output arrays are NaN.
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
        u1_sol   : (inc,) equilibrium solutions u1 (NaN after the limit point)
        F_sol    : (inc,) corresponding force F2 (NaN after the limit point)
        history  : list of (u2n, d_prev, hist_u1, hist_res) per increment up to
                   and including the failed one, useful to plot the residual
                   + the NR iterates
    """
    if u2_f is None:
        u2_f = mdl.u2_f

    u2_steps = np.linspace(0.0, u2_f, inc)
    u1_sol = np.full(inc, np.nan)
    F_sol = np.full(inc, np.nan)
    history = []

    u1_prev = 0.0
    d_prev = 0.0  # damage history variable (damage reached so far)
    for n, u2n in enumerate(u2_steps):
        residual = lambda u, u2n=u2n, d_prev=d_prev: mdl.res_static(u, u2n, d_prev)
        tangent = lambda u, d_prev=d_prev: mdl.tangent_static(u, d_prev)
        u1_n, ok, hist_x, hist_r = mdl.newton_raphson(residual, tangent, u1_prev)
        if not ok:
            # Beyond the limit point there is no root near the previous iterate:
            # the NR cycles around the local maximum of the residual. The
            # quasi-static displacement-controlled calculation cannot go
            # further; it stops here.
            history.append((u2n, d_prev, hist_x, hist_r))
            print(f"[Static] step {n} (u2={u2n:.4g} mm): NR did not converge after "
                  f"{len(hist_x) - 1} iterations, |R| = {abs(hist_r[-1]):.3g} N. "
                  f"No root near the previous solution (limit point reached): "
                  f"the quasi-static calculation stops here.")
            break
        u1_sol[n] = u1_n
        F_sol[n] = mdl.k2 * (u2n - u1_n)
        history.append((u2n, d_prev, hist_x, hist_r))
        u1_prev = u1_n
        d_prev = float(mdl.damage_history(u1_n, d_prev))

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
    u1_sweep = np.linspace(0.0, np.nanmax(u1_sol) * 1.2, 400)
    colors = plt.cm.viridis(np.linspace(0, 1, n_res_demo))
    idx_demo = np.linspace(0, len(history) - 1, n_res_demo).astype(int)

    plt.figure(figsize=(8, 6))
    for color, i in zip(colors, idx_demo):
        u2n, d_prev, hist_x, hist_r = history[i]
        plt.plot(u1_sweep, mdl.res_static(u1_sweep, u2n, d_prev), color=color, lw=1.2, alpha=0.8)
        plt.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    i_worst = int(np.argmax([len(h[2]) for h in history]))
    u2n, d_prev, hist_x, hist_r = history[i_worst]
    plt.plot(u1_sweep, mdl.res_static(u1_sweep, u2n, d_prev), color='red', lw=2.5,
             label=f'increment {i_worst} (u2 = {u2n:.4g} mm): {len(hist_x) - 1} iterations')
    plt.plot(hist_x, hist_r, 'o', color='red', ms=8, mec='k', mew=0.6)
    plt.legend(loc='lower right', fontsize=9)
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel('u1 [mm]')
    plt.ylabel('Static residual R(u1) [N]')
    plt.title('Static: residual(u1) per increment + NR iterates')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
