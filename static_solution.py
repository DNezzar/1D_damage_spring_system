# -*- coding: utf-8 -*-
"""
Static solution: quasi-static Newton-Raphson, u2 imposed by increasing
increments. At each increment we start from the previous solution (warm
start) and solve res_static(u1, u2) = 0.

Beyond the limit point of the snap-back there is no root of the residual
near the previous solution: the NR cycles (plain NR) or stalls at the local
maximum of the residual (NR with line search, parameters.line_search = True)
and does not converge. No trick is used (no restart on the other branch, no
arc-length): the quasi-static calculation simply stops at the first
increment where the NR fails, with a message. The remaining entries of the
output arrays are NaN.
"""

import numpy as np

import model as mdl
import parameters as prm


def solve(inc=prm.inc, u2_f=None, line_search=prm.line_search):
    """
    Solve the static problem over `inc` increments from 0 to u2_f
    (defaults from parameters.py). `line_search` switches the backtracking
    line search of the Newton-Raphson on or off.

        Returns:
        u2_steps : (inc,) imposed displacements
        u1_sol   : (inc,) equilibrium solutions u1 (NaN after the limit point)
        F_sol    : (inc,) corresponding force F2 (NaN after the limit point)
        history  : list of (u2n, kappa, hist_u1, hist_res) per increment up to
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
    kappa = 0.0  # damage history variable (max of u1 reached)
    for n, u2n in enumerate(u2_steps):
        residual = lambda u, u2n=u2n, kappa=kappa: mdl.res_static(u, u2n, kappa)
        tangent = lambda u, kappa=kappa: mdl.tangent_static(u, kappa)
        u1_n, ok, hist_x, hist_r = mdl.newton_raphson(residual, tangent, u1_prev,
                                                      line_search=line_search)
        if not ok:
            # Beyond the limit point there is no root near the previous iterate:
            # the NR cycles around the local maximum of the residual. The
            # quasi-static displacement-controlled calculation cannot go
            # further; it stops here.
            history.append((u2n, kappa, hist_x, hist_r))
            why = ("line search stalled: no step length decreases |R|"
                   if mdl.newton_raphson.last_stalled else "max_iter reached")
            print(f"[Static] step {n} (u2={u2n:.4g} mm): NR did not converge after "
                  f"{len(hist_x) - 1} iterations ({why}), |R| = {abs(hist_r[-1]):.3g} N. "
                  f"No root near the previous solution (limit point reached): "
                  f"the quasi-static calculation stops here.")
            break
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
    u1_sweep = np.linspace(0.0, np.nanmax(u1_sol) * 1.2, 400)
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
