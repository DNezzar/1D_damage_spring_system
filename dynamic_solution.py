# -*- coding: utf-8 -*-
"""
Dynamic solution: implicit Newmark scheme (predictor-corrector; with
beta=0.25, gamma=0.5 -> average acceleration, unconditionally stable, no
numerical damping) with Newton-Raphson at each time step on node 1
(mass m1). The actual beta, gamma values are set in parameters.py.

The dynamic residual is simply the static residual (model.res_static)
augmented by the inertia term m1*a(u1), where a(u1) is the acceleration at
step n+1 expressed through the Newmark predictors:

    u_tilde, v_tilde = predictors from (u_n, v_n, a_n)
    a(u1) = c1 * (u1 - u_tilde)          with c1 = 1/(beta*dt^2)
    v(u1) = v_tilde + gamma*dt*a(u1)

The term m1*c1 is added to the static tangent: it is what "regularises"
the tangent in the zone where spring 1 softens (snap-back), and what
changes the shape of the residual R(u1) compared to the static case.
"""

import numpy as np

import model as mdl
import parameters as prm

beta_nm, gamma_nm = prm.beta_nm, prm.gamma_nm


def predictors(u_n, v_n, a_n, dt):
    u_tilde = u_n + dt * v_n + dt**2 / 2.0 * (1.0 - 2.0 * beta_nm) * a_n
    v_tilde = v_n + (1.0 - gamma_nm) * dt * a_n
    return u_tilde, v_tilde


def res_dynamic(u1, u2, u_tilde, c1, kappa=0.0):
    a = c1 * (u1 - u_tilde)
    return mdl.m1 * a + mdl.res_static(u1, u2, kappa)


def tangent_dynamic(u1, c1, kappa=0.0):
    return mdl.m1 * c1 + mdl.tangent_static(u1, kappa)


def solve(dt=prm.dt, n_steps=prm.n_steps, n_hold=prm.n_hold, u2_f=prm.u2_f):
    """
    Integrate the dynamic response of node 1 during a ramp of imposed
    displacement u2(t), from 0 to u2_f over n_steps steps, followed by a
    hold phase u2 = u2_f during n_hold steps (to let the snap complete and
    observe the free oscillations of the mass m1).

    Defaults come from parameters.py: dt = T0/N_per_period (the
    inertia/stiffness ratio in the tangent is N^2/pi^2 with beta=1/4),
    n_steps from the ramp rate v2, n_hold from the hold duration t_hold.

    Returns:
        t, u2_t, u1_sol, v1_sol, a1_sol, F_sol : time histories
        history : list of (u2_np1, u_tilde, c1, kappa, hist_u1, hist_res)
                  per time step, to plot the residual + the NR iterates
    """
    c1 = 1.0 / (beta_nm * dt**2)
    v2_rate = u2_f / (n_steps * dt)
    n_tot = n_steps + n_hold

    t = np.arange(n_tot + 1) * dt
    u1_sol = np.zeros(n_tot + 1)
    v1_sol = np.zeros(n_tot + 1)
    a1_sol = np.zeros(n_tot + 1)
    u2_t = np.zeros(n_tot + 1)
    F_sol = np.zeros(n_tot + 1)
    history = []
    kappa = 0.0  # damage history variable (max of u1 reached)

    for n in range(n_tot):
        u2_np1 = min((n + 1) * dt * v2_rate, u2_f)
        u_tilde, v_tilde = predictors(u1_sol[n], v1_sol[n], a1_sol[n], dt)

        u1_np1, ok, hist_x, hist_r = mdl.newton_raphson(
            lambda u, u2_np1=u2_np1, u_tilde=u_tilde, kappa=kappa:
                res_dynamic(u, u2_np1, u_tilde, c1, kappa),
            lambda u, kappa=kappa: tangent_dynamic(u, c1, kappa),
            u_tilde,
        )
        if not ok:
            print(f"[Dynamic] step {n} (t={t[n+1]*1e3:.1f} ms): NR not converged, "
                  f"residual={hist_r[-1]:.3e} N")

        a1_np1 = c1 * (u1_np1 - u_tilde)
        v1_np1 = v_tilde + gamma_nm * dt * a1_np1

        u2_t[n + 1] = u2_np1
        u1_sol[n + 1] = u1_np1
        a1_sol[n + 1] = a1_np1
        v1_sol[n + 1] = v1_np1
        F_sol[n + 1] = mdl.k2 * (u2_np1 - u1_np1)

        history.append((u2_np1, u_tilde, c1, kappa, hist_x, hist_r))
        kappa = max(kappa, u1_np1)

    return t, u2_t, u1_sol, v1_sol, a1_sol, F_sol, history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from analytical_solution import analytical_curve

    t, u2_t, u1_sol, v1_sol, a1_sol, F_sol, history = solve()
    _, u2_ana, F_ana = analytical_curve()

    print(f"{len(history)} time steps, dt = {t[1]*1e3:.3f} ms")

    # -- Fig 1: dynamic F-u2 vs analytical
    plt.figure(figsize=(8, 5))
    plt.plot(u2_ana, F_ana, 'r--', lw=2, label='Analytical (with snap-back)')
    plt.plot(u2_t, F_sol, 'g-', lw=1.5, label='Dynamic (Newmark-beta)')
    plt.xlabel('u2 [mm]')
    plt.ylabel('Force F2 [N]')
    plt.title('Dynamic vs analytical solution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 2: time response u1(t), u2(t)
    plt.figure(figsize=(8, 5))
    plt.plot(t * 1e3, u2_t, 'k--', label='u2(t) imposed')
    plt.plot(t * 1e3, u1_sol, 'g-', label='u1(t) (mass m1)')
    plt.xlabel('time [ms]')
    plt.ylabel('displacement [mm]')
    plt.title('Dynamic response of node 1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 3: residual R(u1) per time step + NR iterates
    n_res_demo = 15
    u1_sweep = np.linspace(0.0, u1_sol.max() * 1.2, 400)
    colors = plt.cm.viridis(np.linspace(0, 1, n_res_demo))
    idx_demo = np.linspace(0, len(history) - 1, n_res_demo).astype(int)

    plt.figure(figsize=(8, 6))
    for color, i in zip(colors, idx_demo):
        u2_np1, u_tilde, c1, kappa, hist_x, hist_r = history[i]
        plt.plot(u1_sweep, res_dynamic(u1_sweep, u2_np1, u_tilde, c1, kappa),
                 color=color, lw=1.2, alpha=0.8)
        plt.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel('u1 [mm]')
    plt.ylabel('Dynamic residual R(u1) [N]')
    plt.title('Dynamic: residual(u1) per time step + NR iterates')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
