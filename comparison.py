# -*- coding: utf-8 -*-
"""
Entry point: assembles the three solutions (analytical, static, dynamic)
and produces the comparison figures, in particular the static vs dynamic
comparison of the Newton-Raphson residual curves (one residual curve per
increment, with the NR iterates as dots on the curve) and of the number of
NR iterations per increment / time step.

Run:  python comparison.py          (shows the figures)
      python comparison.py --save   (also saves them in figures/)
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import model as mdl
import parameters as prm
from analytical_solution import analytical_curve, released_energy
from static_solution import solve as solve_static
from dynamic_solution import solve as solve_dynamic, res_dynamic


def main(save=False):
    prm.summary()

    # -- Solutions ----------------------------------------------------------
    u1_ana, u2_ana, F_ana = analytical_curve()
    F_max, u2_fmax, energy, _ = released_energy(u1_ana, u2_ana, F_ana)
    print(f"[Analytical] F_max = {F_max:.1f} N reached at u2 = {u2_fmax:.3f} mm")
    print(f"[Analytical] Energy released at the jump (area jump / equilibrium branch) = {energy:.3f} J")

    u2_steps, u1_static, F_static, hist_static = solve_static()

    t_dyn, u2_dyn, u1_dyn, v1_dyn, a1_dyn, F_dyn, hist_dyn = solve_dynamic()
    print(f"[Dynamic] {len(hist_dyn)} time steps, dt = {t_dyn[1]*1e3:.3f} ms")

    # -- Fig 1: F-u2, the three solutions -----------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(u2_ana, F_ana, 'r--', lw=2, label='Analytical (with snap-back)')
    plt.plot(u2_steps, F_static, 'b-', lw=1.5, label='Static (Newton-Raphson, stops at the limit point)')
    plt.plot(u2_dyn, F_dyn, 'g-', lw=1.5, label='Dynamic (Newmark-beta)')
    plt.xlabel('u2 [mm]')
    plt.ylabel('Force F2 [N]')
    plt.title('Force-displacement response: analytical vs static vs dynamic')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 2: damage of spring 1 ------------------------------------------
    # Left: d versus u1. Right: the same with 1 - d on a log scale, which
    # opens up the region d ~ 1 where the post-rupture oscillation lives.
    # The dynamic solution is split into loading steps (trial damage reaches
    # or exceeds the damage already attained: the point lies on the curve
    # d(u1)) and unloading steps (u1 has moved back below the maximum
    # reached: d is frozen by irreversibility and the point lies off the
    # curve, on a horizontal line). After rupture the mass oscillates, so
    # the post-rupture steps sweep u1 back and forth at constant d.
    d_trial = mdl.damage(u1_dyn)
    d_dyn = np.maximum.accumulate(d_trial)
    # Loading step = u1 at or above the maximum reached so far. This kinematic
    # test is used instead of comparing damages, because beyond u1 ~ 6 uc the
    # trial damage equals 1 to machine precision and damages can no longer
    # tell loading from unloading.
    loading = u1_dyn >= np.maximum.accumulate(u1_dyn) - 1e-12
    u1_plot = np.linspace(1e-4 * mdl.uc, max(np.nanmax(u1_static), u1_dyn.max()) * 1.1, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ylog in zip(axes, (False, True)):
        if ylog:
            y_curve, y_sta = 1.0 - mdl.damage(u1_plot), 1.0 - mdl.damage(u1_static)
            y_dyn = np.maximum(1.0 - d_dyn, 1e-10)
        else:
            y_curve, y_sta, y_dyn = mdl.damage(u1_plot), mdl.damage(u1_static), d_dyn
        ax.plot(u1_plot, y_curve, 'k-', lw=1.5, label='trial damage, loading curve')
        ax.plot(u1_dyn, y_dyn, '-', color='0.8', lw=0.6, label='dynamic path (time ordered)')
        ax.plot(u1_static, y_sta, 'b.', ms=4, label='static')
        ax.plot(u1_dyn[loading], y_dyn[loading], 'g.', ms=4, label='dynamic, loading: on the curve')
        ax.plot(u1_dyn[~loading], y_dyn[~loading], '.', color='tab:orange', ms=4,
                label='dynamic, unloading: d frozen, off the curve')
        ax.set_xlabel('u1 [mm]')
        ax.grid(True, alpha=0.3)
        if ylog:
            ax.set_yscale('log')
            ax.set_ylim(1e-10, 2.0)
            ax.set_ylabel('1 - d (log scale, floored at 1e-10 = numerically d = 1)')
            ax.set_title('Same, 1 - d on a log scale: the post-rupture oscillation at constant d')
            ax.annotate('after rupture the mass oscillates:\nu1 sweeps back and forth at d = 1',
                        xy=(0.5 * u1_dyn.max(), 1e-10), xytext=(0.35 * u1_dyn.max(), 1e-5),
                        arrowprops=dict(arrowstyle='->', color='tab:orange'), fontsize=9, color='tab:orange')
        else:
            ax.set_ylabel('damage d')
            ax.set_title('Damage evolution of spring 1 (loading vs unloading)')
            ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()

    # -- Fig 3: NR residual comparison, static vs dynamic -------------------
    n_res_demo = 15
    u1_max_plot = max(np.nanmax(u1_static), u1_dyn.max()) * 1.2
    u1_sweep = np.linspace(0.0, u1_max_plot, 400)
    colors = plt.cm.viridis(np.linspace(0, 1, n_res_demo))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    ax = axes[0]
    idx_static = np.linspace(0, len(hist_static) - 1, n_res_demo).astype(int)
    for color, i in zip(colors, idx_static):
        u2n, d_prev, hist_x, hist_r = hist_static[i]
        ax.plot(u1_sweep, mdl.res_static(u1_sweep, u2n, d_prev), color=color, lw=1.2, alpha=0.8)
        ax.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('u1 [mm]')
    ax.set_ylabel('Static residual R(u1) [N]')
    ax.set_title('Static: residual(u1) per increment + NR iterates')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    idx_dyn = np.linspace(0, len(hist_dyn) - 1, n_res_demo).astype(int)
    for color, i in zip(colors, idx_dyn):
        u2_np1, u_tilde, c1, d_prev, hist_x, hist_r = hist_dyn[i]
        ax.plot(u1_sweep, res_dynamic(u1_sweep, u2_np1, u_tilde, c1, d_prev),
                color=color, lw=1.2, alpha=0.8)
        ax.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('u1 [mm]')
    ax.set_ylabel('Dynamic residual R(u1) [N]')
    ax.set_title('Dynamic: residual(u1) per time step + NR iterates')
    ax.grid(True, alpha=0.3)

    fig.suptitle("Newton-Raphson residual comparison: inertia (term m1*c1) "
                 "regularises the non-monotonic static residual")
    plt.tight_layout()

    # -- Fig 4: dynamic time response ---------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t_dyn * 1e3, u2_dyn, 'k--', label='u2(t) imposed')
    plt.plot(t_dyn * 1e3, u1_dyn, 'g-', label='u1(t) (mass m1)')
    plt.xlabel('time [ms]')
    plt.ylabel('displacement [mm]')
    plt.title('Dynamic response of node 1 (ramp then hold of u2)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 5: zoom on the dynamic rupture window --------------------------
    i_snap = np.argmax(u1_dyn > 5.0 * mdl.uc)  # first step where spring 1 is broken
    T_free = 2.0 * np.pi * np.sqrt(mdl.m1 / mdl.k2)
    win = (t_dyn > t_dyn[i_snap] - 2 * T_free) & (t_dyn < t_dyn[i_snap] + 6 * T_free)
    plt.figure(figsize=(10, 5))
    plt.plot(t_dyn[win] * 1e3, u2_dyn[win], 'k--', label='u2(t) imposed')
    plt.plot(t_dyn[win] * 1e3, u1_dyn[win], 'g.-', ms=3, label='u1(t) (mass m1)')
    plt.xlabel('time [ms]')
    plt.ylabel('displacement [mm]')
    plt.title(f'Zoom on the rupture: free oscillation of period 2π√(m1/k2) = {T_free*1e3:.2f} ms')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 6: NR iteration count, static vs dynamic ----------------------
    it_static = np.array([len(h[2]) - 1 for h in hist_static])
    it_dyn = np.array([len(h[4]) - 1 for h in hist_dyn])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    ax.plot(u2_steps[:len(it_static)], it_static, 'b.-', ms=3, lw=0.8)
    ax.axhline(prm.max_iter, color='r', ls='--', lw=1, label=f'max_iter = {prm.max_iter}')
    ax.set_xlabel('u2 [mm]')
    ax.set_ylabel('NR iterations per increment')
    ax.set_title('Static: the NR fails at the limit point (calculation stops)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = axes[1]
    ax.plot(u2_dyn[1:], it_dyn, 'g.-', ms=3, lw=0.8)
    ax.axhline(prm.max_iter, color='r', ls='--', lw=1, label=f'max_iter = {prm.max_iter}')
    ax.set_xlabel('u2 [mm]')
    ax.set_ylabel('NR iterations per time step')
    ax.set_title('Dynamic: a few iterations everywhere, including through the snap')
    ax.set_ylim(0, prm.max_iter * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    # -- Fig 7: damage law d(u1), secant stiffness and spring-1 force --------
    u1_law = np.linspace(0.0, 4.0 * mdl.uc, 801)
    x_law = u1_law / mdl.uc
    marks = [(mdl.uc / np.sqrt(2.0), r'$u_c/\sqrt{2}$ (peak)'), (mdl.uc, r'$u_c$'),
             (mdl.uc * np.sqrt(1.5), r'$u_c\sqrt{3/2}$ (steepest slope)')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(u1_law, mdl.damage(u1_law), 'k-', lw=2, label='$d(u_1) = 1 - e^{-(u_1/u_c)^2}$')
    ax.plot(u1_law, mdl.k1_d(u1_law) / mdl.k1, 'b--', lw=1.5, label='$k_1^d/k_1 = 1 - d$')
    for um, lab in marks:
        ax.axvline(um, color='gray', lw=0.8, ls=':')
        ax.plot(um, mdl.damage(um), 'ro', ms=5)
        ax.annotate(lab, (um, mdl.damage(um)), xytext=(5, -12), textcoords='offset points', fontsize=8)
    ax.set_xlabel('u1 [mm]')
    ax.set_ylabel('damage d, relative stiffness')
    ax.set_title('Damage law and secant stiffness of spring 1')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right')
    ax = axes[1]
    ax.plot(u1_law, mdl.k1_d(u1_law) * u1_law, 'k-', lw=2, label='$F_1 = k_1(1-d)\,u_1$')
    ax.plot(u1_law, mdl.k1 * u1_law, 'b:', lw=1, label='$k_1 u_1$ (intact)')
    for um, lab in marks:
        ax.axvline(um, color='gray', lw=0.8, ls=':')
        ax.plot(um, mdl.k1_d(um) * um, 'ro', ms=5)
    ax.set_ylim(0, 1.4 * mdl.k1_d(mdl.uc / np.sqrt(2.0)) * mdl.uc / np.sqrt(2.0))
    ax.set_xlabel('u1 [mm]')
    ax.set_ylabel('F1 [N]')
    ax.set_title('Force-elongation of spring 1')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
        os.makedirs(outdir, exist_ok=True)
        names = ['force_displacement', 'damage', 'residuals_static_dynamic',
                 'time_response', 'rupture_zoom', 'nr_iterations', 'damage_law']
        for num, name in zip(plt.get_fignums(), names):
            plt.figure(num).savefig(os.path.join(outdir, f'{name}.pdf'))
            plt.figure(num).savefig(os.path.join(outdir, f'{name}.png'), dpi=150)
        print(f"[Figures] saved in {outdir}")

    plt.show()


if __name__ == "__main__":
    main(save='--save' in sys.argv)
