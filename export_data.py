# -*- coding: utf-8 -*-
"""
Export of all computed data to plain-text .dat files (whitespace-separated
columns, one header line starting with '#'), for plotting with pgfplots
(\\addplot table) or Manim (numpy.loadtxt).

Run:  python export_data.py            (writes into data/)

Files written (units N, mm, s, t):
  parameters.dat           key / value of every parameter and derived quantity
  damage_law.dat           u1, x = u1/uc, d, k1d = k1(1-d), F1 = k1d*u1, dF1/du1, k_t
  analytical.dat           u1, u2, F, d  along the complete equilibrium branch
  static.dat               u2, u1, F, d, nr_iterations  (NaN after a failure)
  dynamic.dat              t, u2, u1, v1, a1, F, d, nr_iterations
  residuals_static.dat     u1 then one column R(u1; u2_n) per selected increment
                           (the u2_n values are listed in the header)
  residuals_dynamic.dat    same for the dynamic residual at selected time steps
  nr_iterates_static.dat   NR iterates (i, u1, R) of the increment crossing the limit point
  energies.dat             F_max, u2_fmax, released energy |W|, kinetic energy max
"""

import os

import numpy as np

import model as mdl
import parameters as prm
from analytical_solution import analytical_curve, released_energy
from static_solution import solve as solve_static
from dynamic_solution import solve as solve_dynamic, res_dynamic

N_RES_CURVES = 15   # number of residual curves exported (as in the figures)


def _save(path, header, columns):
    data = np.column_stack(columns)
    np.savetxt(path, data, header=header, comments='# ', fmt='%.10e')


def export(outdir=None):
    if outdir is None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(outdir, exist_ok=True)

    # -- parameters ---------------------------------------------------------
    names = ['E1', 'A1', 'L1', 'rho1', 'f_t', 'E2', 'A2', 'L2', 'rho2', 'm_grip', 'u2_f',
             'inc', 'eps_nr', 'max_iter', 'max_iter_dyn', 'beta_nm', 'gamma_nm', 'N_per_period', 'v2', 't_hold',
             'k1', 'k2', 'F_max', 'uc', 'G_f', 'm1', 'omega0', 'T0', 'dt', 'T_ramp',
             'n_steps', 'n_hold', 'c1']
    with open(os.path.join(outdir, 'parameters.dat'), 'w') as f:
        f.write('# name value   (units N, mm, s, t)\n')
        for name in names:
            f.write(f'{name} {getattr(prm, name):.10e}\n')

    # -- damage law ---------------------------------------------------------
    u1 = np.linspace(0.0, 4.0 * prm.uc, 801)
    x = u1 / prm.uc
    d = mdl.damage(u1)
    k1d = mdl.k1_d(u1)
    F1 = k1d * u1
    dF1 = prm.k1 * np.exp(-x**2) * (1.0 - 2.0 * x**2)
    kt = dF1 + prm.k2
    _save(os.path.join(outdir, 'damage_law.dat'),
          'u1 x d k1d F1 dF1_du1 k_t', [u1, x, d, k1d, F1, dF1, kt])

    # -- analytical branch --------------------------------------------------
    u1a, u2a, Fa = analytical_curve()
    _save(os.path.join(outdir, 'analytical.dat'), 'u1 u2 F d', [u1a, u2a, Fa, mdl.damage(u1a)])
    F_max, u2_fmax, energy, _ = released_energy(u1a, u2a, Fa)

    # -- static -------------------------------------------------------------
    u2s, u1s, Fs, hist_s = solve_static()
    it_s = np.full(len(u2s), np.nan)
    it_s[:len(hist_s)] = [len(h[2]) - 1 for h in hist_s]
    _save(os.path.join(outdir, 'static.dat'), 'u2 u1 F d nr_iterations',
          [u2s, u1s, Fs, mdl.damage(u1s), it_s])

    # -- dynamic ------------------------------------------------------------
    t, u2d, u1d, v1d, a1d, Fd, hist_d = solve_dynamic()
    it_d = np.concatenate([[0], [len(h[4]) - 1 for h in hist_d]])
    _save(os.path.join(outdir, 'dynamic.dat'), 't u2 u1 v1 a1 F d nr_iterations',
          [t, u2d, u1d, v1d, a1d, Fd, np.maximum.accumulate(mdl.damage(u1d)), it_d])

    # -- residual curves ----------------------------------------------------
    u1_max = max(np.nanmax(u1s), u1d.max()) * 1.2
    u1_sweep = np.linspace(0.0, u1_max, 400)
    idx = np.linspace(0, len(hist_s) - 1, N_RES_CURVES).astype(int)
    cols, u2_list = [u1_sweep], []
    for i in idx:
        u2n, d_prev, _, _ = hist_s[i]
        cols.append(mdl.res_static(u1_sweep, u2n, d_prev))
        u2_list.append(u2n)
    _save(os.path.join(outdir, 'residuals_static.dat'),
          'u1 ' + ' '.join(f'R_u2={v:.6g}' for v in u2_list), cols)
    idx = np.linspace(0, len(hist_d) - 1, N_RES_CURVES).astype(int)
    cols, u2_list = [u1_sweep], []
    for i in idx:
        u2n, u_tilde, c1, d_prev, _, _ = hist_d[i]
        cols.append(res_dynamic(u1_sweep, u2n, u_tilde, c1, d_prev))
        u2_list.append(u2n)
    _save(os.path.join(outdir, 'residuals_dynamic.dat'),
          'u1 ' + ' '.join(f'R_u2={v:.6g}' for v in u2_list), cols)

    # -- NR iterates of the increment crossing the limit point --------------
    n_lim = int(np.argmax(u2s > u2_fmax))
    n_lim = min(n_lim, len(hist_s) - 1)
    u2n, d_prev, hx, hr = hist_s[n_lim]
    _save(os.path.join(outdir, 'nr_iterates_static.dat'),
          f'i u1 R   (increment {n_lim}, u2 = {u2n:.6g} mm)', [np.arange(len(hx)), hx, hr])

    # -- energies -----------------------------------------------------------
    K_max = 0.5 * prm.m1 * np.max(v1d**2) / 1000.0   # J
    with open(os.path.join(outdir, 'energies.dat'), 'w') as f:
        f.write('# name value\n')
        f.write(f'F_max_N {F_max:.10e}\nu2_at_F_max_mm {u2_fmax:.10e}\n'
                f'released_energy_J {energy:.10e}\nkinetic_energy_max_J {K_max:.10e}\n')

    print(f"[Data] written in {outdir}: " + ', '.join(sorted(os.listdir(outdir))))
    return outdir


if __name__ == "__main__":
    export()
