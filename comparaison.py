# -*- coding: utf-8 -*-
"""
Point d'entrée : assemble les trois solutions (analytique, statique,
dynamique) et produit les figures de comparaison, en particulier la
comparaison statique vs dynamique des courbes de résidu Newton-Raphson
(une courbe de résidu par incrément, avec les itérations NR en points sur
la courbe).

Lancer : python comparaison.py          (affiche les figures)
         python comparaison.py --save   (les enregistre aussi dans figures/)
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import modele as mdl
import parametres as prm
from solution_analytique import courbe_analytique, energie_dissipee
from solution_statique import resoudre as resoudre_statique
from solution_dynamique import resoudre as resoudre_dynamique, res_dynamic


def main(save=False):
    prm.resume()

    # -- Solutions ----------------------------------------------------------
    u1_ana, u2_ana, F_ana = courbe_analytique()
    F_max, u2_fmax, energie, _ = energie_dissipee(u1_ana, u2_ana, F_ana)
    print(f"[Analytique] F_max = {F_max:.1f} N atteint pour u2 = {u2_fmax:.3f} mm")
    print(f"[Analytique] Energie libérée au saut (aire saut / branche d'équilibre) = {energie:.3f} J")

    u2_steps, u1_static, F_static, hist_static = resoudre_statique()

    t_dyn, u2_dyn, u1_dyn, v1_dyn, a1_dyn, F_dyn, hist_dyn = resoudre_dynamique()
    print(f"[Dynamique] {len(hist_dyn)} pas de temps, dt = {t_dyn[1]*1e3:.3f} ms")

    # -- Fig 1 : F-u2, les trois solutions ----------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(u2_ana, F_ana, 'r--', lw=2, label='Analytique (avec snap-back)')
    plt.plot(u2_steps, F_static, 'b-', lw=1.5, label='Statique (Newton-Raphson)')
    plt.plot(u2_dyn, F_dyn, 'g-', lw=1.5, label='Dynamique (Newmark-beta)')
    plt.xlabel('u2 [mm]')
    plt.ylabel('Force F2 [N]')
    plt.title('Réponse force-déplacement : analytique vs statique vs dynamique')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 2 : endommagement du ressort 1 ---------------------------------
    plt.figure(figsize=(8, 5))
    u1_plot = np.linspace(0.0, max(u1_static.max(), u1_dyn.max()) * 1.1, 500)
    plt.plot(u1_plot, mdl.damage(u1_plot), 'k-', label='d(u1)')
    plt.plot(u1_static, mdl.damage(u1_static), 'b.', ms=3, label='Statique')
    plt.plot(u1_dyn, mdl.damage(u1_dyn), 'g.', ms=3, label='Dynamique')
    plt.xlabel('u1 [mm]')
    plt.ylabel('Endommagement d')
    plt.title("Évolution de l'endommagement du ressort 1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 3 : comparaison des résidus NR, statique vs dynamique ----------
    n_res_demo = 15
    u1_max_plot = max(u1_static.max(), u1_dyn.max()) * 1.2
    u1_sweep = np.linspace(0.0, u1_max_plot, 400)
    colors = plt.cm.viridis(np.linspace(0, 1, n_res_demo))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    ax = axes[0]
    idx_static = np.linspace(0, len(hist_static) - 1, n_res_demo).astype(int)
    for color, i in zip(colors, idx_static):
        u2n, kappa, hist_x, hist_r = hist_static[i]
        ax.plot(u1_sweep, mdl.res_static(u1_sweep, u2n, kappa), color=color, lw=1.2, alpha=0.8)
        ax.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('u1 [mm]')
    ax.set_ylabel('Résidu statique R(u1) [N]')
    ax.set_title('Statique : résidu(u1) par incrément + itérations NR')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    idx_dyn = np.linspace(0, len(hist_dyn) - 1, n_res_demo).astype(int)
    for color, i in zip(colors, idx_dyn):
        u2_np1, u_tilde, c1, kappa, hist_x, hist_r = hist_dyn[i]
        ax.plot(u1_sweep, res_dynamic(u1_sweep, u2_np1, u_tilde, c1, kappa),
                  color=color, lw=1.2, alpha=0.8)
        ax.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('u1 [mm]')
    ax.set_ylabel('Résidu dynamique R(u1) [N]')
    ax.set_title('Dynamique : résidu(u1) par pas de temps + itérations NR')
    ax.grid(True, alpha=0.3)

    fig.suptitle("Comparaison des résidus Newton-Raphson : l'inertie (terme m1*c1) "
                  "régularise la non-monotonie du résidu statique")
    plt.tight_layout()

    # -- Fig 4 : réponse temporelle dynamique -------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t_dyn * 1e3, u2_dyn, 'k--', label='u2(t) imposé')
    plt.plot(t_dyn * 1e3, u1_dyn, 'g-', label='u1(t) (masse m1)')
    plt.xlabel('temps [ms]')
    plt.ylabel('déplacement [mm]')
    plt.title('Réponse dynamique du noeud 1 (rampe puis maintien de u2)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 5 : zoom sur la fenêtre de rupture dynamique ------------------
    i_snap = np.argmax(u1_dyn > 5.0 * mdl.uc)  # premier pas où le ressort 1 est rompu
    T_libre = 2.0 * np.pi * np.sqrt(mdl.m1 / mdl.k2)
    win = (t_dyn > t_dyn[i_snap] - 2 * T_libre) & (t_dyn < t_dyn[i_snap] + 6 * T_libre)
    plt.figure(figsize=(10, 5))
    plt.plot(t_dyn[win] * 1e3, u2_dyn[win], 'k--', label='u2(t) imposé')
    plt.plot(t_dyn[win] * 1e3, u1_dyn[win], 'g.-', ms=3, label='u1(t) (masse m1)')
    plt.xlabel('temps [ms]')
    plt.ylabel('déplacement [mm]')
    plt.title(f'Zoom sur la rupture : oscillation libre de période 2π√(m1/k2) = {T_libre*1e3:.2f} ms')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
        os.makedirs(outdir, exist_ok=True)
        names = ['force_deplacement', 'endommagement', 'residus_statique_dynamique',
                 'reponse_temporelle', 'zoom_rupture']
        for num, name in zip(plt.get_fignums(), names):
            plt.figure(num).savefig(os.path.join(outdir, f'{name}.pdf'))
            plt.figure(num).savefig(os.path.join(outdir, f'{name}.png'), dpi=150)
        print(f"[Figures] enregistrées dans {outdir}")

    plt.show()


if __name__ == "__main__":
    main(save='--save' in sys.argv)
