# -*- coding: utf-8 -*-
"""
Solution statique : Newton-Raphson quasi-statique, u2 imposé par incréments
croissants. A chaque incrément, on part de la solution précédente (warm
start) et on résout res_static(u1, u2) = 0.

Comme le résidu peut avoir plusieurs racines (branche instable du
snap-back), le NR piloté en déplacement suit la branche stable la plus
proche et "saute" par-dessus la zone instable : la courbe F-u2 obtenue
diffère donc de la solution analytique autour du pic.
"""

import numpy as np

import modele as mdl
import parametres as prm


def resoudre(inc=prm.inc, u2_f=None):
    """
    Résout le problème statique sur `inc` incréments de 0 à u2_f
    (valeurs par défaut dans parametres.py).

    Renvoie :
        u2_steps : (inc,) déplacements imposés
        u1_sol   : (inc,) solutions u1 à l'équilibre
        F_sol    : (inc,) force F2 correspondante
        historique : liste de (u2n, kappa, hist_u1, hist_res) par incrément,
                     utile pour tracer le résidu + les itérations NR
    """
    if u2_f is None:
        u2_f = mdl.u2_f

    u2_steps = np.linspace(0.0, u2_f, inc)
    u1_sol = np.empty(inc)
    F_sol = np.empty(inc)
    historique = []

    u1_prev = 0.0
    kappa = 0.0  # variable d'histoire de l'endommagement (max de u1 atteint)
    for n, u2n in enumerate(u2_steps):
        residual = lambda u, u2n=u2n, kappa=kappa: mdl.res_static(u, u2n, kappa)
        tangent = lambda u, kappa=kappa: mdl.tangent_static(u, kappa)
        u1_n, ok, hist_x, hist_r = mdl.newton_raphson(residual, tangent, u1_prev)
        if not ok:
            # Au-delà du point limite, plus de racine près de l'itéré précédent :
            # le NR cycle. On relance depuis la branche totalement endommagée
            # (u1 ~ u2), ce qui réalise le saut (snap) vers l'autre branche stable.
            u1_n, ok, hist_x2, hist_r2 = mdl.newton_raphson(residual, tangent, u2n)
            hist_x = np.concatenate([hist_x, hist_x2])
            hist_r = np.concatenate([hist_r, hist_r2])
            print(f"[Statique] pas {n} (u2={u2n:.3f} mm) : saut de branche, "
                  f"relance NR depuis u1=u2 -> {'convergé' if ok else 'ECHEC'}")
        u1_sol[n] = u1_n
        F_sol[n] = mdl.k2 * (u2n - u1_n)
        historique.append((u2n, kappa, hist_x, hist_r))
        u1_prev = u1_n
        kappa = max(kappa, u1_n)

    return u2_steps, u1_sol, F_sol, historique


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from solution_analytique import courbe_analytique

    u2_steps, u1_sol, F_sol, historique = resoudre()
    _, u2_ana, F_ana = courbe_analytique()

    # -- Fig 1 : F-u2 statique vs analytique
    plt.figure(figsize=(8, 5))
    plt.plot(u2_ana, F_ana, 'r--', lw=2, label='Analytique (avec snap-back)')
    plt.plot(u2_steps, F_sol, 'b-', lw=1.5, label='Statique (Newton-Raphson)')
    plt.xlabel('u2 [mm]')
    plt.ylabel('Force F2 [N]')
    plt.title('Solution statique vs analytique')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 2 : résidu R(u1) par incrément + points d'itération NR
    n_res_demo = 15
    u1_sweep = np.linspace(0.0, u1_sol.max() * 1.2, 400)
    colors = plt.cm.viridis(np.linspace(0, 1, n_res_demo))
    idx_demo = np.linspace(0, len(historique) - 1, n_res_demo).astype(int)

    plt.figure(figsize=(8, 6))
    for color, i in zip(colors, idx_demo):
        u2n, kappa, hist_x, hist_r = historique[i]
        plt.plot(u1_sweep, mdl.res_static(u1_sweep, u2n, kappa), color=color, lw=1.2, alpha=0.8)
        plt.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel('u1 [mm]')
    plt.ylabel('Résidu statique R(u1) [N]')
    plt.title('Statique : résidu(u1) par incrément + itérations NR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
