# -*- coding: utf-8 -*-
"""
Solution dynamique : schéma de Newmark implicite (prédicteur-correcteur,
beta=0.25, gamma=0.5 -> moyenne des accélérations, inconditionnellement
stable, pas d'amortissement numérique) avec Newton-Raphson à chaque pas de
temps sur le noeud 1 (masse m1).

Le résidu dynamique est simplement le résidu statique (modele.res_static)
augmenté du terme d'inertie m1*a(u1), où a(u1) est l'accélération au pas
n+1 exprimée via les prédicteurs de Newmark :

    u_tilde, v_tilde = prédicteurs à partir de (u_n, v_n, a_n)
    a(u1) = c1 * (u1 - u_tilde)          avec c1 = 1/(beta*dt^2)
    v(u1) = v_tilde + gamma*dt*a(u1)

Le terme m1*c1 s'ajoute à la tangente statique : c'est lui qui "régularise"
la tangente dans la zone où le ressort 1 s'adoucit (snap-back), et qui
change la forme du résidu R(u1) par rapport au cas statique.
"""

import numpy as np

import modele as mdl
import parametres as prm

beta_nm, gamma_nm = prm.beta_nm, prm.gamma_nm


def predicteurs(u_n, v_n, a_n, dt):
    u_tilde = u_n + dt * v_n + dt**2 / 2.0 * (1.0 - 2.0 * beta_nm) * a_n
    v_tilde = v_n + (1.0 - gamma_nm) * dt * a_n
    return u_tilde, v_tilde


def res_dynamic(u1, u2, u_tilde, c1, kappa=0.0):
    a = c1 * (u1 - u_tilde)
    return mdl.m1 * a + mdl.res_static(u1, u2, kappa)


def tangent_dynamic(u1, c1, kappa=0.0):
    return mdl.m1 * c1 + mdl.tangent_static(u1, kappa)


def resoudre(dt=prm.dt, n_steps=prm.n_steps, n_hold=prm.n_hold, u2_f=prm.u2_f):
    """
    Intègre la réponse dynamique du noeud 1 pendant une rampe de
    déplacement imposé u2(t), de 0 à u2_f sur n_steps pas, suivie d'une
    phase de maintien u2 = u2_f pendant n_hold pas (pour laisser le snap
    se terminer et observer les oscillations libres de la masse m1).

    Les valeurs par défaut viennent de parametres.py : dt = T0/N_par_periode
    (le rapport inertie/raideur dans la tangente vaut N²/pi² avec beta=1/4),
    n_steps déduit de la vitesse de rampe v2, n_hold de la durée t_hold.

    Renvoie :
        t, u2_t, u1_sol, v1_sol, a1_sol, F_sol : historiques temporels
        historique : liste de (u2_np1, u_tilde, c1, kappa, hist_u1, hist_res)
                     par pas de temps, pour tracer le résidu + les itérations NR
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
    historique = []
    kappa = 0.0  # variable d'histoire de l'endommagement (max de u1 atteint)

    for n in range(n_tot):
        u2_np1 = min((n + 1) * dt * v2_rate, u2_f)
        u_tilde, v_tilde = predicteurs(u1_sol[n], v1_sol[n], a1_sol[n], dt)

        u1_np1, ok, hist_x, hist_r = mdl.newton_raphson(
            lambda u, u2_np1=u2_np1, u_tilde=u_tilde, kappa=kappa:
                res_dynamic(u, u2_np1, u_tilde, c1, kappa),
            lambda u, kappa=kappa: tangent_dynamic(u, c1, kappa),
            u_tilde,
        )
        if not ok:
            print(f"[Dynamique] pas {n} (t={t[n+1]*1e3:.1f} ms) : NR non convergé, "
                  f"résidu={hist_r[-1]:.3e} N")

        a1_np1 = c1 * (u1_np1 - u_tilde)
        v1_np1 = v_tilde + gamma_nm * dt * a1_np1

        u2_t[n + 1] = u2_np1
        u1_sol[n + 1] = u1_np1
        a1_sol[n + 1] = a1_np1
        v1_sol[n + 1] = v1_np1
        F_sol[n + 1] = mdl.k2 * (u2_np1 - u1_np1)

        historique.append((u2_np1, u_tilde, c1, kappa, hist_x, hist_r))
        kappa = max(kappa, u1_np1)

    return t, u2_t, u1_sol, v1_sol, a1_sol, F_sol, historique


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from solution_analytique import courbe_analytique

    t, u2_t, u1_sol, v1_sol, a1_sol, F_sol, historique = resoudre()
    _, u2_ana, F_ana = courbe_analytique()

    print(f"{len(historique)} pas de temps, dt = {t[1]*1e3:.3f} ms")

    # -- Fig 1 : F-u2 dynamique vs analytique
    plt.figure(figsize=(8, 5))
    plt.plot(u2_ana, F_ana, 'r--', lw=2, label='Analytique (avec snap-back)')
    plt.plot(u2_t, F_sol, 'g-', lw=1.5, label='Dynamique (Newmark-beta)')
    plt.xlabel('u2 [mm]')
    plt.ylabel('Force F2 [N]')
    plt.title('Solution dynamique vs analytique')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 2 : réponse temporelle u1(t), u2(t)
    plt.figure(figsize=(8, 5))
    plt.plot(t * 1e3, u2_t, 'k--', label='u2(t) imposé')
    plt.plot(t * 1e3, u1_sol, 'g-', label='u1(t) (masse m1)')
    plt.xlabel('temps [ms]')
    plt.ylabel('déplacement [mm]')
    plt.title('Réponse dynamique du noeud 1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -- Fig 3 : résidu R(u1) par pas de temps + points d'itération NR
    n_res_demo = 15
    u1_sweep = np.linspace(0.0, u1_sol.max() * 1.2, 400)
    colors = plt.cm.viridis(np.linspace(0, 1, n_res_demo))
    idx_demo = np.linspace(0, len(historique) - 1, n_res_demo).astype(int)

    plt.figure(figsize=(8, 6))
    for color, i in zip(colors, idx_demo):
        u2_np1, u_tilde, c1, kappa, hist_x, hist_r = historique[i]
        plt.plot(u1_sweep, res_dynamic(u1_sweep, u2_np1, u_tilde, c1, kappa),
                  color=color, lw=1.2, alpha=0.8)
        plt.plot(hist_x, hist_r, 'o', color=color, ms=5, mec='k', mew=0.4)
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel('u1 [mm]')
    plt.ylabel('Résidu dynamique R(u1) [N]')
    plt.title('Dynamique : résidu(u1) par pas de temps + itérations NR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
