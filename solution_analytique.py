# -*- coding: utf-8 -*-
"""
Solution analytique (forme fermée) du système à deux ressorts en série,
ressort 1 endommageable.

A l'équilibre statique : u1*(k1_d(u1) + k2) = k2*u2
  =>  u2(u1) = u1 * (1 + k1_d(u1)/k2)
  =>  F(u1)  = k2*(u2 - u1) = k1_d(u1)*u1

En paramétrant par u1 on obtient directement la courbe F-u2 complète, y
compris la branche instable (snap-back) que le solveur statique piloté en
déplacement ne peut pas suivre.
"""

import numpy as np

import modele as mdl


def courbe_analytique(n=4000, u1_max=None):
    """Renvoie (u1, u2, F) le long de la branche analytique complète."""
    if u1_max is None:
        u1_max = mdl.u2_f
    u1 = np.linspace(0.0, u1_max, n)
    u2 = u1 * (1.0 + mdl.k1_d(u1) / mdl.k2)
    F = mdl.k1_d(u1) * u1
    return u1, u2, F


def energie_dissipee(u1, u2, F):
    """
    Aire |int F du2| entre le saut quasi-statique vertical (à u2 = u2_fmax,
    de F_max à ~0) et la branche d'équilibre analytique qui relie les deux
    mêmes états. Avec Psi l'énergie libre et D_path la dissipation par
    endommagement le long de la branche d'équilibre, on a
        Psi_1 - Psi_2 = D_path + aire,
    c'est-à-dire que cette aire est l'énergie libérée lors du saut
    (convertie en énergie cinétique) en excès de ce que la branche
    d'équilibre aurait dissipé par endommagement.
    """
    F_max = np.max(F)
    id_max = np.argmax(F)
    u2_fmax = u2[id_max]

    tol = 2.0 * (u1[1] - u1[0])  # tolérance liée à la résolution de la courbe
    match = np.where(np.isclose(u2, u2_fmax, atol=tol))[0]
    while len(match) < 2 and tol < u2_fmax:
        tol *= 1.5
        match = np.where(np.isclose(u2, u2_fmax, atol=tol))[0]

    idx1, idx2 = match[0], match[-1]
    u2_loop = u2[idx1:idx2 + 1]
    F_loop = F[idx1:idx2 + 1]
    energie = abs(np.trapezoid(F_loop, u2_loop)) / 1000.0  # N.mm -> J

    return F_max, u2_fmax, energie, (idx1, idx2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    u1, u2, F = courbe_analytique()
    F_max, u2_fmax, energie, _ = energie_dissipee(u1, u2, F)

    print(f"F_max = {F_max:.1f} N atteint pour u2 = {u2_fmax:.3f} mm")
    print(f"Energie libérée au saut (aire saut / branche d'équilibre) = {energie:.3f} J")

    plt.figure(figsize=(8, 5))
    plt.plot(u2, F, 'r--', lw=2, label='Analytique (avec snap-back)')
    plt.xlabel('u2 [mm]')
    plt.ylabel('Force F2 [N]')
    plt.title('Solution analytique')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
