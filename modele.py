# -*- coding: utf-8 -*-
"""
Modèle commun : deux ressorts (barres) en série, ressort 1 endommageable.

Noeud 0 (fixe) --[ressort 1, k1_d(u1)]-- Noeud 1 (masse m1) --[ressort 2, k2]-- Noeud 2 (u2 imposé)

Ce module définit la loi d'endommagement, le résidu statique et le solveur
de Newton-Raphson générique. Tous les paramètres (physiques et numériques)
sont dans parametres.py. Il est importé par solution_analytique.py,
solution_statique.py et solution_dynamique.py.

Unités cohérentes : N, mm, s  =>  masse en tonne (1 tonne = 1000 kg).
"""

import numpy as np

from parametres import k1, k2, uc, m1, u2_f, eps_nr, max_iter


# ---------------------------------------------------------------------------
# Loi d'endommagement du ressort 1 et dérivées
#
# L'endommagement est irréversible : il est piloté par la variable d'histoire
# kappa = max(u1) atteint au cours du chargement. Tant que u1 >= kappa
# (charge), d = d(u1) ; si u1 < kappa (décharge), d reste figé à d(kappa)
# et le ressort décharge élastiquement avec la raideur sécante k1*(1-d).
# Avec kappa = 0 (défaut) et u1 >= 0, on retrouve d = d(u1) : c'est le cas
# du chargement monotone (solutions analytique et statique).
# ---------------------------------------------------------------------------
def damage(u1):
    """d(u1) = 1 - exp(-(u1/uc)^2), d=0 sain, d=1 totalement endommagé."""
    return 1.0 - np.exp(-(u1**2) / uc**2)


def ddamage_du1(u1):
    return (2.0 * u1 / uc**2) * np.exp(-(u1**2) / uc**2)


def k1_d(u1, kappa=0.0):
    """Raideur sécante endommagée du ressort 1."""
    return k1 * (1.0 - damage(np.maximum(u1, kappa)))


def dk1_d_du1(u1, kappa=0.0):
    return np.where(u1 >= kappa, -k1 * ddamage_du1(u1), 0.0)


# ---------------------------------------------------------------------------
# Résidu / tangente statiques (équilibre du noeud 1, u2 imposé)
# ---------------------------------------------------------------------------
def res_static(u1, u2, kappa=0.0):
    return u1 * (k1_d(u1, kappa) + k2) - k2 * u2


def tangent_static(u1, kappa=0.0):
    return k1_d(u1, kappa) + u1 * dk1_d_du1(u1, kappa) + k2


# ---------------------------------------------------------------------------
# Newton-Raphson générique : renvoie la solution, un drapeau de convergence
# et l'historique des itérations (utilisé pour tracer les points
# d'itération sur les courbes de résidu, en statique comme en dynamique).
# ---------------------------------------------------------------------------
def newton_raphson(residual, tangent, x0, tol=eps_nr, max_it=max_iter):
    x = x0
    hist_x = [x]
    hist_r = [residual(x)]
    it = 0
    while abs(hist_r[-1]) > tol and it < max_it:
        x = x - hist_r[-1] / tangent(x)
        hist_x.append(x)
        hist_r.append(residual(x))
        it += 1
    converged = abs(hist_r[-1]) <= tol
    return x, converged, np.array(hist_x), np.array(hist_r)
