# -*- coding: utf-8 -*-
"""
Shared model: two springs (bars) in series, spring 1 damageable.

Node 0 (fixed) --[spring 1, k1_d(u1)]-- Node 1 (mass m1) --[spring 2, k2]-- Node 2 (u2 imposed)

This module defines the damage law, the static residual and the generic
Newton-Raphson solver. All parameters (physical and numerical) live in
parameters.py. It is imported by analytical_solution.py, static_solution.py
and dynamic_solution.py.

Consistent units: N, mm, s  =>  mass in tonne (1 tonne = 1000 kg).
"""

import numpy as np

from parameters import k1, k2, uc, m1, u2_f, eps_nr, max_iter


# ---------------------------------------------------------------------------
# Damage law of spring 1 and derivatives
#
# Damage is irreversible: it is driven by the history variable
# kappa = max(u1) reached during loading. As long as u1 >= kappa
# (loading), d = d(u1); if u1 < kappa (unloading), d stays frozen at
# d(kappa) and the spring unloads elastically with the secant stiffness
# k1*(1-d). With kappa = 0 (default) and u1 >= 0 we recover d = d(u1):
# this is the monotonic loading case (analytical and static solutions).
# ---------------------------------------------------------------------------
def damage(u1):
    """d(u1) = 1 - exp(-(u1/uc)^2), d=0 intact, d=1 fully damaged."""
    return 1.0 - np.exp(-(u1**2) / uc**2)


def ddamage_du1(u1):
    return (2.0 * u1 / uc**2) * np.exp(-(u1**2) / uc**2)


def k1_d(u1, kappa=0.0):
    """Damaged secant stiffness of spring 1."""
    return k1 * (1.0 - damage(np.maximum(u1, kappa)))


def dk1_d_du1(u1, kappa=0.0):
    return np.where(u1 >= kappa, -k1 * ddamage_du1(u1), 0.0)


# ---------------------------------------------------------------------------
# Static residual / tangent (equilibrium of node 1, u2 imposed)
# ---------------------------------------------------------------------------
def res_static(u1, u2, kappa=0.0):
    return u1 * (k1_d(u1, kappa) + k2) - k2 * u2


def tangent_static(u1, kappa=0.0):
    return k1_d(u1, kappa) + u1 * dk1_d_du1(u1, kappa) + k2


# ---------------------------------------------------------------------------
# Generic Newton-Raphson: returns the solution, a convergence flag and the
# iteration history (used to plot the iterates on the residual curves, in
# statics as in dynamics).
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
