# -*- coding: utf-8 -*-
"""
Analytical (closed-form) solution of the two-spring system in series,
spring 1 damageable.

At static equilibrium: u1*(k1_d(u1) + k2) = k2*u2
  =>  u2(u1) = u1 * (1 + k1_d(u1)/k2)
  =>  F(u1)  = k2*(u2 - u1) = k1_d(u1)*u1

Parametrising by u1 gives directly the complete F-u2 curve, including the
unstable branch (snap-back) that the displacement-controlled static solver
cannot follow.
"""

import numpy as np

import model as mdl


def analytical_curve(n=4000, u1_max=None):
    """Return (u1, u2, F) along the complete analytical branch."""
    if u1_max is None:
        u1_max = mdl.u2_f
    u1 = np.linspace(0.0, u1_max, n)
    u2 = u1 * (1.0 + mdl.k1_d(u1) / mdl.k2)
    F = mdl.k1_d(u1) * u1
    return u1, u2, F


def released_energy(u1, u2, F):
    """
    Area |int F du2| between the vertical quasi-static jump (at u2 = u2_fmax,
    from F_max to ~0) and the analytical equilibrium branch joining the same
    two states. With Psi the free energy and D_path the damage dissipation
    along the equilibrium branch,
        Psi_1 - Psi_2 = D_path + area,
    i.e. this area is the energy released at the jump (converted into
    kinetic energy) in excess of what the equilibrium branch would have
    dissipated by damage.
    """
    F_max = np.max(F)
    id_max = np.argmax(F)
    u2_fmax = u2[id_max]

    tol = 2.0 * (u1[1] - u1[0])  # tolerance tied to the curve resolution
    match = np.where(np.isclose(u2, u2_fmax, atol=tol))[0]
    while len(match) < 2 and tol < u2_fmax:
        tol *= 1.5
        match = np.where(np.isclose(u2, u2_fmax, atol=tol))[0]

    idx1, idx2 = match[0], match[-1]
    u2_loop = u2[idx1:idx2 + 1]
    F_loop = F[idx1:idx2 + 1]
    energy = abs(np.trapezoid(F_loop, u2_loop)) / 1000.0  # N.mm -> J

    return F_max, u2_fmax, energy, (idx1, idx2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    u1, u2, F = analytical_curve()
    F_max, u2_fmax, energy, _ = released_energy(u1, u2, F)

    print(f"F_max = {F_max:.1f} N reached at u2 = {u2_fmax:.3f} mm")
    print(f"Energy released at the jump (area jump / equilibrium branch) = {energy:.3f} J")

    plt.figure(figsize=(8, 5))
    plt.plot(u2, F, 'r--', lw=2, label='Analytical (with snap-back)')
    plt.xlabel('u2 [mm]')
    plt.ylabel('Force F2 [N]')
    plt.title('Analytical solution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
