# -*- coding: utf-8 -*-
"""
Figure de la note note_snapback_raideurs.tex : point limite et critère de
snap-back selon le rapport k2/k1. Produit figures/snapback_raideurs.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import parameters as prm

uc, k1, k2 = prm.uc, prm.k1, prm.k2
R_STAR = 2.0 * np.exp(-1.5)


def damage(u1):
    return 1.0 - np.exp(-((u1 / uc) ** 2))


def force1(u1):
    return k1 * (1.0 - damage(u1)) * u1


def dforce1(u1):
    s = u1 / uc
    return k1 * np.exp(-s**2) * (1.0 - 2.0 * s**2)


def u2_of_u1(u1, k2):
    return u1 + force1(u1) / k2


def limit_point(k2):
    """u1 du point limite (dF1/du1 = -k2), None si k2 >= R_STAR k1."""
    if k2 >= R_STAR * k1:
        return None
    return brentq(lambda u: dforce1(u) + k2, uc / np.sqrt(2.0), uc * np.sqrt(1.5))


u1 = np.linspace(0.0, 5.0 * uc, 4000)
f1 = force1(u1)
u1_pk, u1_sb = uc / np.sqrt(2.0), uc * np.sqrt(1.5)
f_pk, f_sb = force1(u1_pk), force1(u1_sb)
r0 = k2 / k1
u1_lim = limit_point(k2)
u2_lim = u2_of_u1(u1_lim, k2)
f_lim = force1(u1_lim)

CASES = [
    (0.5 * R_STAR, "tab:red", r"$k_2 = 0.223\,k_1$"),
    (1.0 * R_STAR, "tab:orange", r"$k_2 = 0.446\,k_1$ (limite)"),
    (2.0 * R_STAR, "tab:green", r"$k_2 = 0.893\,k_1$"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# (a) F1(u1) : tangente de pente minimale et droite de pente -k2 au point limite
ax1.plot(u1, f1, "tab:blue", lw=2, label=r"$F_1(u_1) = k_1(1-d)\,u_1$")
half = 0.45 * uc
ut = np.array([u1_sb - half, u1_sb + half])
ax1.plot(ut, f_sb + dforce1(u1_sb) * (ut - u1_sb), "k--", lw=1,
         label=r"pente minimale $-2e^{-3/2}k_1$ en $u_1 = u_c\sqrt{3/2}$")
ul = np.array([u1_lim - 0.6 * uc, u2_lim])
ax1.plot(ul, -k2 * (ul - u2_lim), "k-", lw=1,
         label=r"droite $F = k_2(u_2^{\lim}-u_1)$, pente $-k_2$")
ax1.plot([u1_pk], [f_pk], "ko", label=r"pic, $u_1 = u_c/\sqrt{2}$")
ax1.plot([u1_lim], [f_lim], "kv", ms=8, label=r"point limite, $F_1' = -k_2$")
ax1.plot([u1_sb], [f_sb], "k^", label=r"$u_1 = u_c\sqrt{3/2}$")
ax1.plot([u2_lim], [0], "k|", ms=10)
ax1.annotate(r"$u_2^{\lim}$", (u2_lim, 0), textcoords="offset points", xytext=(4, 6))
ax1.set_xlim(0, u1[-1])
ax1.set_ylim(0, 1.12 * f_pk)
ax1.set_xlabel(r"$u_1$ [mm]")
ax1.set_ylabel(r"$F_1$ [N]")
ax1.set_title(r"(a) ressort 1 seul, $k_2/k_1 = %.3f$" % r0)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=7, loc="upper right")

# (b) F(u2) pour les trois cas de référence et le cas étudié
for r, color, label in CASES:
    ax2.plot(u2_of_u1(u1, r * k1), f1, color=color, lw=1.3, label=label)
    ul_ = limit_point(r * k1)
    if ul_ is not None:
        ax2.plot([u2_of_u1(ul_, r * k1)], [force1(ul_)], "v", color=color, ms=7)
ax2.plot(u2_of_u1(u1, k2), f1, "k", lw=2.2, label=r"cas étudié, $k_2 = %.3f\,k_1$" % r0)
ax2.plot([u2_lim], [f_lim], "kv", ms=8)
ax2.vlines(u2_lim, 0, f_lim, color="k", ls=":", lw=1)
ax2.annotate("saut", (u2_lim, 0.5 * f_lim), textcoords="offset points", xytext=(5, 0))
ax2.set_xlim(0, 1.05 * u2_of_u1(u1, k2).max())
ax2.set_ylim(0, 1.45 * f_pk)
ax2.set_xlabel(r"$u_2$ [mm]")
ax2.set_ylabel(r"$F$ [N]")
ax2.set_title(r"(b) système, $u_2 = u_1 + F_1/k_2$")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=7, loc="upper right")

fig.tight_layout()
fig.savefig("figures/snapback_raideurs.pdf")
fig.savefig("figures/snapback_raideurs.png", dpi=110)

if __name__ == "__main__":
    import analytical_solution as ana
    u1a, u2a, Fa = ana.analytical_curve()
    _, _, energy, _ = ana.released_energy(u1a, u2a, Fa)
    k2c = R_STAR * k1
    mc1 = prm.m1 * prm.c1
    print(f"uc = {uc:.5g} mm, k1 = {k1:.6g} N/mm, k2 = {k2:.6g} N/mm, k2/k1 = {r0:.4f}")
    print(f"R* = {R_STAR:.4f}, k2 critique = {k2c:.6g} N/mm, marge k2c/k2 = {k2c/k2:.3f}")
    print(f"u1 pic = {u1_pk:.5g} mm, F_max = {f_pk:.6g} N, u2 pic = {u2_of_u1(u1_pk,k2):.5g} mm")
    print(f"u1 pente min = {u1_sb:.5g} mm, pente = {dforce1(u1_sb):.6g} N/mm")
    print(f"point limite : u1 = {u1_lim:.5g} mm, u2 = {u2_lim:.5g} mm, F = {f_lim:.6g} N, d = {damage(u1_lim):.4f}")
    u1_b = brentq(lambda u: dforce1(u) + k2, uc * np.sqrt(1.5), 6 * uc)
    print(f"second passage : u1 = {u1_b:.5g} mm, u2 = {u2_of_u1(u1_b, k2):.5g} mm, F = {force1(u1_b):.5g} N")
    print(f"L2 critique (meme barre) = {prm.L2*k2/k2c:.4g} mm ; diametre critique (meme L2) = "
          f"{20*np.sqrt(k2c/k2):.4g} mm")
    print(f"energie liberee = {energy:.4g} J ; energie elastique acier au pt limite = "
          f"{0.5*f_lim**2/k2/1000:.4g} J")
    print(f"m1*c1 = {mc1:.6g} N/mm = {mc1/k1:.3f} k1 ; (k2 + m1 c1)/k1 = {(k2+mc1)/k1:.3f}")
    print(f"dt = {prm.dt*1e3:.4g} ms, beta = {prm.beta_nm}, N = {prm.N_per_period}")
