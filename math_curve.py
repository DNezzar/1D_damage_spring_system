"""
Graphe interactif du ressort endommageable (ressort 1) en série avec un
ressort élastique (ressort 2), u2 imposé.

Trois panneaux :
  1. d(u1) = 1 - exp(-(u1/uc)^2)                 loi d'endommagement
  2. F1(u1) = k1 (1 - d) u1                       réponse du ressort 1 seul
  3. F(u2) avec u2 = u1 (1 + k1 (1 - d)/k2)       réponse du système (snap-back)

Un seul curseur : le rapport k2/k1. uc et k1 sont lus dans parameters.py.

Lancer avec : ../.venv/bin/python math_curve.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.optimize import brentq

import parameters as prm

# ---------------------------------------------------------------------------
# Étape 1 : les fonctions du modèle
# ---------------------------------------------------------------------------
# dF1/du1 = k1 e^{-s^2} (1 - 2 s^2), s = u1/uc
#   = 0            en s = 1/sqrt(2)      -> pic : F_max = k1 uc e^{-1/2}/sqrt(2)
#   minimale       en s = sqrt(3/2)      -> dF1/du1 = -2 e^{-3/2} k1 = -R_STAR k1
#
# Équilibre du noeud 1 : u2 = u1 (1 + k1 (1-d)/k2)
#   du2/du1 = 1 + (1/k2) dF1/du1 s'annule (tangente verticale = snap-back)
#   seulement si min(dF1/du1) < -k2, c.-à-d. si k2 < R_STAR k1.
R_STAR = 2.0 * np.exp(-1.5)  # ≈ 0.446 : rapport k2/k1 critique


def damage(u1):
    return 1.0 - np.exp(-((u1 / uc) ** 2))


def force1(u1):
    return k1 * (1.0 - damage(u1)) * u1


def dforce1(u1):
    """dF1/du1 = k1 e^{-s^2} (1 - 2 s^2)."""
    s = u1 / uc
    return k1 * np.exp(-s**2) * (1.0 - 2.0 * s**2)


def u2_of_u1(u1, k2):
    return u1 * (1.0 + k1 * (1.0 - damage(u1)) / k2)


def limit_point_u1(k2):
    """u1 du point limite, racine de dF1/du1 = -k2 entre le pic et la pente min.
    None si k2 >= R_STAR k1 (pas de tangence possible)."""
    if k2 >= R_STAR * k1:
        return None
    return brentq(lambda u: dforce1(u) + k2, uc / np.sqrt(2.0), uc * np.sqrt(1.5))


def limit_point(u2):
    """Premier point où u2 cesse de croître (du2/du1 = 0). None s'il n'y en a pas."""
    idx = np.where(np.diff(u2) < 0)[0]
    return None if len(idx) == 0 else idx[0]


# ---------------------------------------------------------------------------
# Étape 2 : paramètres fixes et courbes du ressort 1 (indépendantes de k2)
# ---------------------------------------------------------------------------
uc, k1 = prm.uc, prm.k1
r0 = 0.2  # valeur initiale du curseur (cas étudié : prm.k2/prm.k1 ≈ 0.11)
N = 2000

u1 = np.linspace(0.0, 4.0 * uc, N)
d = damage(u1)
f1 = force1(u1)

# Valeurs clés liées à uc
u1_pk = uc / np.sqrt(2.0)  # pic de F1
f_pk = force1(u1_pk)
u1_sb = uc * np.sqrt(1.5)  # pente minimale de F1 : -R_STAR k1
f_sb = force1(u1_sb)

# Les trois cas de référence du panneau 3, en multiples du rapport critique
CASES = [
    (0.5 * R_STAR, "tab:red", "k2 < 0.446 k1 : snap-back"),
    (1.0 * R_STAR, "tab:orange", "k2 = 0.446 k1 : limite (tangente verticale)"),
    (2.0 * R_STAR, "tab:green", "k2 > 0.446 k1 : adoucissement sans snap-back"),
]

# ---------------------------------------------------------------------------
# Étape 3 : la figure et les trois axes
# ---------------------------------------------------------------------------
fig, (ax_d, ax_f1, ax_f2) = plt.subplots(1, 3, figsize=(16, 6.5))
fig.subplots_adjust(bottom=0.25, top=0.90, wspace=0.30, left=0.06, right=0.98)

# --- Panneau 1 : d(u1), statique ---------------------------------------------
ax_d.set_title("Endommagement")
ax_d.set_xlabel("u1 [mm]")
ax_d.set_ylabel("d")
ax_d.set_ylim(-0.05, 1.05)
ax_d.grid(alpha=0.3)
ax_d.plot(u1, d, "tab:blue", lw=2, label="d = 1 - exp(-(u1/uc)²)")
ax_d.plot([u1_pk], [damage(u1_pk)], "ko", label=f"pic : u1 = uc/√2  (d = {damage(u1_pk):.2f})")
ax_d.plot([uc], [damage(uc)], "ks", mfc="none", label=f"u1 = uc  (d = {damage(uc):.2f})")
ax_d.plot([u1_sb], [damage(u1_sb)], "k^", label=f"u1 = uc√(3/2)  (d = {damage(u1_sb):.2f})")
ax_d.legend(fontsize=8, loc="lower right")

# --- Panneau 2 : F1(u1), statique --------------------------------------------
ax_f1.set_title("Ressort 1 : F1(u1)")
ax_f1.set_xlabel("u1 [mm]")
ax_f1.set_ylabel("F1 [N]")
ax_f1.grid(alpha=0.3)
ax_f1.plot(u1, f1, "tab:blue", lw=2, label="F1 = k1 (1-d) u1")
half = 0.4 * uc
u_tan = np.array([u1_sb - half, u1_sb + half])
ax_f1.plot(u_tan, f_sb - R_STAR * k1 * (u_tan - u1_sb), "k--", lw=1,
           label="pente min = -0.446 k1 (en u1 = uc√(3/2))")
ax_f1.plot([u1_pk], [f_pk], "ko", label="pic : u1 = uc/√2")
ax_f1.plot([uc], [force1(uc)], "ks", mfc="none", label="u1 = uc  (d = 0.63)")
ax_f1.plot([u1_sb], [f_sb], "k^", label="u1 = uc√(3/2)")
(l_k2,) = ax_f1.plot([], [], "k-", lw=1)  # droite de pente -k2 (suit le curseur)
(m_f1_lim,) = ax_f1.plot([], [], "kv", ms=8, label="point limite : F1' = -k2")
ax_f1.set_xlim(0, u1[-1])
ax_f1.set_ylim(0, 1.15 * f_pk)
ax_f1.legend(fontsize=8, loc="upper right")

# --- Panneau 3 : F(u2), dépend de k2 -----------------------------------------
ax_f2.set_title("Système : F(u2), courbe analytique")
ax_f2.set_xlabel("u2 [mm]")
ax_f2.set_ylabel("F [N]")
ax_f2.grid(alpha=0.3)
for r_case, color, label in CASES:
    ax_f2.plot(u2_of_u1(u1, r_case * k1), f1, color=color, lw=1.2, alpha=0.8, label=label)
(l_cur,) = ax_f2.plot([], [], "k", lw=2.2)
(m_f2_pk,) = ax_f2.plot([], [], "ko")
(m_f2_lim,) = ax_f2.plot([], [], "kv", label="point limite (du2/du1 = 0)")
v_jump = ax_f2.axvline(0, color="k", ls=":", lw=1)
ax_f2.set_ylim(0, 1.15 * f_pk)

# Valeurs clés
txt = fig.text(0.70, 0.12, "", fontsize=9, family="monospace", va="top")

# ---------------------------------------------------------------------------
# Étape 4 : le curseur k2/k1
# ---------------------------------------------------------------------------
ax_s = fig.add_axes([0.12, 0.08, 0.5, 0.03])
s_r = Slider(ax_s, "k2 / k1", 0.02, 1.5, valinit=r0)


# ---------------------------------------------------------------------------
# Étape 5 : la mise à jour (seul le panneau 3 dépend de k2)
# ---------------------------------------------------------------------------
def update(_=None):
    r = s_r.val
    k2 = r * k1
    u2 = u2_of_u1(u1, k2)

    # --- panneau 2 : droite de pente -k2
    u1_lim = limit_point_u1(k2)
    if u1_lim is None:
        # pas de tangence : la droite passe par le point de pente minimale,
        # elle y est plus raide que la courbe
        u_a, f_a = u1_sb, f_sb
        l_k2.set_label("pente -k2 (plus raide que F1 : pas de tangence)")
        m_f1_lim.set_data([], [])
    else:
        u_a, f_a = u1_lim, force1(u1_lim)
        l_k2.set_label("F = k2 (u2_lim - u1), tangente au point limite")
        m_f1_lim.set_data([u1_lim], [f_a])
    u_line = np.array([max(0.0, u_a - 0.6 * uc), u_a + f_a / k2])  # jusqu'à F = 0
    l_k2.set_data(u_line, f_a - k2 * (u_line - u_a))
    ax_f1.set_xlim(0, max(u1[-1], 1.03 * u_line[1]))
    ax_f1.legend(fontsize=8, loc="upper right")

    l_cur.set_data(u2, f1)
    l_cur.set_label(f"k2 = {r:.3f} k1 (curseur)")
    m_f2_pk.set_data([u2_of_u1(u1_pk, k2)], [f_pk])

    i_lim = limit_point(u2)
    if i_lim is None:
        m_f2_lim.set_data([], [])
        v_jump.set_visible(False)
        lim_txt = "pas de point limite (u2 monotone)"
    else:
        m_f2_lim.set_data([u2[i_lim]], [f1[i_lim]])
        v_jump.set_xdata([u2[i_lim], u2[i_lim]])
        v_jump.set_visible(True)
        lim_txt = f"point limite : u2 = {u2[i_lim]:.4g} mm, F = {f1[i_lim]:.4g} N"
    ax_f2.legend(fontsize=8, loc="upper right")
    # Le cadre suit la courbe la plus étendue (cas rouge ou curseur)
    ax_f2.set_xlim(0, 1.05 * max(u2.max(), u2_of_u1(u1, CASES[0][0] * k1).max()))

    regime = ("SNAP-BACK" if r < R_STAR else
              "LIMITE" if np.isclose(r, R_STAR, rtol=1e-3) else "SANS SNAP-BACK")
    txt.set_text(
        f"uc = {uc:.4g} mm   k1 = {k1:.4g} N/mm   k2 = {k2:.4g} N/mm\n"
        f"k2/k1 = {r:.3f}  (critique {R_STAR:.3f})  -> {regime}\n"
        f"pic : u1 = uc/√2 = {u1_pk:.4g} mm, F_max = {f_pk:.4g} N\n"
        f"pente min : u1 = uc√(3/2) = {u1_sb:.4g} mm, dF1/du1 = {-R_STAR * k1:.4g} N/mm\n"
        f"{lim_txt}"
    )
    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Étape 6 : brancher le curseur et afficher
# ---------------------------------------------------------------------------
s_r.on_changed(update)
update()
plt.show()
