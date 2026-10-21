# -*- coding: utf-8 -*-
"""
Tous les paramètres de l'étude : physique, chargement, algorithme statique,
algorithme dynamique. Les grandeurs dérivées (raideurs, uc, masse, pas de
temps...) sont calculées en bas de ce fichier ; les autres modules ne
lisent que ce fichier.

Système d'unités cohérent : N, mm, s  =>  MPa = N/mm², masse en tonne
(1 t = 1000 kg), masse volumique en t/mm³, énergie en N.mm (= 1e-3 J).

Cas physique : éprouvette de béton en traction (ressort 1, endommageable)
montée en série avec une barre d'acier de la chaîne de chargement
(ressort 2, élastique). La barre d'acier stocke de l'énergie élastique
que l'éprouvette fragile ne peut pas absorber -> snap-back.
"""

import numpy as np

# ===========================================================================
# 1. PARAMÈTRES PHYSIQUES
# ===========================================================================

# --- Ressort 1 : éprouvette de béton (endommageable) ------------------------
E1 = 30000.0            # MPa   module d'Young du béton
A1 = 100.0 * 100.0      # mm²   section 100 x 100
L1 = 100.0              # mm    longueur
rho1 = 2.4e-9           # t/mm³ masse volumique (2400 kg/m³)
f_t = 3.0               # MPa   résistance en traction -> fixe le paramètre uc
                        #       (pic de force F_max = f_t * A1)

# --- Ressort 2 : barre d'acier Ø20 de la chaîne de chargement (élastique) ---
E2 = 210000.0           # MPa
A2 = np.pi * 10.0**2    # mm²   Ø20
L2 = 1000.0             # mm
rho2 = 7.85e-9          # t/mm³ (7850 kg/m³)

# --- Masse additionnelle au noeud 1 (mors, fixation) -------------------------
m_mors = 2.5e-2         # t  (2.5e-3=2,5 kg)

# --- Chargement : déplacement imposé au noeud 2 ------------------------------
u2_f = 0.6              # mm    déplacement final (au-delà du point limite ~0,47 mm)

# Ancien jeu "numérique" des scripts d'origine (pas physique : uc = 5 mm sur
# une barre de 5 mm, masse de 10 t), conservé pour référence :
#   E1, A1, L1 = 10000.0, 20.0, 5.0     -> k1 = 40000 N/mm
#   E2, A2, L2 = 20000.0, 30.0, 500.0   -> k2 = 1200 N/mm
#   uc = 5.0 mm ; m1 = 10.0 t ; u2_f = 80.0 mm

# ===========================================================================
# 2. ALGORITHME STATIQUE (Newton-Raphson quasi-statique)
# ===========================================================================
inc = 2000              # nombre d'incréments de u2 entre 0 et u2_f
eps_nr = 1e-6           # N     tolérance absolue sur le résidu |R| (commune statique/dynamique)
max_iter = 50           # garde-fou nombre d'itérations NR par incrément / pas de temps

# ===========================================================================
# 3. ALGORITHME DYNAMIQUE (Newmark implicite + Newton-Raphson)
# ===========================================================================
beta_nm = 0.3          # Newmark beta  (0.25, 0.5 = moyenne des accélérations,
gamma_nm = 0.55          # Newmark gamma  inconditionnellement stable, sans amortissement numérique)
N_par_periode = 5       # pas de temps par période propre T0 : dt = T0 / N_par_periode.
                        # Avec beta = 1/4, le rapport inertie/raideur dans la tangente vaut
                        # m1*c1/(k1+k2) = N²/pi² : N = 5 -> 2,5 (régularise la non-monotonie
                        # du résidu statique sans l'écraser ; N = 40 -> 160, résidu ~ droite).
v2 = 1.0                # mm/s  vitesse de la rampe de déplacement imposé u2(t) = v2 * t
                        #       (essai quasi-statique réel : 0,5 mm/min = 8,3e-3 mm/s,
                        #        soit ~1,4e6 pas de temps ; 1 mm/s = compromis réaliste)
t_hold = 10.0e-3        # s     maintien de u2 = u2_f après la rampe (plusieurs périodes
                        #       d'oscillation libre post-rupture 2*pi*sqrt(m1/k2) ~ 1,7 ms)

# ===========================================================================
# 4. GRANDEURS DÉRIVÉES (ne pas modifier)
# ===========================================================================
k1 = E1 * A1 / L1                       # N/mm  raideur ressort 1 sain
k2 = E2 * A2 / L2                       # N/mm  raideur ressort 2

# Loi d'endommagement d = 1 - exp(-(u1/uc)²) : le pic de force vaut
# F_max = k1*uc*exp(-1/2)/sqrt(2) ; on choisit uc pour que F_max = f_t*A1.
F_max = f_t * A1                        # N
uc = F_max / (k1 * np.exp(-0.5) / np.sqrt(2.0))   # mm
G_f = k1 * uc**2 / 2.0 / A1             # N/mm  énergie de rupture (aire sous F1(u1) / A1)

m1 = 0.5 * rho1 * A1 * L1 + 0.5 * rho2 * A2 * L2 + m_mors   # t  masse concentrée au noeud 1

omega0 = np.sqrt((k1 + k2) / m1)        # rad/s pulsation propre du noeud 1 (système sain)
T0 = 2.0 * np.pi / omega0               # s     période propre
dt = T0 / N_par_periode                 # s     pas de temps
T_ramp = u2_f / v2                      # s     durée de la rampe
n_steps = int(np.ceil(T_ramp / dt))     #       pas de temps de la rampe
n_hold = int(np.ceil(t_hold / dt))      #       pas de temps de maintien
c1 = 1.0 / (beta_nm * dt**2)            # 1/s²  coefficient Newmark : m1*c1 s'ajoute à la tangente


def resume():
    """Affiche le résumé des paramètres et des grandeurs dérivées."""
    print("=" * 66)
    print("PARAMÈTRES DE L'ÉTUDE (unités N, mm, s, t)")
    print("=" * 66)
    print(f"Ressort 1 (béton) : k1 = {k1:.4g} N/mm, uc = {uc:.4g} mm, F_max = {F_max:.4g} N, "
          f"G_f = {G_f:.3g} N/mm")
    print(f"Ressort 2 (acier) : k2 = {k2:.4g} N/mm, k2/k1 = {k2/k1:.3g} "
          f"(snap-back si < 0.446)")
    print(f"Masse noeud 1     : m1 = {m1:.3g} t  ->  omega0 = {omega0:.4g} rad/s, "
          f"T0 = {T0*1e3:.4g} ms")
    print(f"Statique          : {inc} incréments, |R| <= {eps_nr:g} N, {max_iter} it. max")
    print(f"Dynamique         : Newmark beta={beta_nm}, gamma={gamma_nm}, dt = T0/{N_par_periode} "
          f"= {dt*1e3:.4g} ms, m1*c1/(k1+k2) = {m1*c1/(k1+k2):.3g}")
    print(f"                    rampe v2 = {v2} mm/s -> {n_steps} pas ({T_ramp*1e3:.4g} ms), "
          f"maintien {n_hold} pas ({t_hold*1e3:.4g} ms)")
    print("=" * 66)


if __name__ == "__main__":
    resume()
