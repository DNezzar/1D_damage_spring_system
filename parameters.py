# -*- coding: utf-8 -*-
"""
All parameters of the study: physics, loading, static algorithm, dynamic
algorithm. Derived quantities (stiffnesses, uc, mass, time step...) are
computed at the bottom of this file; the other modules only read this file.

Consistent unit system: N, mm, s  =>  MPa = N/mm^2, mass in tonne
(1 t = 1000 kg), density in t/mm^3, energy in N.mm (= 1e-3 J).

Physical case: a concrete specimen in tension (spring 1, damageable)
mounted in series with a steel bar of the loading train (spring 2,
elastic). The steel bar stores elastic energy that the brittle specimen
cannot absorb -> snap-back.
"""

import numpy as np

# ===========================================================================
# 1. PHYSICAL PARAMETERS
# ===========================================================================

# --- Spring 1: concrete specimen (damageable) -------------------------------
E1 = 30000.0            # MPa   Young's modulus of concrete
A1 = 100.0 * 100.0      # mm^2  cross-section 100 x 100
L1 = 100.0              # mm    length
rho1 = 2.4e-9           # t/mm^3 density (2400 kg/m^3)
f_t = 3.0               # MPa   tensile strength -> sets the parameter uc
                        #       (peak force F_max = f_t * A1)

# --- Spring 2: steel bar, diameter 20 mm, of the loading train (elastic) ----
E2 = 210000.0           # MPa
A2 = np.pi * 10.0**2    # mm^2  diameter 20
L2 = 1000.0             # mm
rho2 = 7.85e-9          # t/mm^3 (7850 kg/m^3)

# --- Additional mass at node 1 (grips, fixture) ------------------------------
m_grip = 2.5e-2         # t  (25 kg; 2.5e-3 t would be 2.5 kg)

# --- Loading: imposed displacement at node 2 ---------------------------------
u2_f = 0.6              # mm    final displacement (beyond the limit point ~0.47 mm)

# Former "numerical" parameter set of the original scripts (not physical:
# uc = 5 mm on a 5 mm bar, 10 t mass), kept for reference:
#   E1, A1, L1 = 10000.0, 20.0, 5.0     -> k1 = 40000 N/mm
#   E2, A2, L2 = 20000.0, 30.0, 500.0   -> k2 = 1200 N/mm
#   uc = 5.0 mm ; m1 = 10.0 t ; u2_f = 80.0 mm

# ===========================================================================
# 2. STATIC ALGORITHM (quasi-static Newton-Raphson)
# ===========================================================================
inc = 60              # number of u2 increments between 0 and u2_f
eps_nr = 1e-6           # N     absolute tolerance on the residual |R| (shared static/dynamic)
max_iter = 30           # safeguard on the number of NR iterations per increment / time step
line_search = True      # static NR with a backtracking line search on |R| (True) or plain NR (False)
ls_c = 1e-4             # sufficient-decrease constant: alpha accepted if |R(u + alpha du)| <= (1 - ls_c alpha) |R(u)|
ls_max_halvings = 20    # alpha halved at most this many times (alpha_min ~ 1e-6); beyond, the NR is declared stalled

# ===========================================================================
# 3. DYNAMIC ALGORITHM (implicit Newmark + Newton-Raphson)
# ===========================================================================
beta_nm = 0.3           # Newmark beta   (0.25, 0.5 = average acceleration,
gamma_nm = 0.55         # Newmark gamma   unconditionally stable, no numerical damping)
N_per_period = 5        # time steps per natural period T0: dt = T0 / N_per_period.
                        # With beta = 1/4, the inertia/stiffness ratio in the tangent is
                        # m1*c1/(k1+k2) = N^2/pi^2: N = 5 -> 2.5 (regularises the
                        # non-monotonic static residual without flattening it;
                        # N = 40 -> 160, residual ~ straight line).
v2 = 1.0                # mm/s  ramp rate of the imposed displacement u2(t) = v2 * t
                        #       (real quasi-static test: 0.5 mm/min = 8.3e-3 mm/s,
                        #        i.e. ~1.4e6 time steps; 1 mm/s = realistic compromise)
t_hold = 10.0e-3        # s     hold of u2 = u2_f after the ramp (several periods of
                        #       post-rupture free oscillation 2*pi*sqrt(m1/k2) ~ 1.7 ms)

# ===========================================================================
# 4. DERIVED QUANTITIES (do not edit)
# ===========================================================================
k1 = E1 * A1 / L1                       # N/mm  intact stiffness of spring 1
k2 = E2 * A2 / L2                       # N/mm  stiffness of spring 2

# Damage law d = 1 - exp(-(u1/uc)^2): the peak force is
# F_max = k1*uc*exp(-1/2)/sqrt(2); uc is chosen so that F_max = f_t*A1.
F_max = f_t * A1                        # N
uc = F_max / (k1 * np.exp(-0.5) / np.sqrt(2.0))   # mm
G_f = k1 * uc**2 / 2.0 / A1             # N/mm  fracture energy (area under F1(u1) / A1)

m1 = 0.5 * rho1 * A1 * L1 + 0.5 * rho2 * A2 * L2 + m_grip   # t  lumped mass at node 1

omega0 = np.sqrt((k1 + k2) / m1)        # rad/s natural pulsation of node 1 (intact system)
T0 = 2.0 * np.pi / omega0               # s     natural period
dt = T0 / N_per_period                  # s     time step
T_ramp = u2_f / v2                      # s     ramp duration
n_steps = int(np.ceil(T_ramp / dt))     #       time steps of the ramp
n_hold = int(np.ceil(t_hold / dt))      #       time steps of the hold
c1 = 1.0 / (beta_nm * dt**2)            # 1/s^2 Newmark coefficient: m1*c1 is added to the tangent


def summary():
    """Print the summary of the parameters and derived quantities."""
    print("=" * 66)
    print("STUDY PARAMETERS (units N, mm, s, t)")
    print("=" * 66)
    print(f"Spring 1 (concrete): k1 = {k1:.4g} N/mm, uc = {uc:.4g} mm, F_max = {F_max:.4g} N, "
          f"G_f = {G_f:.3g} N/mm")
    print(f"Spring 2 (steel)   : k2 = {k2:.4g} N/mm, k2/k1 = {k2/k1:.3g} "
          f"(snap-back if < 0.446)")
    print(f"Mass at node 1     : m1 = {m1:.3g} t  ->  omega0 = {omega0:.4g} rad/s, "
          f"T0 = {T0*1e3:.4g} ms")
    print(f"Static             : {inc} increments, |R| <= {eps_nr:g} N, {max_iter} it. max, "
          f"line search {'on' if line_search else 'off'}")
    print(f"Dynamic            : Newmark beta={beta_nm}, gamma={gamma_nm}, dt = T0/{N_per_period} "
          f"= {dt*1e3:.4g} ms, m1*c1/(k1+k2) = {m1*c1/(k1+k2):.3g}")
    print(f"                     ramp v2 = {v2} mm/s -> {n_steps} steps ({T_ramp*1e3:.4g} ms), "
          f"hold {n_hold} steps ({t_hold*1e3:.4g} ms)")
    print("=" * 66)


if __name__ == "__main__":
    summary()
