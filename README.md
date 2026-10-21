# 1D damage spring system

Snap-back of a damageable spring in series with an elastic one: closed-form, static Newton-Raphson and dynamic Newmark solutions, with a comparison of the Newton-Raphson residuals.

One-degree-of-freedom 1D model: a concrete specimen (damageable spring, law `d = 1 - exp(-(u/uc)²)`) in series with an elastic steel bar, under displacement control. Since the bar stiffness is lower than the softening slope of the concrete, the force-displacement response shows a snap-back that quasi-static displacement control cannot follow. The code compares three solutions: the complete equilibrium branch in closed form, a static Newton-Raphson with a restart at the jump, and an implicit Newmark scheme that converts the released energy into kinetic energy. The figures show in particular how the inertia term regularises the non-monotonic Newton-Raphson residual.

## Layout

| File | Role |
|---|---|
| `parameters.py` | all physical and numerical parameters, derived quantities (`python parameters.py` prints the summary) |
| `model.py` | damage law with history variable, static residual and tangent, Newton-Raphson |
| `analytical_solution.py` | equilibrium branch parametrised by `u1`, peak, energy balance of the jump |
| `static_solution.py` | `u2` increments, Newton-Raphson, restart at the limit point |
| `dynamic_solution.py` | implicit predictor-corrector Newmark, ramp then hold of `u2` |
| `comparison.py` | assembles the three solutions and produces the figures |
| `note_damage_spring.tex` | calculation note: derivation from the mass-spring system to the residuals, closed-form branch, numerical choices, results |
| `note_nonlinear_dynamics.tex` | Newmark scheme in predictor-corrector form and Newton-Raphson for nonlinear dynamics (residual, consistent tangent, algorithm) |
| `explication.tex` | step-by-step pedagogical note (in French): damage law, graphical equilibrium construction, snap-back criterion in stiffness and in bar lengths, closed-form branch, why Newton-Raphson cycles at the limit point and how inertia regularises it, with hand-calculation boxes |
| `figures/` | figures produced by `comparison.py --save` |

## Usage

```bash
pip install numpy matplotlib
python comparison.py          # shows the figures
python comparison.py --save   # also saves them in figures/
pdflatex note_damage_spring.tex
```

Units: N, mm, s, tonne.
