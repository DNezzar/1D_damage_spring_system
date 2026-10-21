# 1D damage spring system

Snap-back d'un ressort endommageable en série avec un ressort élastique : solutions analytique, Newton-Raphson statique et Newmark dynamique, avec comparaison des résidus NR.

Modèle 1D à un degré de liberté : une éprouvette de béton (ressort endommageable, loi `d = 1 - exp(-(u/uc)²)`) en série avec une barre d'acier élastique, pilotée en déplacement. La raideur de la barre étant inférieure à la pente adoucissante du béton, la réponse force-déplacement présente un snap-back que le pilotage quasi-statique ne peut pas suivre. Le code compare trois solutions : la branche d'équilibre complète en forme fermée, un Newton-Raphson statique avec relance au saut, et un schéma de Newmark implicite qui convertit l'énergie libérée en énergie cinétique. Les figures montrent en particulier comment le terme d'inertie régularise le résidu non monotone du Newton-Raphson.

## Organisation

| Fichier | Rôle |
|---|---|
| `parametres.py` | tous les paramètres physiques et numériques, grandeurs dérivées (`python parametres.py` affiche le résumé) |
| `modele.py` | loi d'endommagement avec variable d'histoire, résidu et tangente statiques, Newton-Raphson |
| `solution_analytique.py` | branche d'équilibre paramétrée par `u1`, pic, bilan d'énergie du saut |
| `solution_statique.py` | incréments de `u2`, Newton-Raphson, relance au point limite |
| `solution_dynamique.py` | Newmark implicite prédicteur-correcteur, rampe puis maintien de `u2` |
| `comparaison.py` | assemble les trois solutions et produit les figures |
| `note_ressort_endommagement.tex` | note de calcul (équations, choix numériques, résultats) |
| `figures/` | figures produites par `comparaison.py --save` |

## Utilisation

```bash
pip install numpy matplotlib
python comparaison.py          # affiche les figures
python comparaison.py --save   # les enregistre aussi dans figures/
pdflatex note_ressort_endommagement.tex
```

Unités : N, mm, s, tonne.
