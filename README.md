# Projet Machine Learning

Projet complet de classification binaire sur le jeu de donnees Breast Cancer Wisconsin Diagnostic.

## Objectif

Predire si une tumeur est maligne ou benigne a partir de 30 variables numeriques calculees sur des noyaux cellulaires.

Le projet respecte les contraintes demandees :

- Tache de classification.
- Methode d'ensemble : Random Forest et Gradient Boosting.
- Methode non supervisee : PCA et KMeans.
- Pipeline propre avec split train/test stratifie, preprocessing dans `Pipeline`, validation croisee et absence de fuite de donnees.
- Comparaison de plusieurs modeles.
- Rapport structure selon les sections demandees.

## Structure

- `src/projet_ml_cancer.py` : script reproductible principal.
- `notebooks/projet_machine_learning_cancer.ipynb` : notebook executable.
- `reports/rapport_projet_machine_learning.md` : rapport final en francais.
- `outputs/figures/` : graphiques EDA, evaluation, PCA et clustering.
- `outputs/tables/` : resultats CSV/JSON.
- `models/best_model.joblib` : meilleur modele sauvegarde.

## Execution

```bash
python3 src/projet_ml_cancer.py
```

Execution du notebook :

```bash
python3 -m nbconvert --to notebook --execute notebooks/projet_machine_learning_cancer.ipynb --output projet_machine_learning_cancer_execute.ipynb
```

## Source des donnees

- scikit-learn `load_breast_cancer` : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- UCI Machine Learning Repository : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
