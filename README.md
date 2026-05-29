# Projet Machine Learning

Projet complet de classification binaire sur le jeu de donnees Breast Cancer Wisconsin Original, utilise ici dans une version plus brute que le dataset integre a `scikit-learn`.

## Objectif

Predire si une tumeur est maligne ou benigne a partir de 9 variables cytologiques ordinales. Le fichier source contient des donnees sans en-tetes, une cible codee `2/4`, un identifiant patient et des valeurs manquantes encodees par `?`.

Le projet respecte les contraintes demandees :

- Tache de classification.
- Methode d'ensemble : Random Forest et Gradient Boosting.
- Methode non supervisee : PCA et KMeans.
- Pipeline propre avec split train/test stratifie, imputation dans `Pipeline`, preprocessing dans `Pipeline`, validation croisee et absence de fuite de donnees.
- Comparaison de plusieurs modeles.
- Analyse du seuil de decision pour expliciter le compromis faux negatifs / faux positifs.
- Rapport structure selon les sections demandees.

## Structure

- `data/raw/breast-cancer-wisconsin.data` : fichier brut UCI.
- `src/projet_ml_cancer.py` : script reproductible principal.
- `notebooks/projet_machine_learning_cancer.ipynb` : notebook executable.
- `reports/rapport_projet_machine_learning.md` : rapport final en francais.
- `outputs/figures/` : graphiques EDA, evaluation, PCA et clustering.
- `outputs/tables/` : resultats CSV/JSON.
- `outputs/breast_cancer_wisconsin_original_prepared.csv` : export prepare et auditable.
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

- UCI Machine Learning Repository : https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
- Fichier brut : https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
- Miroir Kaggle : https://www.kaggle.com/datasets/zzero0/uci-breast-cancer-wisconsin-original/data
