# Projet Machine Learning - Classification de tumeurs mammaires

## 1. Probleme

### Contexte

Le cancer du sein est un cas d'usage classique de classification binaire : a partir de mesures cytologiques ordinales, l'objectif est de distinguer les tumeurs benignes des tumeurs malignes. Le projet utilise le jeu de donnees **Breast Cancer Wisconsin Original** provenant du UCI Machine Learning Repository. Contrairement a la version `scikit-learn` deja nettoyee, cette version est plus brute : elle n'a pas d'en-tetes, contient un identifiant patient, une cible codee `2/4`, et des valeurs manquantes encodees par `?`.

### Objectif metier

L'objectif metier est de fournir une aide a la decision pour prioriser les cas potentiellement malins. Dans ce contexte, le rappel de la classe `malignant` est critique : un faux negatif correspondrait a une tumeur maligne classee comme benigne. La precision et la specificite restent importantes pour limiter les alertes inutiles, mais le choix du meilleur modele privilegie le rappel des cas malins, puis l'AUC ROC et le F1-score.

## 2. Donnees

### Source

- Fichier brut local : `data/raw/breast-cancer-wisconsin.data`
- Source officielle : UCI Machine Learning Repository, Breast Cancer Wisconsin Original : https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
- Fichier de donnees UCI : https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
- Miroir Kaggle possible : https://www.kaggle.com/datasets/zzero0/uci-breast-cancer-wisconsin-original/data

### Description

Le jeu de donnees contient 699 observations et 9 variables explicatives ordinales notees de 1 a 10. La cible originale est codee `2 = benign` et `4 = malignant`; elle a ete recodee en `target_malignant`, avec `1 = malignant` et `0 = benign` pour aligner les metriques de classification sur le risque metier.

Repartition des classes :

| diagnosis | count | percentage |
| --- | --- | --- |
| benign | 458 | 0.6552217453505007 |
| malignant | 241 | 0.3447782546494993 |

### Limitations

- L'echantillon est limite : 699 observations ne suffisent pas a valider un usage clinique reel.
- Les variables sont des scores cytologiques deja extraits : on ne travaille pas sur les images ou examens originaux.
- Certaines valeurs sont manquantes et encodees par `?`, ce qui impose un nettoyage explicite.
- L'identifiant patient ne doit pas etre utilise comme signal predictif.
- Les observations proviennent d'un contexte precis ; une generalisation robuste demanderait une validation externe.

## 3. Analyse exploratoire des donnees

### Points cles

- Le fichier brut contient `16` valeurs manquantes dans les variables explicatives. Variables concernees : bare_nuclei: 16.
- `54` lignes partagent un identifiant deja present, ce qui suggere des visites ou enregistrements multiples. L'identifiant est conserve dans les exports mais exclu des variables explicatives.
- Les classes sont moderement desequilibrees : davantage de cas benins que malins.
- Les variables les plus correlees a la malignite sont : bare_nuclei (0.823), uniformity_cell_shape (0.819), uniformity_cell_size (0.818), bland_chromatin (0.757), clump_thickness (0.716), normal_nucleoli (0.712), marginal_adhesion (0.697), single_epithelial_cell_size (0.683).
- Plusieurs variables cytologiques, notamment l'uniformite cellulaire et les noyaux nus, separent visuellement les deux classes.

![Distribution des classes](../outputs/figures/01_distribution_classes.png)

![Distributions des variables cles](../outputs/figures/02_distributions_variables_cles.png)

![Correlation](../outputs/figures/03_correlation_heatmap.png)

![Boxplots](../outputs/figures/04_boxplots_variables_cles.png)

## 4. Preparation

### Nettoyage

- Verification des valeurs manquantes.
- Separation stricte train/test stratifiee : 559 lignes en entrainement et 140 lignes en test.
- Conversion des `?` en valeurs manquantes `NaN`.
- Conversion des colonnes ordinales en numerique.
- Exclusion de `sample_code_number` des variables predictives pour eviter d'apprendre un identifiant.
- Recodage de la cible `class` : `2 -> 0` et `4 -> 1`.
- Imputation mediane integree dans les pipelines pour traiter `bare_nuclei` sans fuite de donnees.
- Standardisation appliquee uniquement aux modeles sensibles aux echelles, dans un `Pipeline`, donc ajustee uniquement sur les folds d'entrainement pendant la validation croisee.

### Ingenierie des fonctionnalites

- Reencodage de la cible pour faire de `malignant` la classe positive.
- Conservation d'un fichier prepare dans `outputs/breast_cancer_wisconsin_original_prepared.csv` pour auditer le nettoyage.
- PCA non supervisee pour regarder la structure des donnees, et PCA integree dans un modele `PCA + regression logistique`.
- La reduction de dimension est placee apres le split et dans le pipeline lorsque le modele l'utilise, afin d'eviter les fuites de donnees.

## 5. Modelisation

### Plusieurs modeles

Les modeles compares sont :

- Regression logistique : baseline lineaire interpretable.
- K plus proches voisins : baseline non parametrique sensible aux distances.
- SVM RBF : modele non lineaire performant sur petits jeux de donnees tabulaires.
- Random Forest : methode d'ensemble obligatoire, robuste aux relations non lineaires.
- Gradient Boosting : seconde methode d'ensemble, souvent competitive sur donnees tabulaires.
- PCA + regression logistique : integration d'une reduction de dimension non supervisee dans un pipeline supervise.
- Random Forest optimise : recherche d'hyperparametres par validation croisee sur l'ensemble d'entrainement.

### Justification

Cette combinaison couvre des familles complementaires : lineaire, distance, marge, arbres en bagging, boosting et reduction de dimension. Les pipelines limitent les fuites de donnees car chaque transformation est ajustee dans la validation croisee ou sur l'entrainement uniquement.

## 6. Evaluation

### Metriques adaptees

- `recall_malignant` : prioritaire pour limiter les faux negatifs.
- `precision_malignant` : part des alertes malignes qui sont correctes.
- `specificity_benign` : capacite a reconnaitre les cas benins.
- `f1_malignant` : compromis precision/rappel.
- `roc_auc` : qualite globale de classement des scores.
- `accuracy` : information secondaire, car elle peut masquer les erreurs sur la classe critique.

### Validation croisee sur l'entrainement

| model | cv_accuracy_mean | cv_recall_malignant_mean | cv_f1_malignant_mean | cv_roc_auc_mean |
| --- | --- | --- | --- | --- |
| SVM RBF | 0.968 | 0.985 | 0.955 | 0.985 |
| Random Forest optimise | 0.973 | 0.984 | 0.962 | 0.990 |
| Regression logistique | 0.971 | 0.964 | 0.959 | 0.994 |
| Random Forest | 0.970 | 0.964 | 0.956 | 0.992 |
| PCA + regression logistique | 0.970 | 0.959 | 0.956 | 0.995 |
| K plus proches voisins | 0.970 | 0.959 | 0.956 | 0.993 |
| Gradient Boosting | 0.957 | 0.943 | 0.938 | 0.989 |

### Comparaison sur l'ensemble de test

| model | accuracy | precision_malignant | recall_malignant | specificity_benign | f1_malignant | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest optimise | 0.964 | 0.922 | 0.979 | 0.957 | 0.949 | 0.991 |
| SVM RBF | 0.957 | 0.920 | 0.958 | 0.957 | 0.939 | 0.987 |
| Regression logistique | 0.957 | 0.938 | 0.938 | 0.967 | 0.938 | 0.995 |
| PCA + regression logistique | 0.957 | 0.938 | 0.938 | 0.967 | 0.938 | 0.995 |
| Gradient Boosting | 0.957 | 0.938 | 0.938 | 0.967 | 0.938 | 0.993 |
| Random Forest | 0.957 | 0.938 | 0.938 | 0.967 | 0.938 | 0.991 |
| K plus proches voisins | 0.950 | 0.918 | 0.938 | 0.957 | 0.928 | 0.977 |

![Comparaison modeles](../outputs/figures/05_comparaison_modeles.png)

![Matrices de confusion](../outputs/figures/06_matrices_confusion.png)

![Courbes ROC](../outputs/figures/07_courbes_roc.png)

## 7. Analyse

### Interpretation des resultats

Le meilleur modele selon la regle metier est **Random Forest optimise**. Sur l'ensemble de test, il obtient :

- Accuracy : 0.964
- Precision malignant : 0.922
- Recall malignant : 0.979
- Specificite benign : 0.957
- F1 malignant : 0.949
- ROC AUC : 0.991

Le rappel eleve indique que le modele limite fortement le risque de manquer des cas malins sur le test. L'AUC ROC permet aussi de verifier que le modele classe globalement bien les observations, au-dela d'un seuil fixe.

La Random Forest optimisee a ete reglee avec :

```json
{
  "model__max_depth": null,
  "model__min_samples_leaf": 4,
  "model__n_estimators": 200
}
```

![Importance variables](../outputs/figures/08_importance_variables.png)

### Methode non supervisee

La PCA et KMeans ont ete ajustes uniquement sur l'ensemble d'entrainement.

- Variance expliquee par les 2 premieres composantes PCA : 0.742
- Nombre de composantes necessaires pour 95% de variance expliquee : 7
- ARI KMeans sur train : 0.814
- ARI KMeans sur test : 0.807
- Silhouette KMeans sur train : 0.574

KMeans retrouve partiellement la structure benign/malignant sans utiliser les labels, ce qui confirme qu'une partie du signal est visible dans l'espace des variables. Le score ARI reste imparfait, donc le clustering ne remplace pas la classification supervisee.

![PCA diagnostic](../outputs/figures/09_pca_diagnostic.png)

![PCA KMeans](../outputs/figures/10_pca_kmeans.png)

![Variance PCA](../outputs/figures/11_pca_variance.png)

## 8. Conclusion

### Recommandations

- Utiliser **Random Forest optimise** comme modele candidat, car il maximise le rappel de la classe maligne tout en conservant de bonnes performances globales.
- En contexte medical, ajuster le seuil de decision avec les experts metier pour controler explicitement le compromis faux negatifs / faux positifs.
- Completer cette analyse par une validation externe sur un autre centre ou une periode differente avant toute utilisation operationnelle.
- Conserver les pipelines `scikit-learn` pour garantir la reproductibilite et limiter les fuites de donnees.

### Limitations et travaux futurs

- Le test set est petit : les scores peuvent varier selon l'echantillonnage.
- Le dataset reste pedagogique meme s'il est plus brut que la version `scikit-learn`; un cas reel demanderait une gestion plus poussee de qualite, biais, valeurs aberrantes et derive temporelle.
- Les recommandations ne constituent pas un avis medical.
- Travaux futurs : calibration des probabilites, optimisation du seuil, validation externe, comparaison avec XGBoost/LightGBM si autorise, et analyse d'explicabilite plus complete.
