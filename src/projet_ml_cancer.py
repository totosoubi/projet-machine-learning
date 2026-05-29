from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = ROOT / "models"
REPORT_PATH = ROOT / "reports" / "rapport_projet_machine_learning.md"


def ensure_directories() -> None:
    for directory in [FIGURE_DIR, TABLE_DIR, MODEL_DIR, REPORT_PATH.parent]:
        directory.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = (dataset.target == 0).astype(int)
    target.name = "target_malignant"

    df = features.copy()
    df["diagnosis"] = np.where(target == 1, "malignant", "benign")
    df["target_malignant"] = target
    return df, features, target, list(features.columns)


def run_eda(df: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    missing = df[feature_names].isna().sum().sort_values(ascending=False)
    missing.to_csv(TABLE_DIR / "missing_values.csv", header=["missing_count"])

    class_counts = df["diagnosis"].value_counts().rename_axis("diagnosis").reset_index(name="count")
    class_counts["percentage"] = class_counts["count"] / class_counts["count"].sum()
    class_counts.to_csv(TABLE_DIR / "class_distribution.csv", index=False)

    description = df[feature_names].describe().T
    description.to_csv(TABLE_DIR / "feature_description.csv")

    corr_with_target = (
        df[feature_names + ["target_malignant"]]
        .corr(numeric_only=True)["target_malignant"]
        .drop("target_malignant")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    corr_with_target.to_csv(TABLE_DIR / "correlation_with_target.csv", header=["correlation"])

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="diagnosis", hue="diagnosis", palette=["#3f7f93", "#c44e52"], legend=False)
    plt.title("Distribution des classes")
    plt.xlabel("Diagnostic")
    plt.ylabel("Nombre d'observations")
    save_fig(FIGURE_DIR / "01_distribution_classes.png")

    top_features = corr_with_target.head(8).index.tolist()
    melted = df.melt(id_vars="diagnosis", value_vars=top_features, var_name="feature", value_name="value")
    g = sns.FacetGrid(melted, col="feature", col_wrap=4, hue="diagnosis", sharex=False, sharey=False, height=2.4)
    g.map_dataframe(sns.kdeplot, x="value", fill=True, common_norm=False, alpha=0.35)
    g.add_legend(title="Diagnostic")
    g.fig.suptitle("Distributions des variables les plus liees a la cible", y=1.04)
    save_fig(FIGURE_DIR / "02_distributions_variables_cles.png")

    selected_for_corr = top_features[:12]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[selected_for_corr + ["target_malignant"]].corr(), cmap="vlag", center=0, annot=False)
    plt.title("Carte de correlation des variables principales")
    save_fig(FIGURE_DIR / "03_correlation_heatmap.png")

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="diagnosis", palette=["#3f7f93", "#c44e52"])
    plt.xticks(rotation=35, ha="right")
    plt.title("Separation benign/malignant sur les variables les plus informatives")
    plt.xlabel("")
    plt.ylabel("Valeur")
    save_fig(FIGURE_DIR / "04_boxplots_variables_cles.png")

    return {
        "n_rows": int(df.shape[0]),
        "n_features": len(feature_names),
        "missing_total": int(missing.sum()),
        "class_counts": class_counts.to_dict(orient="records"),
        "top_correlations": corr_with_target.head(8).round(3).to_dict(),
        "top_features": top_features,
    }


def build_models(feature_names: list[str]) -> dict[str, Pipeline]:
    numeric_preprocess_scaled = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                feature_names,
            )
        ],
        remainder="drop",
    )
    numeric_preprocess_unscaled = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
    )

    return {
        "Regression logistique": Pipeline(
            [
                ("preprocess", numeric_preprocess_scaled),
                (
                    "model",
                    LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "K plus proches voisins": Pipeline(
            [
                ("preprocess", numeric_preprocess_scaled),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "SVM RBF": Pipeline(
            [
                ("preprocess", numeric_preprocess_scaled),
                ("model", SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", numeric_preprocess_unscaled),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocess", numeric_preprocess_unscaled),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "PCA + regression logistique": Pipeline(
            [
                ("preprocess", numeric_preprocess_scaled),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                (
                    "model",
                    LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def specificity_score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_malignant": precision_score(y_test, y_pred, zero_division=0),
        "recall_malignant": recall_score(y_test, y_pred, zero_division=0),
        "specificity_benign": specificity_score(y_test, y_pred),
        "f1_malignant": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, feature_names: list[str]) -> GridSearchCV:
    preprocess = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
            ),
        ]
    )
    grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        pipeline,
        grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="recall",
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def run_supervised_models(
    X: pd.DataFrame, y: pd.Series, feature_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], str, dict[str, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models(feature_names)
    tuned_rf = tune_random_forest(X_train, y_train, feature_names)
    models["Random Forest optimise"] = tuned_rf.best_estimator_

    scoring = {
        "accuracy": "accuracy",
        "precision_malignant": make_scorer(precision_score, zero_division=0),
        "recall_malignant": make_scorer(recall_score, zero_division=0),
        "f1_malignant": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        model_for_cv = clone(model)
        scores = cross_validate(model_for_cv, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_row: dict[str, Any] = {"model": name}
        for metric in scoring:
            cv_row[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            cv_row[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
        cv_rows.append(cv_row)

        fitted = clone(model).fit(X_train, y_train)
        fitted_models[name] = fitted
        test_metrics = evaluate_model(fitted, X_test, y_test)
        test_rows.append({"model": name, **test_metrics})

    cv_results = pd.DataFrame(cv_rows).sort_values(
        ["cv_recall_malignant_mean", "cv_roc_auc_mean", "cv_f1_malignant_mean"], ascending=False
    )
    test_results = pd.DataFrame(test_rows).sort_values(
        ["recall_malignant", "roc_auc", "f1_malignant"], ascending=False
    )
    cv_results.to_csv(TABLE_DIR / "cross_validation_results.csv", index=False)
    test_results.to_csv(TABLE_DIR / "test_results.csv", index=False)

    best_model_name = test_results.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    plot_model_comparison(test_results)
    plot_confusion_matrices(fitted_models, X_test, y_test)
    plot_roc_curves(fitted_models, X_test, y_test)
    feature_importance(best_model, best_model_name, X_test, y_test, feature_names)

    metadata = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "best_model": str(best_model_name),
        "best_params_random_forest": tuned_rf.best_params_,
        "best_cv_recall_random_forest": float(tuned_rf.best_score_),
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
    }
    (TABLE_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return cv_results, test_results, fitted_models, str(best_model_name), metadata


def plot_model_comparison(test_results: pd.DataFrame) -> None:
    metrics = ["recall_malignant", "specificity_benign", "f1_malignant", "roc_auc"]
    plot_df = test_results.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="score", y="model", hue="metric")
    plt.xlim(0, 1.02)
    plt.title("Comparaison des performances sur l'ensemble de test")
    plt.xlabel("Score")
    plt.ylabel("")
    plt.legend(title="Metrique", loc="lower right")
    save_fig(FIGURE_DIR / "05_comparaison_modeles.png")


def plot_confusion_matrices(fitted_models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    n_models = len(fitted_models)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, model) in zip(axes, fitted_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["benign", "malignant"],
            yticklabels=["benign", "malignant"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Reel")
    for ax in axes[len(fitted_models) :]:
        ax.axis("off")
    fig.suptitle("Matrices de confusion sur l'ensemble de test", y=1.02)
    save_fig(FIGURE_DIR / "06_matrices_confusion.png")


def plot_roc_curves(fitted_models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    plt.figure(figsize=(8, 6))
    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC sur l'ensemble de test")
    plt.legend(fontsize=8)
    save_fig(FIGURE_DIR / "07_courbes_roc.png")


def feature_importance(
    model: Pipeline, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list[str]
) -> None:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="recall",
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(TABLE_DIR / "permutation_importance_best_model.csv", index=False)

    top = importance.head(12).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4c78a8")
    plt.title(f"Importance par permutation - {model_name}")
    plt.xlabel("Baisse moyenne du rappel malignant")
    plt.ylabel("")
    save_fig(FIGURE_DIR / "08_importance_variables.png")


def run_unsupervised_analysis(X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preprocessing = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    X_train_scaled = preprocessing.fit_transform(X_train)
    X_test_scaled = preprocessing.transform(X_test)

    pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_scaled)
    explained = pd.DataFrame(
        {
            "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca_full.explained_variance_ratio_),
        }
    )
    explained.to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)

    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    train_pca = pca_2d.fit_transform(X_train_scaled)
    test_pca = pca_2d.transform(X_test_scaled)

    kmeans = KMeans(n_clusters=2, n_init=30, random_state=RANDOM_STATE)
    train_clusters = kmeans.fit_predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)

    unsupervised_metrics = {
        "pca_2_components_variance": float(pca_2d.explained_variance_ratio_.sum()),
        "pca_components_for_95_percent_variance": int((explained["cumulative_explained_variance"] < 0.95).sum() + 1),
        "kmeans_train_adjusted_rand_index": float(adjusted_rand_score(y_train, train_clusters)),
        "kmeans_test_adjusted_rand_index": float(adjusted_rand_score(y_test, test_clusters)),
        "kmeans_train_normalized_mutual_info": float(normalized_mutual_info_score(y_train, train_clusters)),
        "kmeans_test_normalized_mutual_info": float(normalized_mutual_info_score(y_test, test_clusters)),
        "kmeans_train_silhouette": float(silhouette_score(X_train_scaled, train_clusters)),
        "kmeans_test_silhouette": float(silhouette_score(X_test_scaled, test_clusters)),
    }
    (TABLE_DIR / "unsupervised_metrics.json").write_text(
        json.dumps(unsupervised_metrics, indent=2), encoding="utf-8"
    )

    pca_plot_df = pd.DataFrame(train_pca, columns=["PC1", "PC2"])
    pca_plot_df["diagnosis"] = np.where(y_train == 1, "malignant", "benign")
    pca_plot_df["cluster"] = train_clusters.astype(str)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=pca_plot_df, x="PC1", y="PC2", hue="diagnosis", palette=["#3f7f93", "#c44e52"], alpha=0.85)
    plt.title("Projection PCA 2D coloree par diagnostic reel")
    save_fig(FIGURE_DIR / "09_pca_diagnostic.png")

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=pca_plot_df, x="PC1", y="PC2", hue="cluster", palette="Set2", alpha=0.85)
    plt.title("Projection PCA 2D coloree par clusters KMeans")
    save_fig(FIGURE_DIR / "10_pca_kmeans.png")

    plt.figure(figsize=(8, 5))
    plt.plot(explained["component"], explained["cumulative_explained_variance"], marker="o")
    plt.axhline(0.95, color="#c44e52", linestyle="--", label="95%")
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquee cumulee")
    plt.title("Variance expliquee par PCA")
    plt.legend()
    save_fig(FIGURE_DIR / "11_pca_variance.png")

    pd.DataFrame(test_pca, columns=["PC1", "PC2"]).assign(
        diagnosis=np.where(y_test == 1, "malignant", "benign"), cluster=test_clusters
    ).to_csv(TABLE_DIR / "pca_test_projection.csv", index=False)

    return unsupervised_metrics


def format_float(value: float) -> str:
    return f"{value:.3f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    formatted = df[columns].copy()
    for col in formatted.columns:
        if col != "model":
            formatted[col] = formatted[col].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    return dataframe_to_markdown(formatted)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    rendered = df.astype(str)
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(row[col] for col in headers) + " |")
    return "\n".join(lines)


def build_report(
    eda_summary: dict[str, Any],
    cv_results: pd.DataFrame,
    test_results: pd.DataFrame,
    best_model_name: str,
    model_metadata: dict[str, Any],
    unsupervised_metrics: dict[str, Any],
) -> None:
    best_row = test_results[test_results["model"] == best_model_name].iloc[0]
    top_corr = ", ".join([f"{feature} ({value:.3f})" for feature, value in eda_summary["top_correlations"].items()])

    report = f"""# Projet Machine Learning - Classification de tumeurs mammaires

## 1. Probleme

### Contexte

Le cancer du sein est un cas d'usage classique de classification binaire : a partir de mesures numeriques calculees sur des noyaux cellulaires, l'objectif est de distinguer les tumeurs benignes des tumeurs malignes. Le projet utilise le jeu de donnees Breast Cancer Wisconsin Diagnostic, expose dans `scikit-learn` et provenant du UCI Machine Learning Repository.

### Objectif metier

L'objectif metier est de fournir une aide a la decision pour prioriser les cas potentiellement malins. Dans ce contexte, le rappel de la classe `malignant` est critique : un faux negatif correspondrait a une tumeur maligne classee comme benigne. La precision et la specificite restent importantes pour limiter les alertes inutiles, mais le choix du meilleur modele privilegie le rappel des cas malins, puis l'AUC ROC et le F1-score.

## 2. Donnees

### Source

- `sklearn.datasets.load_breast_cancer`
- Documentation scikit-learn : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Source originale : UCI Machine Learning Repository, Breast Cancer Wisconsin Diagnostic : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### Description

Le jeu de donnees contient {eda_summary["n_rows"]} observations et {eda_summary["n_features"]} variables explicatives numeriques. La cible a ete recodee en `target_malignant`, avec `1 = malignant` et `0 = benign` pour aligner les metriques de classification sur le risque metier.

Repartition des classes :

{dataframe_to_markdown(pd.DataFrame(eda_summary["class_counts"]))}

### Limitations

- L'echantillon est limite : 569 observations ne suffisent pas a valider un usage clinique reel.
- Les donnees sont propres et deja structurees, donc le projet ne couvre pas les problemes frequents de donnees hospitalieres brutes.
- Les observations proviennent d'un contexte precis ; une generalisation robuste demanderait une validation externe.
- Les variables sont derivees d'images, mais les images originales ne sont pas disponibles ici.

## 3. Analyse exploratoire des donnees

### Points cles

- Aucune valeur manquante n'a ete detectee dans les variables numeriques (`{eda_summary["missing_total"]}` valeurs manquantes).
- Les classes sont moderement desequilibrees : davantage de cas benins que malins.
- Les variables les plus correlees a la malignite sont : {top_corr}.
- Plusieurs variables de taille, concavite et texture separent visuellement les deux classes.

![Distribution des classes](../outputs/figures/01_distribution_classes.png)

![Distributions des variables cles](../outputs/figures/02_distributions_variables_cles.png)

![Correlation](../outputs/figures/03_correlation_heatmap.png)

![Boxplots](../outputs/figures/04_boxplots_variables_cles.png)

## 4. Preparation

### Nettoyage

- Verification des valeurs manquantes.
- Separation stricte train/test stratifiee : {model_metadata["train_rows"]} lignes en entrainement et {model_metadata["test_rows"]} lignes en test.
- Imputation mediane integree dans les pipelines, meme si aucune valeur manquante n'est observee, afin de rendre le pipeline robuste.
- Standardisation appliquee uniquement aux modeles sensibles aux echelles, dans un `Pipeline`, donc ajustee uniquement sur les folds d'entrainement pendant la validation croisee.

### Ingenierie des fonctionnalites

- Reencodage de la cible pour faire de `malignant` la classe positive.
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

{markdown_table(cv_results, ["model", "cv_accuracy_mean", "cv_recall_malignant_mean", "cv_f1_malignant_mean", "cv_roc_auc_mean"])}

### Comparaison sur l'ensemble de test

{markdown_table(test_results, ["model", "accuracy", "precision_malignant", "recall_malignant", "specificity_benign", "f1_malignant", "roc_auc"])}

![Comparaison modeles](../outputs/figures/05_comparaison_modeles.png)

![Matrices de confusion](../outputs/figures/06_matrices_confusion.png)

![Courbes ROC](../outputs/figures/07_courbes_roc.png)

## 7. Analyse

### Interpretation des resultats

Le meilleur modele selon la regle metier est **{best_model_name}**. Sur l'ensemble de test, il obtient :

- Accuracy : {format_float(best_row["accuracy"])}
- Precision malignant : {format_float(best_row["precision_malignant"])}
- Recall malignant : {format_float(best_row["recall_malignant"])}
- Specificite benign : {format_float(best_row["specificity_benign"])}
- F1 malignant : {format_float(best_row["f1_malignant"])}
- ROC AUC : {format_float(best_row["roc_auc"])}

Le rappel eleve indique que le modele limite fortement le risque de manquer des cas malins sur le test. L'AUC ROC permet aussi de verifier que le modele classe globalement bien les observations, au-dela d'un seuil fixe.

La Random Forest optimisee a ete reglee avec :

```json
{json.dumps(model_metadata["best_params_random_forest"], indent=2)}
```

![Importance variables](../outputs/figures/08_importance_variables.png)

### Methode non supervisee

La PCA et KMeans ont ete ajustes uniquement sur l'ensemble d'entrainement.

- Variance expliquee par les 2 premieres composantes PCA : {format_float(unsupervised_metrics["pca_2_components_variance"])}
- Nombre de composantes necessaires pour 95% de variance expliquee : {unsupervised_metrics["pca_components_for_95_percent_variance"]}
- ARI KMeans sur train : {format_float(unsupervised_metrics["kmeans_train_adjusted_rand_index"])}
- ARI KMeans sur test : {format_float(unsupervised_metrics["kmeans_test_adjusted_rand_index"])}
- Silhouette KMeans sur train : {format_float(unsupervised_metrics["kmeans_train_silhouette"])}

KMeans retrouve partiellement la structure benign/malignant sans utiliser les labels, ce qui confirme qu'une partie du signal est visible dans l'espace des variables. Le score ARI reste imparfait, donc le clustering ne remplace pas la classification supervisee.

![PCA diagnostic](../outputs/figures/09_pca_diagnostic.png)

![PCA KMeans](../outputs/figures/10_pca_kmeans.png)

![Variance PCA](../outputs/figures/11_pca_variance.png)

## 8. Conclusion

### Recommandations

- Utiliser **{best_model_name}** comme modele candidat, car il maximise le rappel de la classe maligne tout en conservant de bonnes performances globales.
- En contexte medical, ajuster le seuil de decision avec les experts metier pour controler explicitement le compromis faux negatifs / faux positifs.
- Completer cette analyse par une validation externe sur un autre centre ou une periode differente avant toute utilisation operationnelle.
- Conserver les pipelines `scikit-learn` pour garantir la reproductibilite et limiter les fuites de donnees.

### Limitations et travaux futurs

- Le test set est petit : les scores peuvent varier selon l'echantillonnage.
- Le dataset est pedagogique et deja nettoye ; un cas reel demanderait une gestion plus poussee de qualite, biais, valeurs aberrantes et derive temporelle.
- Les recommandations ne constituent pas un avis medical.
- Travaux futurs : calibration des probabilites, optimisation du seuil, validation externe, comparaison avec XGBoost/LightGBM si autorise, et analyse d'explicabilite plus complete.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> dict[str, Any]:
    ensure_directories()
    sns.set_theme(style="whitegrid", context="notebook")

    df, X, y, feature_names = load_data()
    df.to_csv(OUTPUT_DIR / "breast_cancer_wisconsin_prepared.csv", index=False)

    eda_summary = run_eda(df, feature_names)
    cv_results, test_results, fitted_models, best_model_name, model_metadata = run_supervised_models(
        X, y, feature_names
    )
    unsupervised_metrics = run_unsupervised_analysis(X, y, feature_names)
    build_report(eda_summary, cv_results, test_results, best_model_name, model_metadata, unsupervised_metrics)

    summary = {
        "best_model": best_model_name,
        "best_test_metrics": test_results[test_results["model"] == best_model_name].iloc[0].to_dict(),
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "model": str((MODEL_DIR / "best_model.joblib").relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
