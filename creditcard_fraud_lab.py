"""
Здесь реализована классификация транзакций на легальные и мошеннические.
Производится обучение и настройка четырёх моделей:
Bagging, Random Forest, LightGBM и CatBoost.

Выполняется подбор гиперпараметров по метрике Macro F1,
сравнение моделей на тестовой выборке и построение
матрицы ошибок для лучшей модели.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

RANDOM_STATE = 42
TARGET = "Class"
THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


class ProbabilityThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a probabilistic binary classifier and expose a tunable threshold.

    This makes the threshold a regular hyperparameter, so RandomizedSearchCV
    optimizes it using macro F1 inside the training folds rather than using the
    final test set.
    """

    def __init__(self, estimator: BaseEstimator, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "ProbabilityThresholdClassifier":
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = np.asarray(self.estimator_.classes_)

        positive_indices = np.flatnonzero(self.classes_ == 1)
        if len(positive_indices) != 1 or len(self.classes_) != 2:
            raise ValueError("ProbabilityThresholdClassifier expects binary labels 0 and 1.")
        self.positive_index_ = int(positive_indices[0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["estimator_", "classes_"])
        return self.estimator_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, self.positive_index_]
        negative_label = self.classes_[1 - self.positive_index_]
        positive_label = self.classes_[self.positive_index_]
        return np.where(probabilities >= float(self.threshold), positive_label, negative_label)


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read and validate creditcard.csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}. Скачайте creditcard.csv и положите его в папку data/."
        )

    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"В CSV отсутствует целевой столбец {TARGET!r}.")
    if df[TARGET].isna().any():
        raise ValueError("В целевом столбце Class есть пропуски.")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Ожидаются ровно два значения целевого столбца: 0 и 1.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype("int8")

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Обнаружены нечисловые признаки: {non_numeric}")
    if X.isna().any().any():
        raise ValueError("В признаках есть пропуски. Проверьте исходный CSV перед обучением.")

    return X, y


def build_model_spaces() -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    """Create all required models and their randomized-search spaces."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("Не установлен LightGBM. Выполните: pip install lightgbm") from exc

    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Не установлен CatBoost. Выполните: pip install catboost") from exc

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        n_estimators=30,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    lightgbm = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        verbosity=-1,
        force_col_wise=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=1.0,
    )

    catboost = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        random_seed=RANDOM_STATE,
        scale_pos_weight=1.0,
    )

    return {
        "Bagging": (
            bagging,
            {
                "estimator__estimator__max_depth": [6, 10, None],
                "estimator__estimator__min_samples_leaf": [2, 5, 10],
                "estimator__n_estimators": [20, 40, 60],
                "estimator__max_samples": [0.5, 0.8, 1.0],
                "estimator__max_features": [0.7, 1.0],
            },
        ),
        "Random Forest": (
            random_forest,
            {
                "estimator__n_estimators": [150, 250, 350],
                "estimator__max_depth": [None, 8, 12, 16],
                "estimator__min_samples_leaf": [1, 2, 5, 10],
                "estimator__max_features": ["sqrt", 0.5, 0.8],
                "estimator__class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        "LightGBM": (
            lightgbm,
            {
                "estimator__n_estimators": [300, 500, 700],
                "estimator__learning_rate": [0.03, 0.05, 0.08],
                "estimator__num_leaves": [15, 31, 63],
                "estimator__max_depth": [-1, 8, 12],
                "estimator__min_child_samples": [20, 50, 100],
                "estimator__scale_pos_weight": [1, 10, 30, 60, 100, 200, 400, 600],
            },
        ),
        "CatBoost": (
            catboost,
            {
                "estimator__iterations": [400, 600, 800],
                "estimator__depth": [5, 6, 7, 8],
                "estimator__learning_rate": [0.03, 0.05, 0.08],
                "estimator__l2_leaf_reg": [3, 5, 9],
                "estimator__scale_pos_weight": [1, 10, 30, 60, 100, 200, 400, 600],
            },
        ),
    }


def run_experiment(
    data_path: str | Path = "data/creditcard.csv",
    output_dir: str | Path = "results",
    model_dir: str | Path = "models",
    n_iter: int = 8,
    cv_folds: int = 3,
) -> pd.DataFrame:
    """Train/tune models and evaluate once on the untouched test split."""
    if n_iter < 1:
        raise ValueError("n_iter должен быть не меньше 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds должен быть не меньше 2.")

    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_path)
    LOGGER.info("Размер датасета: %s строк, %s признаков", X.shape[0], X.shape[1])
    LOGGER.info("Распределение классов: %s", y.value_counts().sort_index().to_dict())
    df = X.assign(Class=y)

    # Требуемое в задании разбиение: 70% train / 30% test с сохранением пропорций классов.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=df.Class,
    )

    LOGGER.info(
        "Train: %s строк; Test: %s строк; доля Class=1 в test: %.5f",
        len(X_train),
        len(X_test),
        float(y_test.mean()),
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    model_spaces = build_model_spaces()
    thresholds = {"threshold": THRESHOLDS}
    results: list[dict[str, Any]] = []
    test_predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, ProbabilityThresholdClassifier] = {}

    for model_name, (base_estimator, parameter_space) in model_spaces.items():
        LOGGER.info("Старт настройки модели: %s", model_name)
        wrapped_estimator = ProbabilityThresholdClassifier(estimator=base_estimator)
        search_space = {**parameter_space, **thresholds}

        search = RandomizedSearchCV(
            estimator=wrapped_estimator,
            param_distributions=search_space,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            refit=True,
            random_state=RANDOM_STATE,
            # Внутренние модели используют свои потоки CPU; внешний поиск оставляем последовательным.
            n_jobs=1,
            pre_dispatch=1,
            verbose=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, best_model.positive_index_]
        test_predictions[model_name] = y_pred
        fitted_models[model_name] = best_model

        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1], zero_division=0
        )
        row = {
            "model": model_name,
            "cv_macro_f1": float(search.best_score_),
            "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "test_f1_class_0": float(class_f1[0]),
            "test_f1_class_1": float(class_f1[1]),
            "test_precision_fraud": float(class_precision[1]),
            "test_recall_fraud": float(class_recall[1]),
            "test_average_precision": float(average_precision_score(y_test, y_proba)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "threshold": float(best_model.threshold),
            "best_params": json.dumps(search.best_params_, ensure_ascii=False, default=str),
        }
        results.append(row)

        cv_results = pd.DataFrame(search.cv_results_)
        safe_name = model_name.lower().replace(" ", "_")
        cv_results.to_csv(output_dir / f"{safe_name}_cv_results.csv", index=False)

        LOGGER.info(
            "%s | CV macro F1=%.5f | Test macro F1=%.5f | threshold=%.3f",
            model_name,
            row["cv_macro_f1"],
            row["test_macro_f1"],
            row["threshold"],
        )

    results_df = pd.DataFrame(results).sort_values(
        by="test_macro_f1", ascending=False
    ).reset_index(drop=True)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    # Матрица ошибок для модели с максимальным macro F1 на тестовой выборке.
    # Тест используется только для итогового отчёта; поиск параметров проходил на train.
    best_test_name = str(results_df.loc[0, "model"])
    best_test_model = fitted_models[best_test_name]
    best_test_pred = test_predictions[best_test_name]
    matrix = confusion_matrix(y_test, best_test_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Легальная (0)", "Мошенническая (1)"],
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(
        f"Матрица ошибок: {best_test_name}\n"
        f"Test macro F1 = {results_df.loc[0, 'test_macro_f1']:.5f}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "best_confusion_matrix.png", dpi=160)
    plt.close(fig)

    report = classification_report(
        y_test, best_test_pred,
        labels=[0, 1],
        target_names=["Легальная (0)", "Мошенническая (1)"],
        zero_division=0,
        output_dict=True,
    )
    with (output_dir / "best_classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "model_name": best_test_name,
            "model": best_test_model,
            "features": X.columns.tolist(),
            "threshold": float(best_test_model.threshold),
        },
        model_dir / "best_model.joblib",
    )

    print("\nИТОГОВОЕ СРАВНЕНИЕ (модели отсортированы по test macro F1):")
    print(
        results_df[
            ["model", "cv_macro_f1", "test_macro_f1", "test_f1_class_1", "threshold"]
        ].to_string(index=False)
    )
    print(f"\nЛучшая на тесте модель: {best_test_name}")
    print("Матрица ошибок (строки — истинный класс, столбцы — предсказанный):")
    print(matrix)
    print(f"\nРезультаты сохранены в: {output_dir.resolve()}")
    print(f"Модель сохранена в: {(model_dir / 'best_model.joblib').resolve()}")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit card fraud classification lab")
    parser.add_argument("--data", default="data/creditcard.csv", help="Путь к creditcard.csv")
    parser.add_argument("--output", default="results", help="Папка для метрик и графиков")
    parser.add_argument("--models", default="models", help="Папка для сохранения модели")
    parser.add_argument("--n-iter", type=int, default=8, help="Число случайных комбинаций параметров на модель")
    parser.add_argument("--cv-folds", type=int, default=3, help="Количество фолдов StratifiedKFold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        data_path=args.data,
        output_dir=args.output,
        model_dir=args.models,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
