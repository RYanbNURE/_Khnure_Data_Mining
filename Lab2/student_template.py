"""Шаблон для самостійного виконання лабораторної роботи № 2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MINKOWSKI_ORDERS = (1, 2, 3)
SCALING_METHODS = ("none", "minmax", "standard")
KMEANS_CLUSTER_COUNTS = (2, 3, 4, 5)
KNN_NEIGHBORS = (1, 3, 5, 7, 9)
RANDOM_STATE = 7

ScalingMethod = Literal["none", "minmax", "standard"]


def _validate_order(order: int) -> None:
    if (
        isinstance(order, bool)
        or not isinstance(order, (int, np.integer))
        or int(order) not in MINKOWSKI_ORDERS
    ):
        raise ValueError("Порядок Мінковського має дорівнювати 1, 2 або 3.")


def _as_features(data: np.ndarray, name: str = "Ознаки") -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} мають бути непорожньою двовимірною матрицею.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} мають містити лише скінченні значення.")
    return values


def _as_labels(labels: np.ndarray, name: str = "Мітки") -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} мають бути непорожнім одновимірним масивом.")
    return values


def _make_scaler(method: ScalingMethod) -> str | MinMaxScaler | StandardScaler:
    if method == "none":
        return "passthrough"
    if method == "minmax":
        return MinMaxScaler()
    if method == "standard":
        return StandardScaler()
    raise ValueError("Масштабування має бути 'none', 'minmax' або 'standard'.")


def minkowski_distance(
    first: np.ndarray,
    second: np.ndarray,
    order: int,
) -> float:
    """Обчислює відстань Мінковського між двома векторами."""
    _validate_order(order)
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("Обидва аргументи мають бути одновимірними векторами.")
    if left.size == 0 or left.shape != right.shape:
        raise ValueError("Вектори мають бути непорожніми та мати однакову довжину.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Вектори мають містити лише скінченні значення.")

    # TODO 1: реалізуйте формулу відстані Мінковського через NumPy.
    raise NotImplementedError("Виконайте TODO 1: відстань Мінковського.")


def scale_features(data: np.ndarray, method: ScalingMethod) -> np.ndarray:
    """Масштабує ознаки або повертає їхню копію без масштабування."""
    values = _as_features(data)
    scaler = _make_scaler(method)
    if scaler == "passthrough":
        return values.copy()

    # TODO 2: застосуйте scaler до всієї матриці та поверніть новий масив.
    raise NotImplementedError("Виконайте TODO 2: масштабування ознак.")


def run_kmeans(
    data: np.ndarray,
    n_clusters: int,
    scaling_method: ScalingMethod = "minmax",
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Повертає масштабовані ознаки та кластери без використання міток видів."""
    values = _as_features(data)
    if (
        isinstance(n_clusters, bool)
        or not isinstance(n_clusters, (int, np.integer))
        or not 2 <= int(n_clusters) < len(values)
    ):
        raise ValueError("Кількість кластерів має бути від 2 до числа спостережень - 1.")

    # TODO 3: масштабуйте дані та виконайте K-Means із заданим random_state.
    raise NotImplementedError("Виконайте TODO 3: кластеризація K-Means.")


def silhouette_for_clusters(
    scaled_features: np.ndarray,
    cluster_ids: np.ndarray,
) -> float:
    """Обчислює silhouette score для кластерів."""
    values = _as_features(scaled_features, "Масштабовані ознаки")
    clusters = _as_labels(cluster_ids, "Номери кластерів")
    if len(values) != len(clusters):
        raise ValueError("Кількість ознак має збігатися з кількістю номерів кластерів.")
    distinct_clusters = np.unique(clusters).size
    if distinct_clusters < 2 or distinct_clusters >= len(clusters):
        raise ValueError("Silhouette вимагає від 2 до n_samples - 1 кластерів.")

    # TODO 4: обчисліть silhouette у тому самому просторі, де навчався K-Means.
    raise NotImplementedError("Виконайте TODO 4: silhouette score.")


def adjusted_rand_for_species(
    cluster_ids: np.ndarray,
    species_labels: np.ndarray,
) -> float:
    """Порівнює кластери з мітками видів лише після кластеризації."""
    clusters = _as_labels(cluster_ids, "Номери кластерів")
    labels = _as_labels(species_labels, "Мітки видів")
    if len(clusters) != len(labels):
        raise ValueError("Кількість кластерів має збігатися з кількістю міток видів.")

    # TODO 4: поверніть adjusted Rand index для двох наборів міток.
    raise NotImplementedError("Виконайте TODO 4: зовнішня оцінка ARI.")


def split_data(
    data: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Створює відтворюваний стратифікований поділ даних."""
    values = _as_features(data)
    targets = _as_labels(labels)
    if len(values) != len(targets):
        raise ValueError("Кількість ознак має збігатися з кількістю міток.")
    if not isinstance(test_size, (int, float)) or not 0.0 < test_size < 1.0:
        raise ValueError("test_size має бути числом між 0 та 1.")

    # TODO 5: використайте train_test_split зі stratify, test_size та random_state.
    raise NotImplementedError("Виконайте TODO 5: поділ даних.")


def build_knn_pipeline(
    scaling_method: ScalingMethod = "minmax",
    n_neighbors: int = 5,
    order: int = 2,
) -> Pipeline:
    """Створює Pipeline для k-NN з масштабуванням усередині конвеєра."""
    if (
        isinstance(n_neighbors, bool)
        or not isinstance(n_neighbors, (int, np.integer))
        or int(n_neighbors) < 1
    ):
        raise ValueError("Кількість сусідів має бути додатним цілим числом.")
    _validate_order(order)

    # TODO 6: додайте scaler і KNeighborsClassifier до Pipeline.
    raise NotImplementedError("Виконайте TODO 6: Pipeline для k-NN.")


def search_knn_parameters(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> GridSearchCV:
    """Добирає масштабування, число сусідів і p за CV macro-F1."""
    values = _as_features(training_features, "Навчальні ознаки")
    targets = _as_labels(training_labels, "Навчальні мітки")
    if len(values) != len(targets):
        raise ValueError("Кількість навчальних ознак має збігатися з мітками.")
    if (
        isinstance(cv_splits, bool)
        or not isinstance(cv_splits, int)
        or cv_splits < 2
    ):
        raise ValueError("cv_splits має бути цілим числом не меншим за 2.")
    class_counts = np.unique(targets, return_counts=True)[1]
    if class_counts.min() < cv_splits:
        raise ValueError("Кожен клас має містити щонайменше cv_splits спостережень.")

    # TODO 6: створіть GridSearchCV із Pipeline, StratifiedKFold та macro-F1.
    # Переберіть SCALING_METHODS, KNN_NEIGHBORS і MINKOWSKI_ORDERS.
    raise NotImplementedError("Виконайте TODO 6: добір параметрів k-NN.")


def evaluate_classifier(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_labels: np.ndarray | tuple[int, ...] | None = None,
) -> dict[str, float | np.ndarray]:
    """Обчислює accuracy, macro-F1 і матрицю помилок."""
    actual = _as_labels(true_labels, "Справжні мітки")
    predicted = _as_labels(predicted_labels, "Передбачені мітки")
    if actual.shape != predicted.shape:
        raise ValueError("Справжні й передбачені мітки мають мати однакову довжину.")

    ordered_labels = None
    if class_labels is not None:
        ordered_labels = _as_labels(np.asarray(class_labels), "Список класів")
        if np.unique(ordered_labels).size != len(ordered_labels):
            raise ValueError("Список класів не має містити повторів.")

    # TODO 7: поверніть accuracy, macro-F1 та confusion matrix.
    raise NotImplementedError("Виконайте TODO 7: метрики класифікації.")


def plot_comparison(
    features: np.ndarray,
    species_labels: np.ndarray,
    target_names: np.ndarray,
    cluster_ids: np.ndarray,
    fitted_knn: Pipeline,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Малює види Iris, кластери K-Means і межі рішень k-NN."""
    values = _as_features(features)
    species = _as_labels(species_labels, "Мітки видів")
    clusters = _as_labels(cluster_ids, "Номери кластерів")
    names = np.asarray(target_names)
    if values.shape[1] != 2:
        raise ValueError("Для графіка потрібні рівно дві ознаки.")
    if len(values) != len(species) or len(values) != len(clusters):
        raise ValueError("Ознаки, мітки видів і кластери мають однакову довжину.")
    if names.ndim != 1 or names.size == 0:
        raise ValueError("target_names мають бути непорожнім одновимірним масивом.")
    if not np.isin(species, np.arange(len(names))).all():
        raise ValueError("Мітки видів мають відповідати індексам target_names.")

    # TODO 8: побудуйте 3 графіки: справжні види, кластери та межі k-NN.
    # Якщо задано save_path, збережіть зображення; якщо show=True, покажіть його.
    raise NotImplementedError("Виконайте TODO 8: графіки результатів.")


def _scaling_name(fitted_knn: Pipeline) -> str:
    scaler = fitted_knn.named_steps["scaler"]
    if scaler == "passthrough":
        return "none"
    if isinstance(scaler, MinMaxScaler):
        return "minmax"
    if isinstance(scaler, StandardScaler):
        return "standard"
    raise TypeError("У Pipeline знайдено неочікуваний масштабувальник.")


def main(show_plot: bool = False) -> None:
    """Виконує порівняння K-Means і k-NN на вбудованому наборі Iris."""
    iris = load_iris()
    features = np.asarray(iris.data[:, 2:4], dtype=np.float64)
    labels = iris.target

    print("Лабораторна робота № 2: K-Means та k-NN на Iris")
    print(f"Спостережень: {len(features)}; ознаки: довжина та ширина пелюстки")
    print("\nK-Means: мітки видів не використовуються під час навчання")
    print("Масштабування | K | Silhouette | ARI (лише зовнішнє порівняння)")

    plot_clusters: np.ndarray | None = None
    for scaling_method in SCALING_METHODS:
        for cluster_count in KMEANS_CLUSTER_COUNTS:
            scaled, cluster_ids = run_kmeans(
                features,
                cluster_count,
                scaling_method=scaling_method,
            )
            silhouette = silhouette_for_clusters(scaled, cluster_ids)
            ari = adjusted_rand_for_species(cluster_ids, labels)
            print(
                f"{scaling_method:13} | {cluster_count} | "
                f"{silhouette:10.4f} | {ari:.4f}"
            )
            if scaling_method == "minmax" and cluster_count == 3:
                plot_clusters = cluster_ids

    x_train, x_test, y_train, y_test = split_data(features, labels)
    search = search_knn_parameters(x_train, y_train)
    predictions = search.predict(x_test)
    class_labels = tuple(range(len(iris.target_names)))
    metrics = evaluate_classifier(y_test, predictions, class_labels=class_labels)

    print("\nk-NN: добір параметрів лише на навчальній вибірці")
    print(f"Масштабування: {_scaling_name(search.best_estimator_)}")
    print(f"Кількість сусідів: {search.best_params_['knn__n_neighbors']}")
    print(f"Порядок Мінковського p: {search.best_params_['knn__p']}")
    print(f"Найкращий CV macro-F1: {search.best_score_:.4f}")
    print(f"Тестова accuracy: {metrics['accuracy']:.4f}")
    print(f"Тестовий macro-F1: {metrics['macro_f1']:.4f}")
    print("Матриця помилок (рядки — справжні види, стовпці — передбачені):")
    print(metrics["confusion_matrix"])
    print("Метрики за видами:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=class_labels,
            target_names=iris.target_names,
            zero_division=0,
        )
    )
    print(
        "Пам'ятайте: номер кластера K-Means не є назвою виду; "
        "ARI наведено лише для зовнішнього порівняння."
    )

    if show_plot:
        if plot_clusters is None:
            raise RuntimeError("Не вдалося отримати кластери K-Means для графіка.")
        plot_comparison(
            features,
            labels,
            iris.target_names,
            plot_clusters,
            search.best_estimator_,
            show=True,
        )


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--plot",
        action="store_true",
        help="показати графіки кластерів і меж рішень",
    )
    main(show_plot=argument_parser.parse_args().plot)
