"""Спільні навчальні дані та незалежні еталонні обчислення."""

from __future__ import annotations

from math import fsum, sqrt
from typing import Callable, Iterable

import numpy as np
import pytest
from sklearn.datasets import load_iris


@pytest.fixture(params=(1, 2, 3), ids=lambda order: f"p{order}")
def minkowski_order(request: pytest.FixtureRequest) -> int:
    """Повертає підтримувані порядки відстані Мінковського."""
    return request.param


@pytest.fixture(params=(0, -1, 4), ids=("zero", "negative", "unsupported"))
def invalid_order(request: pytest.FixtureRequest) -> int:
    """Повертає непідтримуваний порядок відстані."""
    return request.param


@pytest.fixture
def vector_pair() -> tuple[np.ndarray, np.ndarray]:
    """Повертає дві детерміновані точки для перевірки відстані."""
    return (
        np.asarray([3.0, -4.0, 12.0], dtype=np.float64),
        np.asarray([1.0, -2.0, 2.0], dtype=np.float64),
    )


@pytest.fixture
def features_data() -> np.ndarray:
    """Повертає матрицю з від'ємними та сталими ознаками."""
    return np.asarray(
        [
            [-3.0, -2.0, 10.0],
            [-3.2, -2.8, 10.0],
            [-2.8, -3.2, 10.0],
            [0.0, 3.0, 10.0],
            [0.2, 2.7, 10.0],
            [-0.2, 3.2, 10.0],
            [3.0, -3.0, 10.0],
            [3.2, -2.7, 10.0],
            [2.8, -3.3, 10.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def cluster_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Повертає невеликі відокремлені групи для тестів кластеризації."""
    data = np.asarray(
        [
            [-2.2, -2.0],
            [-1.8, -2.1],
            [0.0, 2.1],
            [0.1, 1.9],
            [2.2, -1.9],
            [1.9, -2.2],
        ],
        dtype=np.float64,
    )
    cluster_ids = np.asarray([0, 0, 1, 1, 2, 2])
    species_labels = np.asarray([0, 0, 1, 1, 2, 2])
    return data, cluster_ids, species_labels


@pytest.fixture(params=("none", "minmax", "standard"))
def scaling_method(request: pytest.FixtureRequest) -> str:
    """Повертає підтримувані способи масштабування."""
    return request.param


@pytest.fixture
def iris_dataset():
    """Повертає вбудований набір Iris зі scikit-learn."""
    return load_iris()


@pytest.fixture
def iris_petal_features(iris_dataset) -> np.ndarray:
    """Повертає довжину та ширину пелюсток для наступних завдань."""
    return np.asarray(iris_dataset.data[:, 2:4], dtype=np.float64)


@pytest.fixture
def reference_minkowski_distance() -> Callable[
    [Iterable[float], Iterable[float], int], float
]:
    """Обчислює відстань Мінковського за допомогою math.fsum."""

    def calculate(
        first: Iterable[float],
        second: Iterable[float],
        order: int,
    ) -> float:
        powered_differences = (
            abs(float(left) - float(right)) ** order
            for left, right in zip(first, second)
        )
        return fsum(powered_differences) ** (1.0 / order)

    return calculate


@pytest.fixture
def reference_scale_features() -> Callable[[np.ndarray, str], np.ndarray]:
    """Масштабує ознаки незалежними формулами без sklearn."""

    def scale(data: np.ndarray, method: str) -> np.ndarray:
        values = np.asarray(data, dtype=np.float64)
        if method == "none":
            return values.copy()
        if method == "minmax":
            minima = values.min(axis=0)
            ranges = values.max(axis=0) - minima
            safe_ranges = np.where(ranges == 0.0, 1.0, ranges)
            return (values - minima) / safe_ranges
        if method == "standard":
            means = values.mean(axis=0)
            deviations = values.std(axis=0)
            safe_deviations = np.where(deviations == 0.0, 1.0, deviations)
            return (values - means) / safe_deviations
        raise ValueError(f"Unknown test scaling method: {method}")

    return scale


@pytest.fixture
def reference_silhouette() -> Callable[[np.ndarray, np.ndarray], float]:
    """Обчислює silhouette score через відстані Евкліда без sklearn."""

    def calculate(data: np.ndarray, cluster_ids: np.ndarray) -> float:
        values = np.asarray(data, dtype=np.float64)
        clusters = np.asarray(cluster_ids)
        scores: list[float] = []

        for index, point in enumerate(values):
            own_mask = clusters == clusters[index]
            own_mask[index] = False
            if not np.any(own_mask):
                scores.append(0.0)
                continue

            own_distances = [
                sqrt(fsum((float(left) - float(right)) ** 2 for left, right in zip(point, other)))
                for other in values[own_mask]
            ]
            average_own_distance = fsum(own_distances) / len(own_distances)

            other_cluster_distances: list[float] = []
            for other_cluster in np.unique(clusters):
                if other_cluster == clusters[index]:
                    continue
                other_points = values[clusters == other_cluster]
                distances = [
                    sqrt(
                        fsum(
                            (float(left) - float(right)) ** 2
                            for left, right in zip(point, other)
                        )
                    )
                    for other in other_points
                ]
                other_cluster_distances.append(fsum(distances) / len(distances))

            average_nearest_cluster_distance = min(other_cluster_distances)
            denominator = max(average_own_distance, average_nearest_cluster_distance)
            scores.append(
                (average_nearest_cluster_distance - average_own_distance) / denominator
                if denominator > 0.0
                else 0.0
            )

        return fsum(scores) / len(scores)

    return calculate


@pytest.fixture
def reference_classification_metrics() -> Callable[
    [np.ndarray, np.ndarray, Iterable[int]], dict[str, float | np.ndarray]
]:
    """Обчислює accuracy, macro-F1 та confusion matrix незалежними циклами."""

    def calculate(
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        class_labels: Iterable[int],
    ) -> dict[str, float | np.ndarray]:
        actual = np.asarray(true_labels)
        predicted = np.asarray(predicted_labels)
        classes = tuple(class_labels)
        matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
        class_f1: list[float] = []

        for true_value, predicted_value in zip(actual, predicted):
            true_index = classes.index(int(true_value))
            predicted_index = classes.index(int(predicted_value))
            matrix[true_index, predicted_index] += 1

        for class_value in classes:
            true_positive = int(np.sum((actual == class_value) & (predicted == class_value)))
            false_positive = int(np.sum((actual != class_value) & (predicted == class_value)))
            false_negative = int(np.sum((actual == class_value) & (predicted != class_value)))
            denominator = 2 * true_positive + false_positive + false_negative
            class_f1.append(2 * true_positive / denominator if denominator else 0.0)

        return {
            "accuracy": float(np.mean(actual == predicted)),
            "macro_f1": fsum(class_f1) / len(class_f1),
            "confusion_matrix": matrix,
        }

    return calculate
