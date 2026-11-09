"""Заготовки тестів, які студент має доповнити."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def load_student_implementation() -> Any:
    """Завантажує реалізацію студента або пропускає тест до її створення."""
    return pytest.importorskip(
        "student_implementation",
        reason="Створіть student_implementation.py на основі student_template.py.",
    )


def test_minkowski_distance(
    vector_pair,
    minkowski_order: int,
    reference_minkowski_distance,
) -> None:
    """TODO: перевірте відстань Мінковського за незалежною фікстурою."""
    load_student_implementation()
    pytest.fail("TODO: додайте assertions для minkowski_distance.")


def test_scale_features(
    features_data: np.ndarray,
    scaling_method: str,
    reference_scale_features,
) -> None:
    """TODO: перевірте всі способи масштабування та сталу ознаку."""
    load_student_implementation()
    pytest.fail("TODO: додайте assertions для scale_features.")


def test_run_kmeans(
    iris_petal_features: np.ndarray,
) -> None:
    """TODO: перевірте K-Means без передачі міток видів."""
    load_student_implementation()
    pytest.fail("TODO: додайте assertions для run_kmeans.")


def test_clustering_metrics(
    cluster_case: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_silhouette,
) -> None:
    """TODO: перевірте silhouette та ARI окремо від навчання кластерів."""
    load_student_implementation()
    pytest.fail("TODO: додайте assertions для метрик кластеризації.")


def test_split_data(
    iris_petal_features: np.ndarray,
    iris_dataset,
) -> None:
    """TODO: перевірте розмір, відтворюваність і стратифікацію поділу."""
    load_student_implementation()
    pytest.fail("TODO: додайте assertions для split_data.")


def test_knn_pipeline_and_search(
    iris_petal_features: np.ndarray,
    iris_dataset,
) -> None:
    """TODO: перевірте Pipeline та добір параметрів через CV macro-F1."""
    load_student_implementation()
    pytest.fail("TODO: додайте assertions для k-NN і GridSearchCV.")


def test_evaluate_classifier(
    reference_classification_metrics,
) -> None:
    """TODO: перевірте accuracy, macro-F1 і матрицю помилок."""
    load_student_implementation()
    pytest.fail("TODO: додайте assertions для evaluate_classifier.")


def test_plot_comparison(
    iris_petal_features: np.ndarray,
    iris_dataset,
    tmp_path,
) -> None:
    """TODO: перевірте збереження графіків для двох ознак."""
    load_student_implementation()
    pytest.fail("TODO: додайте assertions для plot_comparison.")
