"""Спільні дані та незалежні очікувані результати для pytest."""

from __future__ import annotations

import importlib
from math import fsum
from typing import Any, Callable, Iterable

import numpy as np
import pytest


@pytest.fixture(params=(1, 2, 3), ids=lambda order: f"L{order}")
def norm_order(request: pytest.FixtureRequest) -> int:
    """Надає підтримувані порядки норм для параметризованих тестів."""
    return request.param


@pytest.fixture
def norm_orders() -> tuple[int, ...]:
    """Повертає повний набір порядків норм для інтеграційних перевірок."""
    return (1, 2, 3)


@pytest.fixture(
    params=(
        pytest.param((3.0, -4.0, 12.0), id="mixed-signs"),
        pytest.param((1.0, -2.0, 2.0), id="small-vector"),
        pytest.param((1.0, 1.0), id="equal-coordinates"),
    )
)
def vector_case(request: pytest.FixtureRequest) -> np.ndarray:
    """Повертає детерміновані вектори для перевірки норм."""
    return np.asarray(request.param, dtype=np.float64)


@pytest.fixture
def matrix_data() -> np.ndarray:
    """Повертає матрицю з від'ємними та сталими ознаками."""
    return np.asarray(
        [
            [-3.0, 4.0, 10.0],
            [0.0, -2.0, 10.0],
            [1.25, 2.0, 10.0],
            [8.0, 0.5, 10.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture(params=(0, -1, 4), ids=("zero", "negative", "unsupported"))
def invalid_norm_order(request: pytest.FixtureRequest) -> int:
    """Повертає непідтримувані порядки норми."""
    return request.param


@pytest.fixture(
    params=((0.0, 1.0), (-1.0, 1.0)),
    ids=("unit-interval", "symmetric-interval"),
)
def feature_range(request: pytest.FixtureRequest) -> tuple[float, float]:
    """Повертає діапазони для перевірки масштабування ознак."""
    return request.param


@pytest.fixture
def reference_lp_norm() -> Callable[[Iterable[float], int], float]:
    """Створює незалежний еталон Lp-норми на стандартній бібліотеці."""

    def calculate_norm(values: Iterable[float], order: int) -> float:
        powered_values = (abs(float(value)) ** order for value in values)
        return fsum(powered_values) ** (1.0 / order)

    return calculate_norm


@pytest.fixture
def reference_row_norms(
    reference_lp_norm: Callable[[Iterable[float], int], float],
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Створює еталонні норми рядків без залежності від бібліотек-перевірок."""

    def calculate_rows(data: np.ndarray, order: int) -> np.ndarray:
        return np.asarray(
            [reference_lp_norm(values, order) for values in data],
            dtype=np.float64,
        )

    return calculate_rows


@pytest.fixture
def reference_min_max_scale() -> Callable[
    [np.ndarray, tuple[float, float]], np.ndarray
]:
    """Обчислює очікуване масштабування без використання MinMaxScaler."""

    def scale_data(
        data: np.ndarray,
        target_range: tuple[float, float],
    ) -> np.ndarray:
        lower_bound, upper_bound = target_range
        feature_minima = data.min(axis=0)
        feature_ranges = data.max(axis=0) - feature_minima
        safe_ranges = np.where(feature_ranges == 0.0, 1.0, feature_ranges)
        normalized = (data - feature_minima) / safe_ranges
        return normalized * (upper_bound - lower_bound) + lower_bound

    return scale_data


@pytest.fixture
def iris_dataset() -> Any:
    """Повертає вбудований набір Iris зі scikit-learn."""
    sklearn_datasets: Any = importlib.import_module("sklearn.datasets")
    iris_loader: Any = getattr(sklearn_datasets, "load_iris")
    return iris_loader()


@pytest.fixture
def scaled_iris_features(
    iris_dataset: Any,
    reference_min_max_scale: Callable[
        [np.ndarray, tuple[float, float]], np.ndarray
    ],
) -> np.ndarray:
    """Масштабує ознаки Iris незалежною формулою з одиничним діапазоном."""
    petal_features = np.asarray(iris_dataset.data[:, 2:4], dtype=np.float64)
    return reference_min_max_scale(petal_features, (0.0, 1.0))


@pytest.fixture
def centered_iris_features(scaled_iris_features: np.ndarray) -> np.ndarray:
    """Центрує масштабовані ознаки Iris відносно середнього."""
    return scaled_iris_features - scaled_iris_features.mean(axis=0)
