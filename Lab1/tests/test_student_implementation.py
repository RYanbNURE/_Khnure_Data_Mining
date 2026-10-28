"""Заготовки тестів, які студент має реалізувати для власного рішення."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pytest


def load_student_implementation() -> Any:
    """Завантажує реалізацію студента або пропускає тест до її створення."""
    return pytest.importorskip(
        "student_implementation",
        reason="Створіть student_implementation.py на основі student_template.py.",
    )


def test_manual_lp_norm_numpy(
    vector_case: np.ndarray,
    norm_order: int,
    reference_lp_norm: Callable[[Iterable[float], int], float],
) -> None:
    """TODO: протестуйте ручну Lp-норму NumPy за еталонною фікстурою."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки manual_lp_norm_numpy.")


def test_row_norms_numpy(
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """TODO: перевірте норми рядків NumPy для кожного підтримуваного порядку."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки row_norms_numpy.")


def test_row_norms_scipy(
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """TODO: перевірте норми рядків SciPy та параметр axis."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки row_norms_scipy.")


def test_vector_norm_scipy(
    vector_case: np.ndarray,
    norm_order: int,
    reference_lp_norm: Callable[[Iterable[float], int], float],
) -> None:
    """TODO: порівняйте норму одного вектора SciPy з еталонною фікстурою."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки vector_norm_scipy.")


def test_scale_features(
    matrix_data: np.ndarray,
    feature_range: tuple[float, float],
    reference_min_max_scale: Callable[
        [np.ndarray, tuple[float, float]], np.ndarray
    ],
) -> None:
    """TODO: перевірте MinMax-масштабування, зокрема сталу ознаку."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки scale_features.")


def test_row_norms_sklearn(
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """TODO: перевірте відстані Мінковського від рядків до нуля."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки row_norms_sklearn.")


def test_manual_lp_norm_torch(
    vector_case: np.ndarray,
    norm_order: int,
    reference_lp_norm: Callable[[Iterable[float], int], float],
) -> None:
    """TODO: перевірте ручну формулу норми PyTorch."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки manual_lp_norm_torch.")


def test_row_norms_torch(
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """TODO: перевірте норми рядків PyTorch із dim=1."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки row_norms_torch.")


def test_compare_libraries(
    centered_iris_features: np.ndarray,
    norm_orders: tuple[int, ...],
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """TODO: перевірте збіг бібліотек і результат для кожного порядку."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки compare_libraries.")


def test_top_five_observations(
    iris_dataset: Any,
    centered_iris_features: np.ndarray,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """TODO: перевірте індекси, види та оцінки п'яти найбільших норм."""
    load_student_implementation()
    pytest.fail("TODO: додайте власні перевірки top_five_observations.")


def test_norm_methods_reject_unsupported_order(
    invalid_norm_order: int,
    vector_case: np.ndarray,
    matrix_data: np.ndarray,
) -> None:
    """TODO: перевірте ValueError для непідтримуваного порядку норми."""
    load_student_implementation()
    pytest.fail("TODO: перевірте валідацію order у функціях норм.")


def test_vector_norm_methods_reject_matrix(
    matrix_data: np.ndarray,
) -> None:
    """TODO: перевірте відхилення матриці функціями векторних норм."""
    load_student_implementation()
    pytest.fail("TODO: перевірте валідацію розмірності векторних функцій.")


def test_row_norm_methods_reject_vector(
    vector_case: np.ndarray,
) -> None:
    """TODO: перевірте відхилення вектора функціями нормування рядків."""
    load_student_implementation()
    pytest.fail("TODO: перевірте валідацію розмірності функцій рядків.")
