"""Навчальний приклад написання pytest-тестів для шаблону."""

from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np
import pytest

implementation: Any = (
    import_module("student_implementation")
    if find_spec("student_implementation") is not None
    else None
)


@pytest.fixture(autouse=True)
def require_student_implementation() -> None:
    """Пропускає приклади, поки студент не створив файл реалізації."""
    if implementation is None:
        pytest.skip("Створіть student_implementation.py на основі шаблону.")


def test_manual_numpy_norm_matches_reference(
    vector_case: np.ndarray,
    norm_order: int,
    reference_lp_norm,
) -> None:
    """Порівнює власну реалізацію з незалежним еталоном із фікстури."""
    actual = implementation.manual_lp_norm_numpy(vector_case, norm_order)
    expected = reference_lp_norm(vector_case, norm_order)

    assert actual == pytest.approx(expected)


def test_manual_numpy_norm_rejects_matrix_input() -> None:
    """Перевіряє зрозумілу помилку для аргументу не-вектора."""
    with pytest.raises(ValueError, match="одновимірний вектор"):
        implementation.manual_lp_norm_numpy(np.zeros((2, 2)), order=2)


def test_manual_numpy_norm_rejects_unsupported_order(
    vector_case: np.ndarray,
    invalid_norm_order: int,
) -> None:
    """Перевіряє відхилення непідтримуваного порядку норми."""
    with pytest.raises(ValueError, match="Порядок норми"):
        implementation.manual_lp_norm_numpy(vector_case, invalid_norm_order)
