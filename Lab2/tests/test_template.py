"""Навчальні тести першого TODO у шаблоні студента."""

from importlib import import_module
from importlib.util import find_spec
from typing import Any

import pytest

implementation: Any = (
    import_module("student_implementation")
    if find_spec("student_implementation") is not None
    else None
)


@pytest.fixture(autouse=True)
def require_student_implementation() -> None:
    """Пропускає тести, доки студент не створив власний модуль."""
    if implementation is None:
        pytest.skip("Створіть student_implementation.py на основі student_template.py.")


def test_minkowski_distance_matches_reference(
    vector_pair,
    minkowski_order: int,
    reference_minkowski_distance,
) -> None:
    """Перевіряє ручне обчислення відстані Мінковського."""
    first, second = vector_pair
    actual = implementation.minkowski_distance(first, second, minkowski_order)
    expected = reference_minkowski_distance(first, second, minkowski_order)

    assert actual == pytest.approx(expected)


def test_minkowski_distance_rejects_invalid_order(
    vector_pair,
    invalid_order: int,
) -> None:
    """Перевіряє відхилення непідтримуваного порядку."""
    first, second = vector_pair

    with pytest.raises(ValueError, match="Порядок Мінковського"):
        implementation.minkowski_distance(first, second, invalid_order)
