"""Власні тести для реалізації лабораторної роботи № 1."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pytest
import torch

RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-8
LIBRARIES = ("NumPy", "SciPy", "scikit-learn", "PyTorch")


def load_student_implementation() -> Any:
    """Завантажує реалізацію студента або пропускає тест до її створення."""
    return pytest.importorskip(
        "student_implementation",
        reason="Створіть student_implementation.py на основі student_template.py.",
    )


def assert_close(actual: Any, expected: Any) -> None:
    """Порівнює результати зі спільними для всіх тестів допусками."""
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def to_tensor(values: np.ndarray) -> torch.Tensor:
    """Створює незалежний від масиву NumPy тензор подвійної точності."""
    return torch.as_tensor(values.copy(), dtype=torch.float64)


def test_manual_lp_norm_numpy(
    vector_case: np.ndarray,
    norm_order: int,
    reference_lp_norm: Callable[[Iterable[float], int], float],
) -> None:
    """Перевіряє ручну Lp-норму NumPy за еталонною фікстурою."""
    student = load_student_implementation()
    original = vector_case.copy()

    result = student.manual_lp_norm_numpy(vector_case, norm_order)

    assert isinstance(result, float)
    assert result == pytest.approx(reference_lp_norm(vector_case, norm_order))
    np.testing.assert_array_equal(vector_case, original)


def test_row_norms_numpy(
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """Перевіряє норми рядків NumPy для кожного підтримуваного порядку."""
    student = load_student_implementation()

    result = student.row_norms_numpy(matrix_data, norm_order)

    assert result.shape == (matrix_data.shape[0],)
    assert_close(result, reference_row_norms(matrix_data, norm_order))


def test_row_norms_scipy(
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """Перевіряє, що SciPy обчислює норми саме рядків, а не стовпців."""
    student = load_student_implementation()

    result = student.row_norms_scipy(matrix_data, norm_order)

    # Матриця 4x3: вісь 0 дала б три значення замість чотирьох.
    assert result.shape == (matrix_data.shape[0],)
    assert_close(result, reference_row_norms(matrix_data, norm_order))


def test_vector_norm_scipy(
    vector_case: np.ndarray,
    norm_order: int,
    reference_lp_norm: Callable[[Iterable[float], int], float],
) -> None:
    """Порівнює норму одного вектора SciPy з еталонною фікстурою."""
    student = load_student_implementation()

    result = student.vector_norm_scipy(vector_case, norm_order)

    assert isinstance(result, float)
    assert result == pytest.approx(reference_lp_norm(vector_case, norm_order))


def test_scale_features(
    matrix_data: np.ndarray,
    feature_range: tuple[float, float],
    reference_min_max_scale: Callable[
        [np.ndarray, tuple[float, float]], np.ndarray
    ],
) -> None:
    """Перевіряє MinMax-масштабування, зокрема сталу ознаку."""
    student = load_student_implementation()
    lower_bound, upper_bound = feature_range
    original = matrix_data.copy()

    result = student.scale_features(matrix_data, feature_range)

    assert_close(result, reference_min_max_scale(matrix_data, feature_range))
    # Змінні ознаки займають увесь діапазон, а стала переходить у нижню межу.
    assert_close(result[:, :2].min(axis=0), [lower_bound, lower_bound])
    assert_close(result[:, :2].max(axis=0), [upper_bound, upper_bound])
    assert_close(result[:, 2], np.full(matrix_data.shape[0], lower_bound))
    np.testing.assert_array_equal(matrix_data, original)


def test_row_norms_sklearn(
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """Перевіряє відстані Мінковського від рядків до нуля."""
    student = load_student_implementation()

    result = student.row_norms_sklearn(matrix_data, norm_order)

    assert result.shape == (matrix_data.shape[0],)
    assert_close(result, reference_row_norms(matrix_data, norm_order))


def test_manual_lp_norm_torch(
    vector_case: np.ndarray,
    norm_order: int,
    reference_lp_norm: Callable[[Iterable[float], int], float],
) -> None:
    """Перевіряє ручну формулу норми PyTorch."""
    student = load_student_implementation()

    result = student.manual_lp_norm_torch(to_tensor(vector_case), norm_order)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert result.item() == pytest.approx(
        reference_lp_norm(vector_case, norm_order)
    )


def test_row_norms_torch(
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """Перевіряє норми рядків PyTorch із dim=1."""
    student = load_student_implementation()

    result = student.row_norms_torch(to_tensor(matrix_data), norm_order)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (matrix_data.shape[0],)
    assert_close(result.numpy(), reference_row_norms(matrix_data, norm_order))


def test_compare_libraries(
    centered_iris_features: np.ndarray,
    norm_orders: tuple[int, ...],
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Перевіряє збіг бібліотек і результат для кожного порядку."""
    student = load_student_implementation()
    title = "Центровані ознаки Iris"

    results = student.compare_libraries(centered_iris_features, title)

    assert tuple(results) == norm_orders
    output = capsys.readouterr().out
    assert title in output
    for order in norm_orders:
        expected = reference_row_norms(centered_iris_features, order)
        assert_close(results[order], expected)
        assert f"Норма L{order}:" in output
        assert f"рядок={np.argmax(expected)}" in output
    for library in LIBRARIES:
        assert output.count(f"{library}: рядок=") == len(norm_orders)


def test_compare_libraries_detects_mismatch(
    matrix_data: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Перевіряє, що розбіжність між бібліотеками не залишиться непоміченою."""
    student = load_student_implementation()
    row_norms_scipy = student.row_norms_scipy

    def shifted_norms(data: np.ndarray, order: int) -> np.ndarray:
        return row_norms_scipy(data, order) + 1e-3

    monkeypatch.setattr(student, "row_norms_scipy", shifted_norms)

    with pytest.raises(AssertionError):
        student.compare_libraries(matrix_data, "Розбіжність бібліотек")


def test_top_five_observations(
    iris_dataset: Any,
    centered_iris_features: np.ndarray,
    norm_order: int,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """Перевіряє індекси, види та оцінки п'яти найбільших норм."""
    student = load_student_implementation()
    scores = reference_row_norms(centered_iris_features, norm_order)

    observations = student.top_five_observations(
        scores,
        iris_dataset.target,
        iris_dataset.target_names,
    )

    assert len(observations) == 5
    indices = [index for index, _, _ in observations]
    assert len(set(indices)) == 5
    for index, species, score in observations:
        assert species == iris_dataset.target_names[iris_dataset.target[index]]
        assert score == pytest.approx(scores[index])
    # Послідовність оцінок має збігатися з п'ятьма найбільшими за спаданням.
    assert_close(
        [score for _, _, score in observations],
        sorted(scores, reverse=True)[:5],
    )


def test_top_five_observations_reject_invalid_input(
    iris_dataset: Any,
    centered_iris_features: np.ndarray,
    reference_row_norms: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """Перевіряє відхилення неузгоджених оцінок і міток."""
    student = load_student_implementation()
    scores = reference_row_norms(centered_iris_features, 2)
    labels = iris_dataset.target
    target_names = iris_dataset.target_names

    with pytest.raises(ValueError, match="одновимірними"):
        student.top_five_observations(
            scores.reshape(-1, 1),
            labels,
            target_names,
        )
    with pytest.raises(ValueError, match="Кількість оцінок"):
        student.top_five_observations(scores[:-1], labels, target_names)


def test_norm_methods_reject_unsupported_order(
    invalid_norm_order: int,
    vector_case: np.ndarray,
    matrix_data: np.ndarray,
) -> None:
    """Перевіряє ValueError для непідтримуваного порядку норми."""
    student = load_student_implementation()
    calls = (
        (student.manual_lp_norm_numpy, vector_case),
        (student.vector_norm_scipy, vector_case),
        (student.manual_lp_norm_torch, to_tensor(vector_case)),
        (student.row_norms_numpy, matrix_data),
        (student.row_norms_scipy, matrix_data),
        (student.row_norms_sklearn, matrix_data),
        (student.row_norms_torch, to_tensor(matrix_data)),
    )

    for method, values in calls:
        with pytest.raises(ValueError, match="Порядок норми"):
            method(values, invalid_norm_order)


def test_vector_norm_methods_reject_matrix(
    matrix_data: np.ndarray,
) -> None:
    """Перевіряє відхилення матриці функціями векторних норм."""
    student = load_student_implementation()
    calls = (
        (student.manual_lp_norm_numpy, matrix_data),
        (student.vector_norm_scipy, matrix_data),
        (student.manual_lp_norm_torch, to_tensor(matrix_data)),
    )

    for method, values in calls:
        with pytest.raises(ValueError, match="одновимірний вектор"):
            method(values, 2)


def test_row_norm_methods_reject_vector(
    vector_case: np.ndarray,
) -> None:
    """Перевіряє відхилення вектора функціями нормування рядків."""
    student = load_student_implementation()
    calls = (
        (student.row_norms_numpy, vector_case),
        (student.row_norms_scipy, vector_case),
        (student.row_norms_sklearn, vector_case),
        (student.row_norms_torch, to_tensor(vector_case)),
    )

    for method, values in calls:
        with pytest.raises(ValueError, match="двовимірна матриця"):
            method(values, 2)
