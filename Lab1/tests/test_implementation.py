"""Повний набір pytest-перевірок для студентської реалізації."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import student_template as implementation


@pytest.mark.parametrize(
    "function_name",
    ("manual_lp_norm_numpy", "manual_lp_norm_torch"),
)
def test_manual_norms_match_independent_reference(
    function_name: str,
    vector_case: np.ndarray,
    norm_order: int,
    reference_lp_norm,
) -> None:
    """Перевіряє власні формули NumPy та PyTorch."""
    function = getattr(implementation, function_name)
    values = vector_case
    if function_name.endswith("torch"):
        values = torch.as_tensor(vector_case.copy(), dtype=torch.float64)

    actual = function(values, norm_order)
    if isinstance(actual, torch.Tensor):
        actual = actual.item()
    expected = reference_lp_norm(vector_case, norm_order)

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "function_name",
    (
        "row_norms_numpy",
        "row_norms_scipy",
        "row_norms_sklearn",
        "row_norms_torch",
    ),
)
def test_row_norms_match_independent_reference(
    function_name: str,
    matrix_data: np.ndarray,
    norm_order: int,
    reference_row_norms,
) -> None:
    """Перевіряє норми рядків у кожній бібліотеці."""
    function = getattr(implementation, function_name)
    values = matrix_data
    if function_name.endswith("torch"):
        values = torch.as_tensor(matrix_data.copy(), dtype=torch.float64)

    actual = function(values, norm_order)
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    expected = reference_row_norms(matrix_data, norm_order)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_feature_scaling_matches_independent_formula(
    matrix_data: np.ndarray,
    feature_range: tuple[float, float],
    reference_min_max_scale,
) -> None:
    """Перевіряє MinMax-масштабування, включно зі сталою ознакою."""
    actual = implementation.scale_features(matrix_data, feature_range)
    expected = reference_min_max_scale(matrix_data, feature_range)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "function_name",
    (
        "manual_lp_norm_numpy",
        "manual_lp_norm_torch",
        "row_norms_numpy",
        "row_norms_scipy",
        "row_norms_sklearn",
        "row_norms_torch",
    ),
)
def test_norm_functions_reject_unsupported_orders(
    function_name: str,
    vector_case: np.ndarray,
    matrix_data: np.ndarray,
    invalid_norm_order: int,
) -> None:
    """Перевіряє відхилення непідтримуваного порядку в усіх функціях норм."""
    function = getattr(implementation, function_name)
    values = vector_case if function_name.startswith("manual_") else matrix_data
    if function_name.endswith("torch"):
        values = torch.as_tensor(values.copy(), dtype=torch.float64)

    with pytest.raises(ValueError, match="Порядок норми"):
        function(values, invalid_norm_order)


@pytest.mark.parametrize(
    "function_name",
    (
        "manual_lp_norm_numpy",
        "manual_lp_norm_torch",
    ),
)
def test_manual_norms_reject_matrices(
    function_name: str,
    matrix_data: np.ndarray,
) -> None:
    """Перевіряє, що власні функції приймають лише одновимірні вектори."""
    function = getattr(implementation, function_name)
    values = matrix_data
    if function_name.endswith("torch"):
        values = torch.as_tensor(matrix_data.copy(), dtype=torch.float64)

    with pytest.raises(ValueError, match="одновимірний вектор"):
        function(values, order=2)


@pytest.mark.parametrize(
    "function_name",
    (
        "row_norms_numpy",
        "row_norms_scipy",
        "row_norms_sklearn",
        "row_norms_torch",
    ),
)
def test_row_norm_functions_reject_vectors(
    function_name: str,
    vector_case: np.ndarray,
) -> None:
    """Перевіряє, що функції рядкових норм приймають лише матриці."""
    function = getattr(implementation, function_name)
    values = vector_case
    if function_name.endswith("torch"):
        values = torch.as_tensor(vector_case.copy(), dtype=torch.float64)

    with pytest.raises(ValueError, match="двовимірна матриця"):
        function(values, order=2)


def test_iris_comparison_matches_reference_for_each_order(
    centered_iris_features: np.ndarray,
    norm_orders: tuple[int, ...],
    reference_row_norms,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Перевіряє інтеграційне порівняння бібліотек на Iris."""
    actual_by_order = implementation.compare_libraries(
        centered_iris_features,
        "Перевірка Iris у тесті",
    )
    capsys.readouterr()

    assert set(actual_by_order) == set(norm_orders)
    for order in norm_orders:
        expected = reference_row_norms(centered_iris_features, order)
        np.testing.assert_allclose(
            actual_by_order[order],
            expected,
            rtol=1e-6,
            atol=1e-8,
        )


def test_main_reports_reference_top_five_iris_observations(
    iris_dataset,
    centered_iris_features: np.ndarray,
    norm_orders: tuple[int, ...],
    reference_row_norms,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Перевіряє виведення динамічно обчислених п'яти найбільших відхилень."""
    implementation.main()
    output = capsys.readouterr().out
    assert "П'ять найбільших відхилень" in output

    for order in norm_orders:
        expected_scores = reference_row_norms(centered_iris_features, order)
        expected_indices = np.argsort(expected_scores)[-5:][::-1]
        for index in expected_indices:
            species_name = iris_dataset.target_names[iris_dataset.target[index]]
            expected_label = f"індекс={index}, вид={species_name}"
            assert expected_label in output
