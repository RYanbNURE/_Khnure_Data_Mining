"""Лабораторна робота № 1: норми L1, L2 та L3 у бібліотеках Python.

Скрипт порівнює обчислення NumPy, SciPy, scikit-learn і PyTorch
на однаковому векторі та одному наборі даних.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import norm as scipy_norm
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

NORM_ORDERS = (1, 2, 3)


def validate_order(order: int) -> None:
    """Перевіряє, чи підтримується порядок норми."""
    if order not in NORM_ORDERS:
        raise ValueError("Підтримуються норми порядків 1, 2 та 3.")


def vector_norms_numpy(vector: np.ndarray) -> dict[int, float]:
    """Обчислює норми вектора засобами NumPy."""
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")
    return {order: float(np.linalg.norm(vector, ord=order)) for order in NORM_ORDERS}


def vector_norms_scipy(vector: np.ndarray) -> dict[int, float]:
    """Обчислює норми вектора засобами SciPy."""
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")
    return {order: float(scipy_norm(vector, ord=order)) for order in NORM_ORDERS}


def vector_norms_sklearn(vector: np.ndarray) -> dict[int, float]:
    """Обчислює відстані вектора до нуля засобами scikit-learn."""
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")
    return {
        order: float(sklearn_row_norms(vector.reshape(1, -1), order)[0])
        for order in NORM_ORDERS
    }


def vector_norms_torch(vector: torch.Tensor) -> dict[int, torch.Tensor]:
    """Обчислює норми одновимірного тензора засобами PyTorch."""
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")
    return {
        order: torch.linalg.vector_norm(vector, ord=order)
        for order in NORM_ORDERS
    }


def row_norms_numpy(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює норму кожного рядка матриці засобами NumPy."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")
    return np.linalg.norm(data, ord=order, axis=1)


def row_norms_scipy(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює норму кожного рядка матриці засобами SciPy."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")
    return scipy_norm(data, ord=order, axis=1)


def sklearn_row_norms(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює відстань кожного рядка до нуля через scikit-learn."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")

    origin = np.zeros((1, data.shape[1]), dtype=data.dtype)
    return pairwise_distances(
        data,
        origin,
        metric="minkowski",
        p=order,
    ).ravel()


def row_norms_torch(data: torch.Tensor, order: int) -> torch.Tensor:
    """Обчислює норму кожного рядка матриці засобами PyTorch."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")
    return torch.linalg.vector_norm(data, ord=order, dim=1)


def min_max_scale(
    data: np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Масштабує ознаки за допомогою MinMaxScaler з scikit-learn."""
    return MinMaxScaler(feature_range=feature_range).fit_transform(data)


def main() -> None:
    # Використовуємо фіксований генератор випадкових чисел для відтворюваності.
    rng = np.random.default_rng(7)

    sample = np.array([3.0, -4.0, 12.0], dtype=np.float64)
    sample_tensor = torch.from_numpy(sample.copy())
    torch_norms = {
        order: float(value.item())
        for order, value in vector_norms_torch(sample_tensor).items()
    }
    implementations = {
        "NumPy": vector_norms_numpy(sample),
        "SciPy": vector_norms_scipy(sample),
        "scikit-learn": vector_norms_sklearn(sample),
        "PyTorch": torch_norms,
    }

    print("Порівняння норм вектора в чотирьох бібліотеках")
    print(f"Вектор: {sample.tolist()}")
    for library, values in implementations.items():
        formatted = ", ".join(
            f"L{order}={values[order]:.4f}" for order in NORM_ORDERS
        )
        print(f"{library}: {formatted}")

    reference = implementations["NumPy"]
    for values in implementations.values():
        np.testing.assert_allclose(
            [values[order] for order in NORM_ORDERS],
            [reference[order] for order in NORM_ORDERS],
            rtol=1e-6,
            atol=1e-8,
        )
    print("Перевірку збігу результатів пройдено.")

    # Створюємо однакові для всіх бібліотек дані з одним доданим викидом.
    regular_points = rng.normal(size=(12, 2)) * np.array([1.0, 0.6])
    dataset = np.vstack((regular_points, np.array([[5.0, 4.0]])))

    # Масштабуємо ознаки інструментом MinMaxScaler з scikit-learn.
    scaled_data = min_max_scale(dataset)
    scaled_tensor = torch.from_numpy(scaled_data.copy())
    print("\nНорми рядків масштабованого набору даних")
    print("Кожен рядок порівнюється з нульовим вектором.")

    for order in NORM_ORDERS:
        scores_by_library = {
            "NumPy": row_norms_numpy(scaled_data, order),
            "SciPy": row_norms_scipy(scaled_data, order),
            "scikit-learn": sklearn_row_norms(scaled_data, order),
            "PyTorch": row_norms_torch(scaled_tensor, order).cpu().numpy(),
        }
        numpy_scores = scores_by_library["NumPy"]

        print(f"\nНорма L{order}:")
        for library, scores in scores_by_library.items():
            np.testing.assert_allclose(
                scores,
                numpy_scores,
                rtol=1e-6,
                atol=1e-8,
            )
            top_index = int(np.argmax(scores))
            max_difference = float(np.max(np.abs(scores - numpy_scores)))
            print(
                f"  {library}: рядок {top_index}, "
                f"оцінка={scores[top_index]:.4f}, "
                f"це доданий викид={top_index == len(dataset) - 1}, "
                f"максимальна різниця з NumPy={max_difference:.2e}"
            )

    # Демонструємо чутливість норм до збільшення координати.
    print("\nЧутливість до великої координати (NumPy)")
    for magnitude in (1.0, 5.0, 20.0):
        vector = np.array([1.0, magnitude])
        values = vector_norms_numpy(vector)
        formatted = ", ".join(
            f"L{order}={values[order]:.3f}" for order in NORM_ORDERS
        )
        print(f"[1, {magnitude:.0f}]: {formatted}")


if __name__ == "__main__":
    main()
