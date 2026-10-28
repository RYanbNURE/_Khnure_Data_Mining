"""Шаблон для самостійного виконання лабораторної роботи № 1."""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import norm as scipy_norm
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

NORM_ORDERS = (1, 2, 3)


def validate_order(order: int) -> None:
    """Перевіряє порядок норми."""
    if order not in NORM_ORDERS:
        raise ValueError("Порядок норми має дорівнювати 1, 2 або 3.")


def manual_lp_norm_numpy(vector: np.ndarray, order: int) -> float:
    """Обчислює норму вектора за формулою за допомогою NumPy."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")

    # TODO 2: реалізуйте формулу Lp-норми за допомогою np.abs та np.sum.
    raise NotImplementedError("Виконайте TODO 2: реалізація норми NumPy.")


def row_norms_numpy(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює норму кожного рядка за допомогою NumPy."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")

    # TODO 2: використайте np.linalg.norm із параметром axis=1.
    raise NotImplementedError("Виконайте TODO 2: норми рядків NumPy.")


def row_norms_scipy(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює норму кожного рядка за допомогою SciPy."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")

    # TODO 3: використайте scipy.linalg.norm із параметром axis=1.
    raise NotImplementedError("Виконайте TODO 3: обчислення норм SciPy.")


def vector_norm_scipy(vector: np.ndarray, order: int) -> float:
    """Обчислює норму вектора за допомогою SciPy."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")

    # TODO 3: обчисліть норму через scipy.linalg.norm із заданим порядком.
    raise NotImplementedError("Виконайте TODO 3: норма вектора SciPy.")


def scale_features(
    data: np.ndarray,
    feature_range: tuple[float, float],
) -> np.ndarray:
    """Масштабує ознаки за допомогою scikit-learn."""
    # TODO 4: створіть MinMaxScaler із feature_range та викличте fit_transform.
    raise NotImplementedError("Виконайте TODO 4: масштабування ознак.")


def row_norms_sklearn(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює відстань рядків до нуля через scikit-learn."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")

    # TODO 4: створіть нульовий вектор потрібної ширини.
    # TODO 4: викличте pairwise_distances з metric="minkowski" та p=order.
    raise NotImplementedError("Виконайте TODO 4: відстані scikit-learn.")


def manual_lp_norm_torch(vector: torch.Tensor, order: int) -> torch.Tensor:
    """Обчислює норму тензора за формулою за допомогою PyTorch."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")

    # TODO 5: реалізуйте формулу за допомогою torch.abs, степеня та torch.sum.
    raise NotImplementedError("Виконайте TODO 5: власна реалізація PyTorch.")


def row_norms_torch(data: torch.Tensor, order: int) -> torch.Tensor:
    """Обчислює норму кожного рядка за допомогою PyTorch."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")

    # TODO 5: використайте torch.linalg.vector_norm із ord=order та dim=1.
    raise NotImplementedError("Виконайте TODO 5: норми рядків PyTorch.")


def compare_libraries(data: np.ndarray, title: str) -> dict[int, np.ndarray]:
    """Обчислює й порівнює норми рядків у чотирьох бібліотеках."""
    print(f"\n{title}")
    results_by_order: dict[int, np.ndarray] = {}

    for order in NORM_ORDERS:
        tensor = torch.as_tensor(data.copy(), dtype=torch.float64)
        scores = {
            "NumPy": row_norms_numpy(data, order),
            "SciPy": row_norms_scipy(data, order),
            "scikit-learn": row_norms_sklearn(data, order),
            "PyTorch": row_norms_torch(tensor, order).cpu().numpy(),
        }

        # TODO 6: перевірте всі результати через np.testing.assert_allclose.
        # TODO 6: знайдіть і виведіть індекс найбільшої норми для кожної бібліотеки.
        results_by_order[order] = scores["NumPy"]

    return results_by_order


def top_five_observations(
    scores: np.ndarray,
    labels: np.ndarray,
    target_names: np.ndarray,
) -> list[tuple[int, str, float]]:
    """Повертає індекс, назву виду й оцінку для п'яти найбільших норм."""
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("Оцінки й мітки мають бути одновимірними масивами.")
    if len(scores) != len(labels):
        raise ValueError("Кількість оцінок має збігатися з кількістю міток.")

    # TODO 7: знайдіть п'ять найбільших оцінок і зіставте їх із назвами видів.
    raise NotImplementedError("Виконайте TODO 7: вибір топ-5 спостережень.")


def main() -> None:
    """Виконує завдання лабораторної роботи в заданій послідовності."""
    # TODO 1: вручну обчисліть норми цього вектора та запишіть їх у звіт.
    sample = np.array([3.0, -4.0, 12.0])
    print("Контрольний вектор:", sample)

    # TODO 2: обчисліть норми NumPy та порівняйте власну функцію з np.linalg.norm.
    for order in NORM_ORDERS:
        print(f"L{order} NumPy:", np.linalg.norm(sample, ord=order))
        print(f"L{order} власна реалізація:", manual_lp_norm_numpy(sample, order))

    # TODO 3: порівняйте норму вектора SciPy з результатом NumPy.
    for order in NORM_ORDERS:
        print(f"L{order} SciPy:", vector_norm_scipy(sample, order))

    # TODO 5: створіть тензор із sample і порівняйте обидва обчислення PyTorch.
    sample_tensor = torch.as_tensor(sample.copy(), dtype=torch.float64)
    for order in NORM_ORDERS:
        print(
            f"L{order} PyTorch:",
            torch.linalg.vector_norm(sample_tensor, ord=order).item(),
        )
        print(
            f"L{order} власна реалізація PyTorch:",
            manual_lp_norm_torch(sample_tensor, order).item(),
        )

    iris = load_iris()
    features = iris.data[:, 2:4]

    # TODO 4: порівняйте результати для обох діапазонів масштабування.
    scaled = scale_features(features, feature_range=(0.0, 1.0))
    compare_libraries(scaled, "Масштабовані ознаки Iris, діапазон [0, 1]")
    scaled_signed = scale_features(features, feature_range=(-1.0, 1.0))
    compare_libraries(scaled_signed, "Масштабовані ознаки Iris, діапазон [-1, 1]")

    # TODO 6: відцентруйте scaled, віднявши середнє кожної ознаки.
    centered = scaled - scaled.mean(axis=0)
    centered_scores = compare_libraries(centered, "Центровані ознаки Iris")

    # TODO 7: виведіть результати методу top_five_observations для кожної норми.
    for order, scores in centered_scores.items():
        print(f"\nП'ять найбільших відхилень для L{order}:")
        observations = top_five_observations(
            scores,
            iris.target,
            iris.target_names,
        )
        for index, species, score in observations:
            print(
                f"індекс={index}, вид={species}, норма={score:.4f}"
            )


if __name__ == "__main__":
    main()
