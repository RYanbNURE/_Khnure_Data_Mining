"""Зразок повного виконання лабораторної роботи № 1."""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import norm as scipy_norm
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

NORM_ORDERS = (1, 2, 3)


def validate_order(order: int) -> None:
    """Перевіряє порядок норми, який використовується в лабораторній."""
    if order not in NORM_ORDERS:
        raise ValueError("Порядок норми має дорівнювати 1, 2 або 3.")


def manual_lp_norm_numpy(vector: np.ndarray, order: int) -> float:
    """Обчислює норму вектора за формулою, використовуючи NumPy."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")
    return float(np.sum(np.abs(vector) ** order) ** (1.0 / order))


def manual_lp_norm_torch(vector: torch.Tensor, order: int) -> torch.Tensor:
    """Обчислює норму тензора за формулою, використовуючи PyTorch."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")
    return torch.sum(torch.abs(vector) ** order) ** (1.0 / order)


def vector_norms_numpy(vector: np.ndarray) -> dict[int, float]:
    """Обчислює норми вектора засобами NumPy."""
    return {
        order: float(np.linalg.norm(vector, ord=order))
        for order in NORM_ORDERS
    }


def vector_norms_scipy(vector: np.ndarray) -> dict[int, float]:
    """Обчислює норми вектора засобами SciPy."""
    return {
        order: float(scipy_norm(vector, ord=order))
        for order in NORM_ORDERS
    }


def vector_norm_scipy(vector: np.ndarray, order: int) -> float:
    """Обчислює норму одного вектора засобами SciPy."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")
    return float(scipy_norm(vector, ord=order))


def vector_norms_sklearn(vector: np.ndarray) -> dict[int, float]:
    """Обчислює відстань вектора до нуля засобами scikit-learn."""
    return {
        order: float(row_norms_sklearn(vector.reshape(1, -1), order)[0])
        for order in NORM_ORDERS
    }


def vector_norms_torch(vector: torch.Tensor) -> dict[int, float]:
    """Обчислює норми вектора засобами PyTorch."""
    return {
        order: float(torch.linalg.vector_norm(vector, ord=order).item())
        for order in NORM_ORDERS
    }


def row_norms_numpy(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює норму кожного рядка за допомогою NumPy."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")
    return np.linalg.norm(data, ord=order, axis=1)


def row_norms_scipy(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює норму кожного рядка за допомогою SciPy."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")
    return scipy_norm(data, ord=order, axis=1)


def row_norms_sklearn(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює відстань кожного рядка до нуля за допомогою scikit-learn."""
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


def row_norms_torch(data: np.ndarray | torch.Tensor, order: int) -> np.ndarray:
    """Обчислює норму кожного рядка за допомогою PyTorch."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")
    tensor = torch.as_tensor(data, dtype=torch.float64)
    return torch.linalg.vector_norm(tensor, ord=order, dim=1).cpu().numpy()


def scale_features(
    data: np.ndarray,
    feature_range: tuple[float, float],
) -> np.ndarray:
    """Масштабує ознаки за допомогою MinMaxScaler."""
    return MinMaxScaler(feature_range=feature_range).fit_transform(data)


def compare_row_norms(data: np.ndarray, title: str) -> dict[int, np.ndarray]:
    """Порівнює норми рядків у чотирьох бібліотеках."""
    print(f"\n{title}")
    numpy_scores_by_order: dict[int, np.ndarray] = {}

    for order in NORM_ORDERS:
        scores_by_library = {
            "NumPy": row_norms_numpy(data, order),
            "SciPy": row_norms_scipy(data, order),
            "scikit-learn": row_norms_sklearn(data, order),
            "PyTorch": row_norms_torch(data, order),
        }
        reference = scores_by_library["NumPy"]
        numpy_scores_by_order[order] = reference

        print(f"Норма L{order}:")
        for library, scores in scores_by_library.items():
            np.testing.assert_allclose(
                scores,
                reference,
                rtol=1e-6,
                atol=1e-8,
            )
            maximum_difference = float(np.max(np.abs(scores - reference)))
            print(
                f"  {library}: найбільша різниця з NumPy "
                f"{maximum_difference:.2e}"
            )

        largest_index = int(np.argmax(reference))
        print(
            f"  Найбільша норма: рядок {largest_index}, "
            f"значення {reference[largest_index]:.4f}"
        )

    return numpy_scores_by_order


def top_five_observations(
    scores: np.ndarray,
    labels: np.ndarray,
    target_names: np.ndarray,
) -> list[tuple[int, str, float]]:
    """Повертає п'ять спостережень із найбільшими нормами."""
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("Оцінки й мітки мають бути одновимірними масивами.")
    if len(scores) != len(labels):
        raise ValueError("Кількість оцінок має збігатися з кількістю міток.")

    indices = np.argsort(scores)[-5:][::-1]
    return [
        (int(index), str(target_names[labels[index]]), float(scores[index]))
        for index in indices
    ]


def demonstrate_vector_norms() -> None:
    """Порівнює ручне обчислення норм і чотири бібліотеки."""
    samples = (
        np.array([3.0, -4.0, 12.0]),
        np.array([1.0, -2.0, 2.0]),
        np.array([1.0, 1.0]),
    )

    print("Завдання 1–3 і 5. Норми контрольних векторів")
    for sample in samples:
        print(f"\nВектор: {sample.tolist()}")
        sample_tensor = torch.as_tensor(sample.copy(), dtype=torch.float64)
        implementations = {
            "Власна формула NumPy": {
                order: manual_lp_norm_numpy(sample, order)
                for order in NORM_ORDERS
            },
            "NumPy": vector_norms_numpy(sample),
            "SciPy": vector_norms_scipy(sample),
            "scikit-learn": vector_norms_sklearn(sample),
            "Власна формула PyTorch": {
                order: float(manual_lp_norm_torch(sample_tensor, order).item())
                for order in NORM_ORDERS
            },
            "PyTorch": vector_norms_torch(sample_tensor),
        }

        reference = implementations["NumPy"]
        for library, values in implementations.items():
            np.testing.assert_allclose(
                [values[order] for order in NORM_ORDERS],
                [reference[order] for order in NORM_ORDERS],
                rtol=1e-6,
                atol=1e-8,
            )
            formatted = ", ".join(
                f"L{order}={values[order]:.4f}" for order in NORM_ORDERS
            )
            print(f"{library}: {formatted}")


def demonstrate_scaling_and_centering() -> None:
    """Показує вплив масштабування та центрування на оцінки."""
    generator = np.random.default_rng(7)
    ordinary_points = generator.normal(size=(12, 2)) * np.array([1.0, 0.6])
    artificial_outlier = np.array([[5.0, 4.0]])
    data = np.vstack((ordinary_points, artificial_outlier))

    print("\nЗавдання 4 і 6. Масштабування та центрування")
    for feature_range in ((0.0, 1.0), (-1.0, 1.0)):
        scaled = scale_features(data, feature_range)
        compare_row_norms(
            scaled,
            f"Діапазон масштабування {feature_range}; відстань до нуля",
        )

    centered = scaled - scaled.mean(axis=0)
    compare_row_norms(centered, "Центровані дані; відстань до середнього")


def demonstrate_iris() -> None:
    """Порівнює норми для ознак пелюсток у наборі Iris."""
    iris = load_iris()
    # Використовуємо довжину та ширину пелюстки кожної квітки.
    features = iris.data[:, 2:4]
    scaled = scale_features(features, feature_range=(0.0, 1.0))
    centered = scaled - scaled.mean(axis=0)

    scores_by_order = compare_row_norms(
        centered,
        "Завдання 7. Набір Iris, центровані ознаки пелюсток",
    )

    print("\nП'ять найбільших відхилень від середнього")
    for order, scores in scores_by_order.items():
        print(f"L{order}:")
        for index, species, score in top_five_observations(
            scores,
            iris.target,
            iris.target_names,
        ):
            print(
                f"  індекс={index}, вид={species}, "
                f"норма={score:.4f}"
            )


def main() -> None:
    """Послідовно виконує приклади й перевірки лабораторної роботи."""
    demonstrate_vector_norms()
    demonstrate_scaling_and_centering()
    demonstrate_iris()
    print("\nУсі перевірки збігу результатів пройдено.")


if __name__ == "__main__":
    main()
