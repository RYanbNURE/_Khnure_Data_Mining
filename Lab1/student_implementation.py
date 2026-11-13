"""Шаблон для самостійного виконання лабораторної роботи № 1."""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import norm as scipy_norm
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

NORM_ORDERS = (1, 2, 3)
TOLERANCE = 1e-6


def validate_order(order: int) -> None:
    """Перевіряє порядок норми."""
    if order not in NORM_ORDERS:
        raise ValueError("Порядок норми має дорівнювати 1, 2 або 3.")


def manual_lp_norm_numpy(vector: np.ndarray, order: int) -> float:
    """Обчислює норму вектора за формулою за допомогою NumPy."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")

    # Сума модулів координат у степені p, з якої добувається корінь степеня p.
    powered_sum = np.sum(np.abs(vector) ** order)
    return float(powered_sum ** (1.0 / order))


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


def vector_norm_scipy(vector: np.ndarray, order: int) -> float:
    """Обчислює норму вектора за допомогою SciPy."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")

    return float(scipy_norm(vector, ord=order))


def scale_features(
    data: np.ndarray,
    feature_range: tuple[float, float],
) -> np.ndarray:
    """Масштабує ознаки за допомогою scikit-learn."""
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit_transform(data)


def row_norms_sklearn(data: np.ndarray, order: int) -> np.ndarray:
    """Обчислює відстань рядків до нуля через scikit-learn."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")

    # Нульовий вектор має ту саму кількість ознак, що й дані.
    origin = np.zeros((1, data.shape[1]), dtype=data.dtype)
    distances = pairwise_distances(data, origin, metric="minkowski", p=order)
    return distances.ravel()


def manual_lp_norm_torch(vector: torch.Tensor, order: int) -> torch.Tensor:
    """Обчислює норму тензора за формулою за допомогою PyTorch."""
    validate_order(order)
    if vector.ndim != 1:
        raise ValueError("Очікувався одновимірний вектор.")

    powered_sum = torch.sum(torch.abs(vector) ** order)
    return powered_sum ** (1.0 / order)


def row_norms_torch(data: torch.Tensor, order: int) -> torch.Tensor:
    """Обчислює норму кожного рядка за допомогою PyTorch."""
    validate_order(order)
    if data.ndim != 2:
        raise ValueError("Очікувалася двовимірна матриця.")

    return torch.linalg.vector_norm(data, ord=order, dim=1)


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

        # Відмінність кожної бібліотеки від NumPy не має перевищувати 1e-6.
        numpy_scores = scores["NumPy"]
        print(f"  Норма L{order}:")
        for library, library_scores in scores.items():
            np.testing.assert_allclose(
                library_scores,
                numpy_scores,
                rtol=0.0,
                atol=TOLERANCE,
            )
            top_index = int(np.argmax(library_scores))
            max_difference = float(np.max(np.abs(library_scores - numpy_scores)))
            print(
                f"    {library}: рядок={top_index}, "
                f"оцінка={library_scores[top_index]:.4f}, "
                f"різниця з NumPy={max_difference:.2e}"
            )
        results_by_order[order] = numpy_scores

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

    # Індекси п'яти найбільших оцінок у порядку спадання.
    largest_indices = np.argsort(scores)[-5:][::-1]
    return [
        (int(index), str(target_names[labels[index]]), float(scores[index]))
        for index in largest_indices
    ]


def print_vector_norms(vector: np.ndarray) -> None:
    """Порівнює норми вектора в NumPy, SciPy, scikit-learn і PyTorch."""
    tensor = torch.as_tensor(vector.copy(), dtype=torch.float64)
    print(f"\nВектор {vector.tolist()}:")
    for order in NORM_ORDERS:
        values = {
            "власна NumPy": manual_lp_norm_numpy(vector, order),
            "NumPy": float(np.linalg.norm(vector, ord=order)),
            "SciPy": vector_norm_scipy(vector, order),
            "scikit-learn": float(
                row_norms_sklearn(vector.reshape(1, -1), order)[0]
            ),
            "PyTorch": torch.linalg.vector_norm(tensor, ord=order).item(),
            "власна PyTorch": manual_lp_norm_torch(tensor, order).item(),
        }
        np.testing.assert_allclose(
            list(values.values()),
            values["NumPy"],
            rtol=0.0,
            atol=TOLERANCE,
        )
        formatted = ", ".join(f"{name}={value:.6f}" for name, value in values.items())
        print(f"  L{order}: {formatted}")


def main() -> None:
    """Виконує завдання лабораторної роботи в заданій послідовності."""
    # Завдання 1: ручний розрахунок для (3, -4, 12) дає L1=19, L2=13, L3≈12.2071.
    sample = np.array([3.0, -4.0, 12.0])
    print("Контрольний вектор:", sample)

    # Завдання 2: власна функція NumPy порівнюється з np.linalg.norm.
    for order in NORM_ORDERS:
        print(f"L{order} NumPy:", np.linalg.norm(sample, ord=order))
        print(f"L{order} власна реалізація:", manual_lp_norm_numpy(sample, order))

    # Завдання 3: норма вектора SciPy.
    for order in NORM_ORDERS:
        print(f"L{order} SciPy:", vector_norm_scipy(sample, order))

    # Завдання 5: тензор із sample, вбудована та власна функції PyTorch.
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

    # Завдання 2–5: порівняння тестових векторів у всіх бібліотеках.
    for vector in (sample, np.array([1.0, -2.0, 2.0])):
        print_vector_norms(vector)

    iris = load_iris()
    features = iris.data[:, 2:4]

    # Завдання 4: порівняння результатів для обох діапазонів масштабування.
    scaled = scale_features(features, feature_range=(0.0, 1.0))
    compare_libraries(scaled, "Масштабовані ознаки Iris, діапазон [0, 1]")
    scaled_signed = scale_features(features, feature_range=(-1.0, 1.0))
    compare_libraries(scaled_signed, "Масштабовані ознаки Iris, діапазон [-1, 1]")

    # Завдання 6: центрування масштабованих ознак відносно середнього.
    centered = scaled - scaled.mean(axis=0)
    centered_scores = compare_libraries(centered, "Центровані ознаки Iris")

    # Завдання 7: п'ять найбільших відхилень від середнього для кожної норми.
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
