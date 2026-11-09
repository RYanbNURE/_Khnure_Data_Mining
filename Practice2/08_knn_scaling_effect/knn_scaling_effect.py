"""Показує, як масштаб ознак змінює найближчого сусіда k-NN."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

QUERY = np.asarray([0.0, 0.0], dtype=np.float64)
POINT_NAMES = ("A1", "B1", "A2")
POINT_LABELS = ("A", "B", "A")
TRAINING_POINTS = np.asarray(
    [
        [1.0, 100.0],
        [4.0, 0.0],
        [3.0, 90.0],
    ],
    dtype=np.float64,
)
SCALING_METHODS = ("none", "minmax", "standard")


def rank_neighbors(
    query: np.ndarray,
    training_points: np.ndarray,
    scaling_method: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[float, str, str]],
    MinMaxScaler | StandardScaler | None,
]:
    """Масштабує train і query та повертає відсортованих сусідів."""
    features = np.asarray(training_points, dtype=np.float64)
    point = np.asarray(query, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("training_points мають бути непорожньою матрицею.")
    if point.shape != (features.shape[1],):
        raise ValueError("query має містити одне значення на кожну ознаку.")
    if len(POINT_NAMES) != len(features) or len(POINT_LABELS) != len(features):
        raise ValueError("Для кожної тренувальної точки потрібні назва та мітка.")
    if not np.isfinite(features).all() or not np.isfinite(point).all():
        raise ValueError("Координати мають бути скінченними числами.")

    scaler: MinMaxScaler | StandardScaler | None
    if scaling_method == "none":
        scaler = None
        scaled_training = features.copy()
        scaled_query = point.copy()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
        scaled_training = scaler.fit_transform(features)
        scaled_query = scaler.transform(point.reshape(1, -1))[0]
    elif scaling_method == "standard":
        scaler = StandardScaler()
        scaled_training = scaler.fit_transform(features)
        scaled_query = scaler.transform(point.reshape(1, -1))[0]
    else:
        raise ValueError("Метод має бути none, minmax або standard.")

    distances = np.linalg.norm(scaled_training - scaled_query, axis=1)
    ranked = sorted(
        (
            (float(distance), name, label)
            for distance, name, label in zip(distances, POINT_NAMES, POINT_LABELS)
        ),
        key=lambda row: (row[0], row[1]),
    )
    return scaled_query, scaled_training, ranked, scaler


def main() -> None:
    """Порівнює ранжування точок у початкових і масштабованих координатах."""
    print("=" * 76)
    print("ПРАКТИКА 2.8 — ВПЛИВ МАСШТАБУ НА k-NN")
    print("=" * 76)
    print(f"Точка запиту Q={QUERY.tolist()}")
    print("Тренувальні дані:")
    for name, point, label in zip(POINT_NAMES, TRAINING_POINTS, POINT_LABELS):
        print(f"  {name}: координати={point.tolist()}, клас={label}")
    print("Ознака x₂ має набагато більший числовий діапазон, ніж x₁.")

    for method in SCALING_METHODS:
        print(f"\n{'=' * 20} МАСШТАБУВАННЯ: {method} {'=' * 20}")
        scaled_query, scaled_training, ranked, scaler = rank_neighbors(
            QUERY,
            TRAINING_POINTS,
            method,
        )
        if scaler is None:
            print("  Scaler не використовується; координати лишаються без змін.")
        elif isinstance(scaler, MinMaxScaler):
            print(f"  Мінімум train за ознаками: {scaler.data_min_.round(3).tolist()}")
            print(f"  Максимум train за ознаками: {scaler.data_max_.round(3).tolist()}")
        else:
            print(f"  Середнє train за ознаками: {scaler.mean_.round(3).tolist()}")
            print(f"  Стандартне відхилення train: {scaler.scale_.round(3).tolist()}")

        print(f"  Масштабована Q: {scaled_query.round(4).tolist()}")
        print("  Відстані L2 до точок:")
        for distance, name, label in ranked:
            point_index = POINT_NAMES.index(name)
            transformed_point = scaled_training[point_index]
            print(
                f"    {name} ({label}): координати "
                f"{transformed_point.round(4).tolist()}, d₂={distance:.6f}"
            )
        print(
            f"  k=1 обирає {ranked[0][1]}, клас {ranked[0][2]}."
        )

    print("\nВИСНОВОК")
    print("  Без масштабування найближчий B1; після MinMax/Standard — A1.")
    print("  Масштабування змінює геометрію, тому його вибір є частиною експерименту.")
    print("  У справжній оцінці scaler треба fit лише на train усередині Pipeline.")


if __name__ == "__main__":
    main()
