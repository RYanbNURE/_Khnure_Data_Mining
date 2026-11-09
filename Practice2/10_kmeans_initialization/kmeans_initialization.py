"""Ілюструє вплив ініціалізації, викиду та форми груп на K-Means."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

SEED_DATA = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
SEEDS_TO_COMPARE = (0, 5)


def fit_kmeans(
    data: np.ndarray,
    cluster_count: int,
    random_state: int,
    n_init: int,
    init: str | np.ndarray = "k-means++",
) -> KMeans:
    """Навчає K-Means після перевірки форми даних і параметрів."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Ознаки мають бути непорожньою двовимірною матрицею.")
    if (
        isinstance(cluster_count, bool)
        or not isinstance(cluster_count, (int, np.integer))
        or not 1 <= int(cluster_count) <= len(values)
    ):
        raise ValueError("K має бути цілим числом від 1 до кількості спостережень.")
    if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 1:
        raise ValueError("n_init має бути додатним цілим числом.")

    model = KMeans(
        n_clusters=int(cluster_count),
        init=init,
        n_init=n_init,
        random_state=random_state,
        max_iter=100,
    )
    model.fit(values)
    return model


def centroid_and_inertia(values: tuple[float, ...]) -> tuple[float, float]:
    """Рахує середнє та суму квадратів відхилень для K=1."""
    if not values:
        raise ValueError("Потрібна хоча б одна точка.")
    centroid = sum(values) / len(values)
    inertia = sum((value - centroid) ** 2 for value in values)
    return centroid, inertia


def print_seed_run(seed: int, n_init: int) -> float:
    """Показує центри, призначення та inertia для одного запуску."""
    model = fit_kmeans(SEED_DATA, 3, random_state=seed, n_init=n_init)
    ordered_centers = np.sort(model.cluster_centers_.ravel())
    print(
        f"  random_state={seed}, n_init={n_init}: "
        f"центри={ordered_centers.tolist()}, inertia={model.inertia_:.3f}"
    )
    for point, cluster_id in zip(SEED_DATA.ravel(), model.labels_):
        print(f"    x={point:g} -> cluster ID {cluster_id}")
    return float(model.inertia_)


def main() -> None:
    """Виконує три малі експерименти про чутливість K-Means."""
    print("=" * 76)
    print("ПРАКТИКА 2.10 — ІНІЦІАЛІЗАЦІЯ ТА ОБМЕЖЕННЯ K-MEANS")
    print("=" * 76)

    print("\nЕКСПЕРИМЕНТ 1 — різні seed, один запуск на seed")
    print(f"  Точки: {SEED_DATA.ravel().tolist()}, K=3")
    for seed in SEEDS_TO_COMPARE:
        print_seed_run(seed, n_init=1)

    print("\nЕКСПЕРИМЕНТ 2 — повторні ініціалізації")
    print("  n_init=20 запускає кілька стартів і зберігає найменшу inertia.")
    print_seed_run(seed=7, n_init=20)

    print("\nЕКСПЕРИМЕНТ 3 — як викид зсуває середнє при K=1")
    without_outlier = (0.0, 1.0, 2.0, 3.0, 4.0)
    with_outlier = (*without_outlier, 20.0)
    for name, values in (
        ("без викиду", without_outlier),
        ("з викидом 20", with_outlier),
    ):
        centroid, inertia = centroid_and_inertia(values)
        print(
            f"  {name}: точки={values}, центр={centroid:.3f}, "
            f"inertia={inertia:.3f}"
        )
    print("  Один далекий об'єкт зміщує середнє та збільшує квадратичну помилку.")

    print("\nЕКСПЕРИМЕНТ 4 — K примусово ділить видовжену групу")
    elongated = np.asarray([[float(value), 0.0] for value in range(10)])
    fixed_centers = np.asarray([[2.0, 0.0], [7.0, 0.0]])
    model = fit_kmeans(
        elongated,
        cluster_count=2,
        random_state=7,
        n_init=1,
        init=fixed_centers,
    )
    print("  Точки лежать на одній видовженій лінії x=0...9, задано K=2.")
    print(f"  Знайдені центри: {model.cluster_centers_.round(3).tolist()}")
    for point, cluster_id in zip(elongated, model.labels_):
        print(f"    x={point[0]:g} -> cluster ID {cluster_id}")
    print(
        "  Алгоритм завжди повертає K груп; перевірте, чи мають вони "
        "предметний сенс."
    )


if __name__ == "__main__":
    main()
