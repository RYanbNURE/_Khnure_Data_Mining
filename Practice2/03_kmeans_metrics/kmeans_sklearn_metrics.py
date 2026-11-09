"""K-Means зі scikit-learn: inertia, silhouette та ARI після fit."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

DATA = np.asarray([[1.0], [2.0], [8.0], [9.0]], dtype=np.float64)
REFERENCE_LABELS = np.asarray(["A", "A", "B", "B"])
CLUSTER_COUNTS = (1, 2, 3, 4)
RANDOM_STATE = 7


def fit_kmeans(
    data: np.ndarray,
    cluster_count: int,
    random_state: int = RANDOM_STATE,
    n_init: int = 20,
) -> tuple[KMeans, np.ndarray]:
    """Навчає K-Means і повертає модель та номери кластерів."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Ознаки мають бути непорожньою двовимірною матрицею.")
    if (
        isinstance(cluster_count, bool)
        or not isinstance(cluster_count, (int, np.integer))
        or not 1 <= int(cluster_count) <= len(values)
    ):
        raise ValueError("K має бути цілим числом від 1 до кількості точок.")
    if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 1:
        raise ValueError("n_init має бути додатним цілим числом.")

    model = KMeans(
        n_clusters=int(cluster_count),
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
        max_iter=100,
    )
    cluster_ids = model.fit_predict(values)
    return model, cluster_ids


def adjusted_rand_after_fit(
    true_labels: np.ndarray,
    cluster_ids: np.ndarray,
) -> float:
    """Обчислює ARI після fit і перевіряє відповідність двох масивів."""
    labels = np.asarray(true_labels)
    clusters = np.asarray(cluster_ids)
    if labels.ndim != 1 or clusters.ndim != 1 or labels.size == 0:
        raise ValueError("Обидва аргументи мають бути непорожніми векторами.")
    if labels.shape != clusters.shape:
        raise ValueError("Справжні мітки й кластери мають однакову довжину.")
    return float(adjusted_rand_score(labels, clusters))


def run_one_k(cluster_count: int) -> None:
    """Навчає один K-Means і друкує проміжні та фінальні результати."""
    print("\n" + "-" * 72)
    print(f"ЕКСПЕРИМЕНТ ДЛЯ K={cluster_count}")
    print("-" * 72)
    print("До fit передаємо тільки DATA. Мітки A/B не передаються алгоритму.")

    # fit_predict отримує лише ознаки; міток A/B у цьому виклику немає.
    model, cluster_ids = fit_kmeans(
        DATA,
        cluster_count,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    distances_to_centers = model.transform(DATA)

    print(f"  Центри після навчання: {model.cluster_centers_.ravel().tolist()}")
    print(f"  Номери кластерів: {cluster_ids.tolist()}")
    print("  Точки та відстань до кожного знайденого центра:")
    for row_index, (point, cluster_id, distances) in enumerate(
        zip(DATA.ravel(), cluster_ids, distances_to_centers),
        start=1,
    ):
        formatted = ", ".join(
            f"центр {center_index + 1}: {distance:.3f}"
            for center_index, distance in enumerate(distances)
        )
        print(
            f"    рядок {row_index}: x={point:g}; {formatted}; "
            f"обрано кластер {cluster_id + 1}"
        )

    print(f"  Inertia = {model.inertia_:.6f}")
    unique_cluster_count = np.unique(cluster_ids).size
    if 2 <= unique_cluster_count < len(DATA):
        score = silhouette_score(DATA, cluster_ids)
        print(f"  Silhouette = {score:.6f} (внутрішня оцінка, мітки A/B не потрібні)")
    else:
        print(
            "  Silhouette не обчислюємо: потрібно щонайменше 2 кластери "
            "і менше кластерів, ніж точок."
        )

    # Справжні мітки використовуються тільки зараз, після завершення fit.
    ari = adjusted_rand_after_fit(REFERENCE_LABELS, cluster_ids)
    print(f"  ARI = {ari:.6f} (зовнішнє порівняння після fit)")


def main() -> None:
    """Порівнює кілька значень K на чотирьох одновимірних точках."""
    print("=" * 72)
    print("ПРАКТИКА 2.3 — K-MEANS, SILHOUETTE ТА ARI")
    print("=" * 72)
    print(f"Ознаки: {DATA.ravel().tolist()}")
    print(f"Мітки для зовнішньої перевірки: {REFERENCE_LABELS.tolist()}")
    print("Важливо: мітки не використовуються при fit K-Means.")
    print("Параметри відтворюваності: random_state=7, n_init=20.")

    for cluster_count in CLUSTER_COUNTS:
        run_one_k(cluster_count)

    print("\nПІДСУМОК")
    print("  Inertia зазвичай зменшується зі збільшенням K.")
    print("  Silhouette допомагає оцінити відокремленість без справжніх міток.")
    print("  ARI порівнює готове розбиття з A/B, але не має керувати fit.")


if __name__ == "__main__":
    main()
