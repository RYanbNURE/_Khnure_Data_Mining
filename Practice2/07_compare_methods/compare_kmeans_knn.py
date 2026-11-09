"""Порівнює K-Means та k-NN на одному простому наборі точок."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = np.asarray(
    [
        [0.0, 0.0],
        [0.2, 0.1],
        [0.1, 0.4],
        [0.5, 0.2],
        [3.0, 3.0],
        [3.2, 3.1],
        [2.8, 3.2],
        [3.4, 2.9],
    ],
    dtype=np.float64,
)
LABELS = np.asarray(["A", "A", "A", "A", "B", "B", "B", "B"])
QUERY = np.asarray([[0.15, 0.2]], dtype=np.float64)
RANDOM_STATE = 7


def main() -> None:
    """Навчає обидва методи та пояснює різницю між їхніми відповідями."""
    print("=" * 76)
    print("ПРАКТИКА 2.7 — K-MEANS І k-NN НА ОДНИХ ОЗНАКАХ")
    print("=" * 76)
    print("Таблиця даних (перед fit K-Means мітки не використовуються):")
    for index, (point, label) in enumerate(zip(FEATURES, LABELS), start=1):
        print(
            f"  {index}: x={point.round(2).tolist()}, "
            f"довідкова мітка={label}"
        )

    print("\nЧАСТИНА A — K-MEANS, БЕЗ УЧИТЕЛЯ")
    clustering_scaler = StandardScaler()
    scaled_features = clustering_scaler.fit_transform(FEATURES)
    clusterer = KMeans(
        n_clusters=2,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    # K-Means бачить X, але LABELS не передаються до fit_predict.
    cluster_ids = clusterer.fit_predict(scaled_features)
    centers_original_units = clustering_scaler.inverse_transform(
        clusterer.cluster_centers_
    )

    print("  На fit передано тільки масштабовані ознаки, не LABELS.")
    print(f"  Центри у стандартизованому просторі: {clusterer.cluster_centers_.round(3).tolist()}")
    print(f"  Центри у початкових одиницях: {centers_original_units.round(3).tolist()}")
    print(f"  Inertia: {clusterer.inertia_:.6f}")
    for index, (point, cluster_id) in enumerate(zip(FEATURES, cluster_ids), start=1):
        print(f"  Точка {index} {point.round(2).tolist()} -> cluster ID {cluster_id}")

    # Мітки підключаємо тільки для зовнішньої оцінки після кластеризації.
    ari = adjusted_rand_score(LABELS, cluster_ids)
    print(f"  ARI після fit (зовнішнє порівняння з LABELS): {ari:.6f}")

    print("\nЧАСТИНА B — k-NN, З УЧИТЕЛЕМ")
    x_train, x_test, y_train, y_test = train_test_split(
        FEATURES,
        LABELS,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=3,
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )
    # На відміну від K-Means, k-NN навчається на ознаках та відомих класах.
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    print(f"  Train: {len(x_train)} точок, класи {y_train.tolist()}")
    print(f"  Test:  {len(x_test)} точок")
    print("  Для кожної test-точки показуємо сусідів і результат голосування:")

    scaler: StandardScaler = classifier.named_steps["scaler"]
    knn: KNeighborsClassifier = classifier.named_steps["knn"]
    for point_index, (point, actual, predicted) in enumerate(
        zip(x_test, y_test, predictions),
        start=1,
    ):
        transformed = scaler.transform(point.reshape(1, -1))
        distances, neighbor_indices = knn.kneighbors(transformed)
        neighbors = [
            str(y_train[neighbor_index])
            for neighbor_index in neighbor_indices[0]
        ]
        print(
            f"  test {point_index}: x={point.round(2).tolist()}, "
            f"сусіди={neighbors}, прогноз={predicted}, істинний клас={actual}"
        )

    accuracy = accuracy_score(y_test, predictions)
    print(f"  Accuracy на test: {accuracy:.3f}")

    scaled_query = clustering_scaler.transform(QUERY)
    query_cluster = int(clusterer.predict(scaled_query)[0])
    query_class = str(classifier.predict(QUERY)[0])
    print("\nОДНА НОВА ТОЧКА Q=(0.15, 0.20)")
    print(f"  K-Means повертає cluster ID {query_cluster}.")
    print(f"  k-NN повертає клас {query_class}.")
    print(
        "  Це різні типи відповідей: кластер — номер групи, "
        "клас — мітка, вивчена з прикладів."
    )


if __name__ == "__main__":
    main()
