"""Короткий приклад кластеризації та класифікації Iris."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 7


def main() -> None:
    """Показує один приклад K-Means та k-NN на ознаках пелюсток Iris."""
    iris = load_iris()
    features = np.asarray(iris.data[:, 2:4], dtype=np.float64)
    labels = iris.target

    scaled_features = MinMaxScaler().fit_transform(features)
    clusters = KMeans(
        n_clusters=3,
        n_init=20,
        random_state=RANDOM_STATE,
    ).fit_predict(scaled_features)

    print("K-Means: кластери навчаються без міток видів")
    print(f"Silhouette: {silhouette_score(scaled_features, clusters):.4f}")
    print(f"ARI після порівняння з видами Iris: {adjusted_rand_score(labels, clusters):.4f}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    classifier = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=5,
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    print("\nk-NN: класифікація за відомими мітками навчальних даних")
    print(f"Accuracy на тестовій вибірці: {accuracy_score(y_test, predictions):.4f}")
    print("Матриця помилок:")
    print(confusion_matrix(y_test, predictions, labels=np.arange(len(iris.target_names))))


if __name__ == "__main__":
    main()
