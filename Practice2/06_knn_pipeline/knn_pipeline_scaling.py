"""k-NN із Pipeline: детальний лог масштабування, сусідів і тестових прогнозів."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
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
        [0.4, 0.4],
        [3.0, 3.0],
        [3.2, 3.1],
        [2.8, 3.2],
        [3.4, 2.9],
        [3.1, 2.7],
    ],
    dtype=np.float64,
)
LABELS = np.asarray(["A"] * 5 + ["B"] * 5)
FEATURE_NAMES = ("ознака x₁", "ознака x₂")
RANDOM_STATE = 7


def main() -> None:
    """Навчає scaler лише на train і пояснює кожен тестовий прогноз."""
    print("=" * 76)
    print("ПРАКТИКА 2.6 — k-NN PIPELINE ТА МАСШТАБУВАННЯ")
    print("=" * 76)
    print("Мінінабір має дві ознаки й два класи A/B.")
    for row_index, (features, label) in enumerate(zip(FEATURES, LABELS), start=1):
        print(
            f"  рядок {row_index:02d}: x₁={features[0]:.2f}, "
            f"x₂={features[1]:.2f}, клас={label}"
        )

    # Стратифікація зберігає співвідношення класів у train та test.
    x_train, x_test, y_train, y_test = train_test_split(
        FEATURES,
        LABELS,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )

    print("\nРОЗБИТТЯ ДАНИХ")
    print(f"  random_state={RANDOM_STATE}, test_size=0.4, stratify=LABELS")
    print(f"  train: {len(x_train)} точок; класи {y_train.tolist()}")
    print(f"  test:  {len(x_test)} точок; класи приховані від fit")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=3,
                    metric="minkowski",
                    p=2,
                    weights="uniform",
                ),
            ),
        ]
    )

    print("\nНАВЧАННЯ PIPELINE")
    print("  Крок 1: StandardScaler обчислює параметри тільки з x_train.")
    # Pipeline навчає scaler на x_train і передає перетворені дані класифікатору.
    pipeline.fit(x_train, y_train)
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    classifier: KNeighborsClassifier = pipeline.named_steps["knn"]
    print(f"  Середні train-ознак: {scaler.mean_.round(4).tolist()}")
    print(f"  Масштаби train-ознак: {scaler.scale_.round(4).tolist()}")
    print(
        f"  k-NN: k={classifier.n_neighbors}, p={classifier.p}, "
        f"weights={classifier.weights}"
    )

    scaled_train = scaler.transform(x_train)
    print("\nМАСШТАБОВАНІ НАВЧАЛЬНІ ТОЧКИ")
    for index, (original, scaled, label) in enumerate(
        zip(x_train, scaled_train, y_train),
        start=1,
    ):
        print(
            f"  train {index}: {original.round(3).tolist()} -> "
            f"{scaled.round(3).tolist()}, клас={label}"
        )

    predictions: list[str] = []
    print("\nТЕСТОВІ ПРОГНОЗИ")
    for test_index, test_point in enumerate(x_test, start=1):
        scaled_query = scaler.transform(test_point.reshape(1, -1))
        distances, neighbor_indices = classifier.kneighbors(scaled_query)
        prediction = str(pipeline.predict(test_point.reshape(1, -1))[0])
        predictions.append(prediction)

        print(
            f"\n  Тестова точка {test_index}: "
            f"raw={test_point.round(3).tolist()}, "
            f"scaled={scaled_query[0].round(3).tolist()}"
        )
        neighbor_labels: list[str] = []
        for rank, (distance, train_index) in enumerate(
            zip(distances[0], neighbor_indices[0]),
            start=1,
        ):
            neighbor_label = str(y_train[train_index])
            neighbor_labels.append(neighbor_label)
            print(
                f"    сусід {rank}: train[{train_index}] "
                f"{x_train[train_index].round(3).tolist()}, "
                f"клас={neighbor_label}, d₂={distance:.4f}"
            )
        print(f"    голоси: {neighbor_labels}")
        print(f"    прогноз: {prediction}; правильна відповідь: {y_test[test_index - 1]}")

    accuracy = accuracy_score(y_test, predictions)
    print("\nПІДСУМОК")
    print(f"  Передбачення: {predictions}")
    print(f"  Справжні класи: {y_test.tolist()}")
    print(f"  Accuracy на малому test-наборі: {accuracy:.3f}")
    print(
        "  Важливо: scaler застосовує до test ті самі mean/scale, "
        "які були обчислені на train."
    )
    print(f"  Ознаки у прикладі: {', '.join(FEATURE_NAMES)}.")


if __name__ == "__main__":
    main()
