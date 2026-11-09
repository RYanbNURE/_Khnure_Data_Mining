"""Автоматизовані перевірки еталонного рішення."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import solution as implementation


def test_minkowski_distance_matches_independent_reference(
    vector_pair: tuple[np.ndarray, np.ndarray],
    minkowski_order: int,
    reference_minkowski_distance,
) -> None:
    first, second = vector_pair
    actual = implementation.minkowski_distance(first, second, minkowski_order)
    expected = reference_minkowski_distance(first, second, minkowski_order)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_minkowski_distance_rejects_invalid_order(
    vector_pair: tuple[np.ndarray, np.ndarray],
    invalid_order: int,
) -> None:
    first, second = vector_pair

    with pytest.raises(ValueError, match="Порядок Мінковського"):
        implementation.minkowski_distance(first, second, invalid_order)


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.asarray([1.0, 2.0]), np.asarray([1.0])),
        (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)),
    ),
)
def test_minkowski_distance_rejects_invalid_vectors(
    first: np.ndarray,
    second: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        implementation.minkowski_distance(first, second, order=2)


def test_minkowski_distance_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="скінченні"):
        implementation.minkowski_distance(
            np.asarray([1.0, np.inf]),
            np.asarray([0.0, 2.0]),
            order=2,
        )


def test_scale_features_matches_independent_reference(
    features_data: np.ndarray,
    scaling_method: str,
    reference_scale_features,
) -> None:
    original = features_data.copy()
    actual = implementation.scale_features(features_data, scaling_method)
    expected = reference_scale_features(features_data, scaling_method)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(features_data, original)
    assert actual is not features_data


def test_scale_features_rejects_unknown_method(features_data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Масштабування"):
        implementation.scale_features(features_data, "robust")


def test_run_kmeans_is_reproducible_and_returns_scaled_data(
    features_data: np.ndarray,
    reference_scale_features,
) -> None:
    scaled, first_run = implementation.run_kmeans(
        features_data,
        n_clusters=3,
        scaling_method="minmax",
        random_state=11,
    )
    repeated_scaled, second_run = implementation.run_kmeans(
        features_data,
        n_clusters=3,
        scaling_method="minmax",
        random_state=11,
    )

    np.testing.assert_allclose(
        scaled,
        reference_scale_features(features_data, "minmax"),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_array_equal(scaled, repeated_scaled)
    np.testing.assert_array_equal(first_run, second_run)
    assert np.unique(first_run).size == 3


@pytest.mark.parametrize("n_clusters", (1, 0, 9, 2.5, True))
def test_run_kmeans_rejects_invalid_cluster_count(
    features_data: np.ndarray,
    n_clusters: int,
) -> None:
    with pytest.raises(ValueError, match="Кількість кластерів"):
        implementation.run_kmeans(features_data, n_clusters=n_clusters)


def test_silhouette_matches_independent_reference(
    cluster_case: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_silhouette,
) -> None:
    data, cluster_ids, _ = cluster_case
    actual = implementation.silhouette_for_clusters(data, cluster_ids)
    expected = reference_silhouette(data, cluster_ids)

    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-9)


@pytest.mark.parametrize(
    "cluster_ids",
    (
        np.asarray([0, 0, 0, 0, 0, 0]),
        np.asarray([0, 1, 2, 3, 4, 5]),
    ),
)
def test_silhouette_rejects_invalid_cluster_counts(
    cluster_case: tuple[np.ndarray, np.ndarray, np.ndarray],
    cluster_ids: np.ndarray,
) -> None:
    data, _, _ = cluster_case

    with pytest.raises(ValueError, match="Silhouette"):
        implementation.silhouette_for_clusters(data, cluster_ids)


def test_silhouette_rejects_mismatched_data_and_labels(
    cluster_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    data, cluster_ids, _ = cluster_case

    with pytest.raises(ValueError, match="збігатися"):
        implementation.silhouette_for_clusters(data[:-1], cluster_ids)


def test_adjusted_rand_is_invariant_to_cluster_numbering(
    cluster_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, cluster_ids, species_labels = cluster_case
    permuted_cluster_ids = np.asarray([2, 2, 0, 0, 1, 1])

    assert implementation.adjusted_rand_for_species(
        permuted_cluster_ids,
        species_labels,
    ) == pytest.approx(1.0)
    assert implementation.adjusted_rand_for_species(
        cluster_ids,
        species_labels,
    ) == pytest.approx(1.0)


def test_adjusted_rand_rejects_mismatched_labels(
    cluster_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, cluster_ids, species_labels = cluster_case

    with pytest.raises(ValueError, match="збігатися"):
        implementation.adjusted_rand_for_species(cluster_ids, species_labels[:-1])


def test_split_data_is_stratified_and_reproducible(
    iris_petal_features: np.ndarray,
    iris_dataset,
) -> None:
    first_split = implementation.split_data(iris_petal_features, iris_dataset.target)
    repeated_split = implementation.split_data(iris_petal_features, iris_dataset.target)
    x_train, x_test, y_train, y_test = first_split

    assert x_train.shape == (120, 2)
    assert x_test.shape == (30, 2)
    np.testing.assert_array_equal(x_train, repeated_split[0])
    np.testing.assert_array_equal(x_test, repeated_split[1])
    np.testing.assert_array_equal(y_train, repeated_split[2])
    np.testing.assert_array_equal(y_test, repeated_split[3])
    np.testing.assert_array_equal(np.bincount(y_train), np.asarray([40, 40, 40]))
    np.testing.assert_array_equal(np.bincount(y_test), np.asarray([10, 10, 10]))


def test_split_data_rejects_mismatched_labels(iris_petal_features: np.ndarray) -> None:
    with pytest.raises(ValueError, match="збігатися"):
        implementation.split_data(iris_petal_features, np.asarray([0, 1]))


def test_build_knn_pipeline_uses_requested_steps(
    scaling_method: str,
    minkowski_order: int,
) -> None:
    pipeline = implementation.build_knn_pipeline(
        scaling_method=scaling_method,
        n_neighbors=5,
        order=minkowski_order,
    )

    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps[0][0] == "scaler"
    assert pipeline.steps[1][0] == "knn"
    classifier = pipeline.named_steps["knn"]
    assert isinstance(classifier, KNeighborsClassifier)
    assert classifier.n_neighbors == 5
    assert classifier.p == minkowski_order
    assert classifier.metric == "minkowski"


def test_build_knn_pipeline_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="сусідів"):
        implementation.build_knn_pipeline(n_neighbors=0)
    with pytest.raises(ValueError, match="Порядок Мінковського"):
        implementation.build_knn_pipeline(order=4)


def test_search_knn_parameters_uses_stratified_macro_f1_cv(
    iris_petal_features: np.ndarray,
    iris_dataset,
) -> None:
    x_train, _, y_train, _ = implementation.split_data(
        iris_petal_features,
        iris_dataset.target,
    )
    search = implementation.search_knn_parameters(
        x_train,
        y_train,
        cv_splits=3,
    )

    assert isinstance(search.cv, StratifiedKFold)
    assert search.n_splits_ == 3
    assert search.scoring == "f1_macro"
    assert len(search.cv_results_["params"]) == 45
    assert 0.0 <= search.best_score_ <= 1.0
    assert set(search.best_params_) == {
        "scaler",
        "knn__n_neighbors",
        "knn__p",
    }
    assert isinstance(search.best_estimator_, Pipeline)


def test_search_knn_parameters_rejects_too_few_samples_per_class(
    cluster_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    data, _, labels = cluster_case

    with pytest.raises(ValueError, match="щонайменше"):
        implementation.search_knn_parameters(data, labels, cv_splits=3)


def test_evaluate_classifier_matches_independent_reference(
    reference_classification_metrics,
) -> None:
    true_labels = np.asarray([0, 0, 1, 1, 2, 2])
    predicted_labels = np.asarray([0, 1, 1, 1, 2, 0])
    classes = (0, 1, 2)

    actual = implementation.evaluate_classifier(
        true_labels,
        predicted_labels,
        class_labels=classes,
    )
    expected = reference_classification_metrics(
        true_labels,
        predicted_labels,
        classes,
    )

    assert actual["accuracy"] == pytest.approx(expected["accuracy"])
    assert actual["macro_f1"] == pytest.approx(expected["macro_f1"])
    np.testing.assert_array_equal(
        actual["confusion_matrix"],
        expected["confusion_matrix"],
    )


def test_evaluate_classifier_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="однакову довжину"):
        implementation.evaluate_classifier(
            np.asarray([0, 1, 2]),
            np.asarray([0, 1]),
        )


def test_plot_comparison_saves_image(
    iris_petal_features: np.ndarray,
    iris_dataset,
    tmp_path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    scaled, clusters = implementation.run_kmeans(
        iris_petal_features,
        n_clusters=3,
        scaling_method="minmax",
    )
    del scaled
    fitted_knn = implementation.build_knn_pipeline(
        scaling_method="minmax",
        n_neighbors=5,
        order=2,
    )
    fitted_knn.fit(iris_petal_features, iris_dataset.target)
    output_path = tmp_path / "comparison.png"

    implementation.plot_comparison(
        iris_petal_features,
        iris_dataset.target,
        iris_dataset.target_names,
        clusters,
        fitted_knn,
        save_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_main_reports_both_learning_approaches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    implementation.main()
    output = capsys.readouterr().out

    assert "K-Means" in output
    assert "Silhouette" in output
    assert "ARI" in output
    assert "k-NN" in output
    assert "Тестова accuracy" in output
    assert "Матриця помилок" in output
