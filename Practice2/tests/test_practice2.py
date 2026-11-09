"""Автоматичні перевірки ручних формул і навчальних прикладів Practice2."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_example(module_name: str, relative_path: str) -> ModuleType:
    """Завантажує автономний скрипт з каталогу прикладу для тестування функцій."""
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не вдалося завантажити модуль {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DISTANCE = load_example(
    "practice2_distance",
    "01_minkowski_distance/minkowski_distance.py",
)
MANUAL_KMEANS = load_example(
    "practice2_manual_kmeans",
    "02_kmeans_manual/kmeans_manual_1d.py",
)
KMEANS_METRICS = load_example(
    "practice2_kmeans_metrics",
    "03_kmeans_metrics/kmeans_sklearn_metrics.py",
)
KNN_P = load_example(
    "practice2_knn_p",
    "04_knn_distance_p/knn_distance_p.py",
)
KNN_VOTING = load_example(
    "practice2_knn_voting",
    "05_knn_voting/knn_vote_k.py",
)
KNN_SCALING = load_example(
    "practice2_knn_scaling",
    "08_knn_scaling_effect/knn_scaling_effect.py",
)
KNN_WEIGHTED = load_example(
    "practice2_knn_weighted",
    "09_knn_weighted_vote/knn_weighted_vote.py",
)
KMEANS_INIT = load_example(
    "practice2_kmeans_init",
    "10_kmeans_initialization/kmeans_initialization.py",
)


@pytest.mark.parametrize(
    ("order", "expected"),
    ((1, 7.0), (2, 5.0), (3, 91.0 ** (1.0 / 3.0))),
)
def test_minkowski_manual_formula(
    order: int,
    expected: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    actual = DISTANCE.minkowski_distance_verbose(
        (1.0, 2.0),
        (4.0, 6.0),
        order,
    )
    capsys.readouterr()

    assert actual == pytest.approx(expected)


def test_minkowski_rejects_invalid_order_and_shape() -> None:
    with pytest.raises(ValueError, match="Порядок"):
        DISTANCE.minkowski_distance_verbose((1.0, 2.0), (3.0, 4.0), 4)
    with pytest.raises(ValueError, match="однакової розмірності"):
        DISTANCE.minkowski_distance_verbose((1.0, 2.0), (3.0,), 2)


def test_manual_kmeans_centers_and_inertia(
    capsys: pytest.CaptureFixture[str],
) -> None:
    points = MANUAL_KMEANS.DATA
    initial_centers = list(MANUAL_KMEANS.INITIAL_CENTERS)
    assignments = MANUAL_KMEANS.assign_points(
        points,
        initial_centers,
        iteration=1,
    )
    centers = MANUAL_KMEANS.update_centers(
        points,
        assignments,
        cluster_count=2,
    )
    inertia = MANUAL_KMEANS.calculate_inertia(points, assignments, centers)
    capsys.readouterr()

    assert assignments == [0, 0, 1, 1]
    assert centers == pytest.approx([1.5, 8.5])
    assert inertia == pytest.approx(1.0)


def test_kmeans_fixed_seed_is_reproducible_and_ari_ignores_cluster_names() -> None:
    first_model, first_ids = KMEANS_METRICS.fit_kmeans(
        KMEANS_METRICS.DATA,
        cluster_count=2,
        random_state=11,
        n_init=5,
    )
    second_model, second_ids = KMEANS_METRICS.fit_kmeans(
        KMEANS_METRICS.DATA,
        cluster_count=2,
        random_state=11,
        n_init=5,
    )

    np.testing.assert_array_equal(first_ids, second_ids)
    np.testing.assert_allclose(
        first_model.cluster_centers_,
        second_model.cluster_centers_,
    )
    assert first_model.inertia_ == pytest.approx(second_model.inertia_)
    assert KMEANS_METRICS.adjusted_rand_after_fit(
        np.asarray(["A", "A", "B", "B"]),
        np.asarray([7, 7, 3, 3]),
    ) == pytest.approx(1.0)
    assert KMEANS_METRICS.adjusted_rand_after_fit(
        np.asarray(["A", "A", "B", "B"]),
        np.asarray([0, 1, 0, 1]),
    ) == pytest.approx(-0.5)


@pytest.mark.parametrize("cluster_count", (0, 5, True))
def test_kmeans_rejects_invalid_cluster_count(cluster_count: Any) -> None:
    with pytest.raises(ValueError, match="K має бути"):
        KMEANS_METRICS.fit_kmeans(
            KMEANS_METRICS.DATA,
            cluster_count=cluster_count,
        )


def test_kmeans_ari_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="однакову довжину"):
        KMEANS_METRICS.adjusted_rand_after_fit(
            np.asarray(["A", "B"]),
            np.asarray([0]),
        )


def test_minkowski_order_changes_nearest_neighbor() -> None:
    nearest_by_order = {
        order: KNN_P.rank_neighbors(order, verbose=False)[0][1]
        for order in (1, 2, 3)
    }

    assert nearest_by_order == {1: "A1", 2: "B1", 3: "B1"}
    assert KNN_P.minkowski_distance((0.0, 0.0), (2.0, 2.0), 2) == pytest.approx(
        8.0**0.5
    )


def test_minkowski_example_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Порядок"):
        KNN_P.minkowski_distance((0.0, 0.0), (1.0, 1.0), 4)
    with pytest.raises(ValueError, match="однакової довжини"):
        KNN_P.minkowski_distance((0.0,), (1.0, 2.0), 2)


def test_uniform_knn_vote_handles_winner_and_tie(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ranked = [
        (8.0**0.5, "B1", (2.0, 2.0), "B"),
        (3.0, "A1", (0.0, 3.0), "A"),
        (4.0, "A2", (4.0, 0.0), "A"),
    ]

    assert KNN_VOTING.predict_by_vote(1, ranked) == "B"
    assert KNN_VOTING.predict_by_vote(2, ranked) is None
    assert KNN_VOTING.predict_by_vote(3, ranked) == "A"
    capsys.readouterr()


@pytest.mark.parametrize("neighbor_count", (0, 4, True))
def test_uniform_knn_vote_rejects_invalid_k(neighbor_count: Any) -> None:
    ranked = [(1.0, "A1", (1.0, 0.0), "A")]
    with pytest.raises(ValueError, match="k має бути"):
        KNN_VOTING.predict_by_vote(neighbor_count, ranked)


def test_scaling_can_change_the_nearest_neighbor() -> None:
    raw_query, _, raw_ranked, _ = KNN_SCALING.rank_neighbors(
        KNN_SCALING.QUERY,
        KNN_SCALING.TRAINING_POINTS,
        "none",
    )
    minmax_query, _, minmax_ranked, _ = KNN_SCALING.rank_neighbors(
        KNN_SCALING.QUERY,
        KNN_SCALING.TRAINING_POINTS,
        "minmax",
    )
    standard_query, _, standard_ranked, _ = KNN_SCALING.rank_neighbors(
        KNN_SCALING.QUERY,
        KNN_SCALING.TRAINING_POINTS,
        "standard",
    )

    assert raw_query.tolist() == [0.0, 0.0]
    assert raw_ranked[0][1] == "B1"
    assert minmax_ranked[0][1] == "A1"
    assert standard_ranked[0][1] == "A1"
    assert not np.array_equal(minmax_query, standard_query)


def test_scaling_example_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="none, minmax або standard"):
        KNN_SCALING.rank_neighbors(
            KNN_SCALING.QUERY,
            KNN_SCALING.TRAINING_POINTS,
            "unknown",
        )


def test_weighted_vote_changes_winner_and_handles_zero_distance() -> None:
    uniform_winner, uniform_votes = KNN_WEIGHTED.calculate_votes(
        KNN_WEIGHTED.NONZERO_NEIGHBORS,
        k=3,
        weighting="uniform",
    )
    distance_winner, distance_votes = KNN_WEIGHTED.calculate_votes(
        KNN_WEIGHTED.NONZERO_NEIGHBORS,
        k=3,
        weighting="distance",
    )
    zero_winner, zero_votes = KNN_WEIGHTED.calculate_votes(
        KNN_WEIGHTED.NEIGHBORS_WITH_EXACT_MATCH,
        k=4,
        weighting="distance",
    )
    uniform_tie, _ = KNN_WEIGHTED.calculate_votes(
        KNN_WEIGHTED.NEIGHBORS_WITH_EXACT_MATCH,
        k=4,
        weighting="uniform",
    )

    assert uniform_winner == "A"
    assert uniform_votes == {"A": 2.0, "B": 1.0}
    assert distance_winner == "B"
    assert distance_votes == pytest.approx({"A": 1.5, "B": 4.0})
    assert zero_winner == "B"
    assert zero_votes == {"B": 1.0}
    assert uniform_tie is None


def test_weighted_vote_rejects_invalid_weighting_and_negative_distance() -> None:
    with pytest.raises(ValueError, match="weighting"):
        KNN_WEIGHTED.calculate_votes(KNN_WEIGHTED.NONZERO_NEIGHBORS, 2, "other")
    with pytest.raises(ValueError, match="невід'ємними"):
        KNN_WEIGHTED.calculate_votes(((-1.0, "A", "A"),), 1, "uniform")


def test_kmeans_initialization_reproducibility_and_outlier_sensitivity() -> None:
    first = KMEANS_INIT.fit_kmeans(
        KMEANS_INIT.SEED_DATA,
        cluster_count=3,
        random_state=13,
        n_init=5,
    )
    second = KMEANS_INIT.fit_kmeans(
        KMEANS_INIT.SEED_DATA,
        cluster_count=3,
        random_state=13,
        n_init=5,
    )
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
    assert first.inertia_ == pytest.approx(second.inertia_)

    center_without_outlier, inertia_without_outlier = KMEANS_INIT.centroid_and_inertia(
        (0.0, 1.0, 2.0, 3.0, 4.0)
    )
    center_with_outlier, inertia_with_outlier = KMEANS_INIT.centroid_and_inertia(
        (0.0, 1.0, 2.0, 3.0, 4.0, 20.0)
    )
    assert center_without_outlier == pytest.approx(2.0)
    assert inertia_without_outlier == pytest.approx(10.0)
    assert center_with_outlier == pytest.approx(5.0)
    assert inertia_with_outlier == pytest.approx(280.0)


def test_kmeans_initialization_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="K має бути"):
        KMEANS_INIT.fit_kmeans(KMEANS_INIT.SEED_DATA, 0, random_state=7, n_init=1)
    with pytest.raises(ValueError, match="n_init"):
        KMEANS_INIT.fit_kmeans(KMEANS_INIT.SEED_DATA, 2, random_state=7, n_init=0)


def test_kmeans_splits_long_line_because_k_is_forced() -> None:
    elongated = np.asarray([[float(value), 0.0] for value in range(10)])
    fixed_centers = np.asarray([[2.0, 0.0], [7.0, 0.0]])
    model = KMEANS_INIT.fit_kmeans(
        elongated,
        cluster_count=2,
        random_state=7,
        n_init=1,
        init=fixed_centers,
    )

    assert np.unique(model.labels_).size == 2
    np.testing.assert_array_equal(np.bincount(model.labels_), np.asarray([5, 5]))
