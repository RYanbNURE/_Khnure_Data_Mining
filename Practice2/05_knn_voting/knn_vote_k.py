"""Покрокове голосування k-NN для k=1, k=2 та k=3."""

from __future__ import annotations

from collections import Counter
from math import fsum

Point = tuple[float, ...]
TRAINING_POINTS: tuple[tuple[str, Point, str], ...] = (
    ("A1", (0.0, 3.0), "A"),
    ("B1", (2.0, 2.0), "B"),
    ("A2", (4.0, 0.0), "A"),
)
QUERY: Point = (0.0, 0.0)
ORDER = 2


def euclidean_distance_with_log(query: Point, point: Point) -> float:
    """Друкує квадрати різниць і повертає евклідову відстань."""
    squared_differences: list[float] = []
    for axis, (query_value, point_value) in enumerate(zip(query, point), start=1):
        squared = (query_value - point_value) ** 2
        squared_differences.append(squared)
        print(
            f"      вісь {axis}: ({query_value:g}-{point_value:g})² = {squared:g}"
        )

    squared_sum = fsum(squared_differences)
    distance = squared_sum**0.5
    print(f"      d₂ = √({squared_sum:g}) = {distance:.6f}")
    return distance


def predict_by_vote(
    k: int,
    ranked_points: list[tuple[float, str, Point, str]],
) -> str | None:
    """Показує голоси та повертає клас або None, якщо є нічия."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(ranked_points):
        raise ValueError("k має бути цілим числом від 1 до кількості сусідів.")
    if not ranked_points:
        raise ValueError("Для голосування потрібен хоча б один сусід.")

    neighbors = ranked_points[:k]
    # Counter рахує голоси лише тих сусідів, яких відібрали для цього k.
    votes = Counter(class_name for _, _, _, class_name in neighbors)
    highest_vote_count = max(votes.values())
    winners = sorted(
        class_name
        for class_name, vote_count in votes.items()
        if vote_count == highest_vote_count
    )

    print(f"\nГОЛОСУВАННЯ ДЛЯ k={k}")
    for rank, (distance, name, point, class_name) in enumerate(neighbors, start=1):
        print(
            f"  сусід {rank}: {name}={point}, d₂={distance:.6f}, "
            f"голос за {class_name}"
        )
    print(f"  Підрахунок голосів: {dict(sorted(votes.items()))}")

    # Нічия залишається явним результатом, а не розв'язується випадково.
    if len(winners) == 1:
        print(f"  Переможець: клас {winners[0]} ({highest_vote_count} голосів).")
        return winners[0]
    else:
        print(
            f"  НІЧИЯ: класи {winners} отримали по {highest_vote_count} голосу."
        )
        return None


def main() -> None:
    """Рахує сусідів один раз і повторює голосування для різних k."""
    print("=" * 72)
    print("ПРАКТИКА 2.5 — ЯК k ЗМІНЮЄ ГОЛОСУВАННЯ")
    print("=" * 72)
    print(f"Точка запиту Q={QUERY}; метрика L{ORDER}.")

    ranked_points: list[tuple[float, str, Point, str]] = []
    for name, point, class_name in TRAINING_POINTS:
        print(f"\nВІДСТАНЬ ДО {name}={point}, клас {class_name}")
        distance = euclidean_distance_with_log(QUERY, point)
        ranked_points.append((distance, name, point, class_name))

    ranked_points.sort()
    print("\nВІДСОРТОВАНИЙ СПИСОК")
    for rank, (distance, name, point, class_name) in enumerate(ranked_points, start=1):
        print(
            f"  {rank}. {name}: координати {point}, клас {class_name}, "
            f"d₂={distance:.6f}"
        )

    for neighbor_count in (1, 2, 3):
        predict_by_vote(neighbor_count, ranked_points)

    print("\nПІДСУМОК")
    print("  k=1: один голос B.")
    print("  k=2: один голос A та один голос B — нічия.")
    print("  k=3: два голоси A та один голос B — перемагає A.")


if __name__ == "__main__":
    main()
