"""Порівнює рівне та зважене за відстанню голосування k-NN."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite

Neighbor = tuple[float, str, str]
NONZERO_NEIGHBORS: tuple[Neighbor, ...] = (
    (0.25, "B1", "B"),
    (1.0, "A1", "A"),
    (2.0, "A2", "A"),
)
NEIGHBORS_WITH_EXACT_MATCH: tuple[Neighbor, ...] = (
    (0.0, "B0", "B"),
    *NONZERO_NEIGHBORS,
)


def calculate_votes(
    neighbors: tuple[Neighbor, ...],
    k: int,
    weighting: str,
) -> tuple[str | None, dict[str, float]]:
    """Повертає переможця та суму голосів, явно обробляючи відстань 0."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(neighbors):
        raise ValueError("k має бути цілим числом від 1 до кількості сусідів.")
    if weighting not in ("uniform", "distance"):
        raise ValueError("weighting має бути uniform або distance.")

    selected = neighbors[:k]
    for distance, _, _ in selected:
        if not isfinite(distance) or distance < 0:
            raise ValueError("Відстані мають бути скінченними та невід'ємними.")

    if weighting == "distance":
        zero_distance_neighbors = tuple(row for row in selected if row[0] == 0.0)
        if zero_distance_neighbors:
            # Точні збіги мають вагу; нечислові 1/0 ігноруємо явно.
            weighted_neighbors = tuple(
                (1.0, name, label) for _, name, label in zero_distance_neighbors
            )
        else:
            weighted_neighbors = tuple(
                (1.0 / distance, name, label)
                for distance, name, label in selected
            )
    else:
        weighted_neighbors = tuple((1.0, name, label) for _, name, label in selected)

    vote_totals: dict[str, float] = defaultdict(float)
    for weight, _, label in weighted_neighbors:
        vote_totals[label] += weight

    highest_score = max(vote_totals.values())
    winners = sorted(
        label for label, score in vote_totals.items() if score == highest_score
    )
    winner = winners[0] if len(winners) == 1 else None
    return winner, dict(sorted(vote_totals.items()))


def print_vote(
    neighbors: tuple[Neighbor, ...],
    k: int,
    weighting: str,
) -> None:
    """Друкує відібраних сусідів, їхні ваги та результат голосування."""
    print(f"\nГОЛОСУВАННЯ: k={k}, weighting={weighting}")
    winner, votes = calculate_votes(neighbors, k, weighting)
    for rank, (distance, name, label) in enumerate(neighbors[:k], start=1):
        print(
            f"  сусід {rank}: {name}, клас={label}, d={distance:g}"
        )
    if weighting == "distance" and any(distance == 0 for distance, _, _ in neighbors[:k]):
        print("  Є точний збіг d=0: голосують лише сусіди з нульовою відстанню.")
    print(f"  Суми голосів: {votes}")
    if winner is None:
        print("  Результат: нічия.")
    else:
        print(f"  Результат: клас {winner}.")


def main() -> None:
    """Показує різницю рівного та зваженого голосування."""
    print("=" * 76)
    print("ПРАКТИКА 2.9 — ВАГИ СУСІДІВ І НУЛЬОВА ВІДСТАНЬ")
    print("=" * 76)
    print("Q=(0,0); сусіди вже відсортовані за L2.")

    print("\nВИПАДОК 1 — усі відстані додатні")
    print("  B1: клас B, d=0.25; A1: клас A, d=1; A2: клас A, d=2.")
    print_vote(NONZERO_NEIGHBORS, k=3, weighting="uniform")
    print_vote(NONZERO_NEIGHBORS, k=3, weighting="distance")
    print("  Рівні голоси обирають A (2 проти 1); відстані віддають перевагу B.")

    print("\nВИПАДОК 2 — точка Q вже є в тренувальних даних")
    print_vote(NEIGHBORS_WITH_EXACT_MATCH, k=4, weighting="uniform")
    print_vote(NEIGHBORS_WITH_EXACT_MATCH, k=4, weighting="distance")
    print("  Зважене голосування не ділить на нуль і використовує точні збіги.")


if __name__ == "__main__":
    main()
