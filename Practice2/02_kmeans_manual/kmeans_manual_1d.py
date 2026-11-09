"""Ручна ітерація K-Means на чотирьох точках одновимірного набору."""

from __future__ import annotations

from math import fsum

DATA = (1.0, 2.0, 8.0, 9.0)
INITIAL_CENTERS = (2.0, 8.0)
MAX_ITERATIONS = 10


def assign_points(
    points: tuple[float, ...],
    centers: list[float],
    iteration: int,
) -> list[int]:
    """Призначає кожну точку найближчому центру й друкує всі відстані."""
    print(f"\nКРОК 1 — призначення точок, ітерація {iteration}")
    assignments: list[int] = []

    for point in points:
        distances = [abs(point - center) for center in centers]
        nearest_cluster = min(range(len(centers)), key=distances.__getitem__)
        formatted_distances = ", ".join(
            f"|{point:g}-{center:g}|={distance:g}"
            for center, distance in zip(centers, distances)
        )
        print(
            f"  Точка x={point:g}: {formatted_distances} "
            f"-> кластер {nearest_cluster + 1}"
        )
        assignments.append(nearest_cluster)

    return assignments


def update_centers(
    points: tuple[float, ...],
    assignments: list[int],
    cluster_count: int,
) -> list[float]:
    """Перераховує центр кожного кластера як середнє його точок."""
    print("\nКРОК 2 — перерахунок центрів")
    new_centers: list[float] = []

    for cluster_index in range(cluster_count):
        members = [
            point
            for point, assignment in zip(points, assignments)
            if assignment == cluster_index
        ]
        if not members:
            raise RuntimeError(
                f"Кластер {cluster_index + 1} порожній; для прикладу змініть ініціалізацію."
            )

        center = fsum(members) / len(members)
        values = " + ".join(f"{member:g}" for member in members)
        print(
            f"  Кластер {cluster_index + 1}: "
            f"μ = ({values}) / {len(members)} = {center:g}"
        )
        new_centers.append(center)

    return new_centers


def calculate_inertia(
    points: tuple[float, ...],
    assignments: list[int],
    centers: list[float],
) -> float:
    """Рахує суму квадратів відстаней точок до їхніх центрів."""
    print("\nКРОК 3 — цільова функція (inertia)")
    squared_errors: list[float] = []

    for point, cluster_index in zip(points, assignments):
        center = centers[cluster_index]
        squared_error = (point - center) ** 2
        squared_errors.append(squared_error)
        print(
            f"  Для x={point:g} у кластері {cluster_index + 1}: "
            f"({point:g}-{center:g})² = {squared_error:g}"
        )

    inertia = fsum(squared_errors)
    terms = " + ".join(f"{error:g}" for error in squared_errors)
    print(f"  J = {terms} = {inertia:g}")
    return inertia


def main() -> None:
    """Виконує K-Means вручну та зупиняється після збіжності."""
    centers = list(INITIAL_CENTERS)
    previous_assignments: list[int] | None = None

    print("=" * 72)
    print("ПРАКТИКА 2.2 — K-MEANS КРОК ЗА КРОКОМ")
    print("=" * 72)
    print(f"Дані: {DATA}")
    print(f"K = {len(centers)}")
    print(f"Початкові центри: {centers}")
    print("Алгоритм бачить тільки числа, а не назви груп.")

    # Кожна ітерація спочатку призначає точки, а потім рухає центри до середніх.
    for iteration in range(1, MAX_ITERATIONS + 1):
        assignments = assign_points(DATA, centers, iteration)
        new_centers = update_centers(DATA, assignments, len(centers))
        inertia = calculate_inertia(DATA, assignments, new_centers)

        assignments_are_stable = assignments == previous_assignments
        centers_are_stable = new_centers == centers
        print("\nПЕРЕВІРКА ЗУПИНКИ")
        print(f"  Призначення не змінилися: {assignments_are_stable}")
        print(f"  Центри не змінилися: {centers_are_stable}")
        print(f"  Inertia після оновлення: {inertia:g}")

        if assignments_are_stable and centers_are_stable:
            print("  Умова збіжності виконана. Завершуємо цикл.")
            centers = new_centers
            break

        previous_assignments = assignments
        centers = new_centers
    else:
        print("Досягнуто MAX_ITERATIONS; цикл зупинено за лімітом.")

    print("\nФІНАЛЬНИЙ РЕЗУЛЬТАТ")
    print(f"  Центри: {centers}")
    for point, cluster_index in zip(DATA, assignments):
        print(f"  x={point:g} -> кластер {cluster_index + 1}")
    print(f"  Фінальна inertia: {inertia:g}")


if __name__ == "__main__":
    main()
