"""Показує, як порядок p змінює найближчого сусіда у k-NN."""

from __future__ import annotations

from math import fsum

Point = tuple[float, ...]
TRAINING_POINTS: tuple[tuple[str, Point, str], ...] = (
    ("A1", (0.0, 3.0), "A"),
    ("B1", (2.0, 2.0), "B"),
    ("A2", (4.0, 0.0), "A"),
)
QUERY: Point = (0.0, 0.0)
MINKOWSKI_ORDERS = (1, 2, 3)


def minkowski_distance(query: Point, point: Point, order: int) -> float:
    """Обчислює відстань Мінковського без друку."""
    if order not in MINKOWSKI_ORDERS:
        raise ValueError("Порядок Мінковського має бути 1, 2 або 3.")
    if not query or len(query) != len(point):
        raise ValueError("Вектори мають бути непорожніми та однакової довжини.")
    return fsum(
        abs(query_value - point_value) ** order
        for query_value, point_value in zip(query, point)
    ) ** (1.0 / order)


def distance_with_log(
    query: Point,
    point: Point,
    point_name: str,
    order: int,
    verbose: bool = True,
) -> float:
    """Рахує одну відстань і друкує вклад кожної координати."""
    if order not in MINKOWSKI_ORDERS:
        raise ValueError("Порядок Мінковського має бути 1, 2 або 3.")
    if not query or len(query) != len(point):
        raise ValueError("Вектори мають бути непорожніми та однакової довжини.")

    powers: list[float] = []
    for axis, (query_value, point_value) in enumerate(zip(query, point), start=1):
        difference = abs(query_value - point_value)
        contribution = difference**order
        powers.append(contribution)
        if verbose:
            print(
                f"      вісь {axis}: |{query_value:g}-{point_value:g}|^{order} "
                f"= {contribution:g}"
            )

    powered_sum = fsum(powers)
    distance = powered_sum ** (1.0 / order)
    if verbose:
        print(
            f"      d_{order}(Q,{point_name}) = "
            f"({powered_sum:g})^(1/{order}) = {distance:.6f}"
        )
    return distance


def rank_neighbors(
    order: int,
    verbose: bool = True,
) -> list[tuple[float, str, Point, str]]:
    """Повертає сусідів, відсортованих за відстанню та назвою точки."""
    ranked: list[tuple[float, str, Point, str]] = []
    for name, point, class_name in TRAINING_POINTS:
        if verbose:
            print(f"\n  Обчислюємо відстань до {name}={point}:")
        distance = distance_with_log(
            QUERY,
            point,
            name,
            order,
            verbose=verbose,
        )
        ranked.append((distance, name, point, class_name))
    return sorted(ranked, key=lambda row: (row[0], row[1]))


def main() -> None:
    """Порівнює ранжування трьох точок за L1, L2 та L3."""
    print("=" * 72)
    print("ПРАКТИКА 2.4 — ЯК p ЗМІНЮЄ НАЙБЛИЖЧОГО СУСІДА")
    print("=" * 72)
    print(f"Точка запиту Q = {QUERY}")
    print("Навчальні точки: A1=(0,3), B1=(2,2), A2=(4,0).")
    print("У кожної точки є мітка класу, але відстань від неї не залежить.")

    # Для кожного p залишаємо точки незмінними й порівнюємо лише метрику.
    summary: list[tuple[int, dict[str, float], str]] = []
    for order in MINKOWSKI_ORDERS:
        print(f"\n{'=' * 20} МЕТРИКА p={order} {'=' * 20}")
        ranked = rank_neighbors(order)
        print("\n  Відсортований список сусідів:")
        for rank, (distance, name, point, class_name) in enumerate(ranked, start=1):
            print(
                f"    {rank}. {name}={point}, клас {class_name}, "
                f"відстань {distance:.6f}"
            )
        distances_by_name = {name: distance for distance, name, _, _ in ranked}
        summary.append(
            (order, distances_by_name, f"{ranked[0][3]} ({ranked[0][1]})")
        )
        print(
            f"  За k=1 прогноз: клас {ranked[0][3]} "
            f"(найближча точка {ranked[0][1]})."
        )

    print("\nПІДСУМКОВА ТАБЛИЦЯ")
    print("  p | d(A1)     | d(B1)     | d(A2)     | k=1: клас (сусід)")
    for order, distances, prediction in summary:
        print(
            f"  {order} | {distances['A1']:9.6f} | {distances['B1']:9.6f} | "
            f"{distances['A2']:9.6f} | {prediction}"
        )
    print("\nВИСНОВОК")
    print("  p=1 обирає A1; p=2 та p=3 обирають B1.")
    print("  Дані ті самі; змінилася тільки формула відстані.")


if __name__ == "__main__":
    main()
