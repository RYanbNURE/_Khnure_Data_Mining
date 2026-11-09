"""Покроковий розрахунок відстані Мінковського для двох точок."""

from __future__ import annotations

from math import fsum

Point = tuple[float, ...]


def minkowski_distance_verbose(
    first: Point,
    second: Point,
    order: int,
    first_name: str = "A",
    second_name: str = "B",
) -> float:
    """Обчислює відстань і докладно друкує кожен доданок формули."""
    if order not in (1, 2, 3):
        raise ValueError("Порядок має бути 1, 2 або 3.")
    if len(first) != len(second) or not first:
        raise ValueError("Точки мають бути непорожніми та однакової розмірності.")

    print(f"\nВідстань L{order} між {first_name}={first} і {second_name}={second}")
    print("Формула: d_p(A, B) = (Σ |A_j - B_j|^p)^(1/p)")

    powered_differences: list[float] = []
    for coordinate, (left, right) in enumerate(zip(first, second), start=1):
        # Кожна координата утворює окремий доданок у сумі формули.
        difference = abs(left - right)
        powered = difference**order
        powered_differences.append(powered)
        print(
            f"  Координата {coordinate}: "
            f"|{left:g} - {right:g}| = {difference:g}; "
            f"{difference:g}^{order} = {powered:g}"
        )

    # Спочатку підсумовуємо степені, потім застосовуємо корінь порядку 1/p.
    powered_sum = fsum(powered_differences)
    distance = powered_sum ** (1.0 / order)
    terms = " + ".join(f"{value:g}" for value in powered_differences)
    print(f"  Сума степенів: {terms} = {powered_sum:g}")
    print(f"  Корінь порядку {order}: {powered_sum:g}^(1/{order}) = {distance:.6f}")
    return distance


def main() -> None:
    """Порівнює L1, L2 і L3 для фіксованих двовимірних точок."""
    point_a = (1.0, 2.0)
    point_b = (4.0, 6.0)
    results: dict[int, float] = {}

    print("=" * 72)
    print("ПРАКТИКА 2.1 — РУЧНА ВІДСТАНЬ МІНКОВСЬКОГО")
    print("=" * 72)
    print("Точки лежать на простій координатній площині:")
    print(f"  A = {point_a}")
    print(f"  B = {point_b}")
    print("Різниці координат: Δx₁ = 3, Δx₂ = 4.")

    for order in (1, 2, 3):
        results[order] = minkowski_distance_verbose(
            point_a,
            point_b,
            order,
        )

    print("\nПІДСУМОК")
    for order, distance in results.items():
        print(f"  L{order} = {distance:.6f}")
    print("Для L2 маємо знайому трійку 3-4-5, тому евклідова відстань дорівнює 5.")


if __name__ == "__main__":
    main()
