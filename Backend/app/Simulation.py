import random


def simulate_points(
    projected_points,
    start_probability=1.0,
    runs=10000,
):
    results = []

    for _ in range(runs):
        starts = random.random() <= start_probability

        if starts:
            points = projected_points
        else:
            points = 0

        results.append(points)

    return {
        "mean": sum(results) / len(results),
        "runs": runs,
    }
