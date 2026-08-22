from .config import WEIGHTS


def calculate_projection(player):
    points_per_game = float(player.get("points_per_game", 0) or 0)

    chance_of_playing = player.get("chance_of_playing_next_round")

    if chance_of_playing is None:
        start_probability = 1.0
    else:
        start_probability = chance_of_playing / 100

    form = float(player.get("form", 0) or 0)

    fixture_factor = 1.0

    projection = (
        points_per_game
        * start_probability
        * WEIGHTS["minutes"]
    )

    projection += (
        form
        * WEIGHTS["form"]
    )

    projection *= fixture_factor

    return round(projection, 2)
