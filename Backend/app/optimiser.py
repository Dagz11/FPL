from .projections import calculate_projection


def rank_players(players):
    ranked = []

    for player in players:
        projection = calculate_projection(player)

        ranked.append(
            {
                "id": player["id"],
                "name": player["web_name"],
                "team": player["team"],
                "position": player["element_type"],
                "price": player["now_cost"] / 10,
                "projection": projection,
            }
        )

    return sorted(
        ranked,
        key=lambda player: player["projection"],
        reverse=True,
    )
