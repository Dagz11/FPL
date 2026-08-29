from itertools import combinations

import numpy as np

from scipy.optimize import (
    Bounds,
    LinearConstraint,
    milp,
)

from .config import (
    GAMEWEEK_DECAY,
    MAX_FREE_TRANSFERS,
    MAX_FORMATION,
    MAX_PLAYERS_PER_TEAM,
    MIN_FORMATION,
    PATH_BEAM_WIDTH,
    PATH_MAX_TRANSFERS_PER_GW,
    SQUAD_POSITION_COUNTS,
    TRANSFER_CANDIDATES_PER_POSITION,
    TRANSFER_HIT_COST,
    TRANSFER_RESULTS_RETURNED,
)


def projection_map(
    projections,
):

    return {
        player["id"]: player
        for player
        in projections
    }


def is_valid_squad(
    squad,
):

    if len(squad) != 15:
        return False

    positions = {}

    clubs = {}

    for player in squad:

        position = player[
            "position"
        ]

        team = player[
            "team_id"
        ]

        positions[position] = (
            positions.get(
                position,
                0,
            )
            + 1
        )

        clubs[team] = (
            clubs.get(
                team,
                0,
            )
            + 1
        )

    for position, required in (
        SQUAD_POSITION_COUNTS.items()
    ):

        if (
            positions.get(
                position,
                0,
            )
            != required
        ):
            return False

    if any(
        count
        > MAX_PLAYERS_PER_TEAM
        for count
        in clubs.values()
    ):
        return False

    return True


def optimal_xi(
    squad,
    gw=None,
):

    def score(player):

        if gw is None:
            return player[
                "gw1_xpts"
            ]

        return player[
            "gw_xpts"
        ].get(
            str(gw),
            0,
        )

    goalkeepers = sorted(
        [
            player
            for player in squad
            if player[
                "position"
            ]
            == "GK"
        ],
        key=score,
        reverse=True,
    )

    defenders = sorted(
        [
            player
            for player in squad
            if player[
                "position"
            ]
            == "DEF"
        ],
        key=score,
        reverse=True,
    )

    midfielders = sorted(
        [
            player
            for player in squad
            if player[
                "position"
            ]
            == "MID"
        ],
        key=score,
        reverse=True,
    )

    forwards = sorted(
        [
            player
            for player in squad
            if player[
                "position"
            ]
            == "FWD"
        ],
        key=score,
        reverse=True,
    )

    best = None

    for defenders_count in range(
        MIN_FORMATION["DEF"],
        MAX_FORMATION["DEF"] + 1,
    ):

        for midfield_count in range(
            MIN_FORMATION["MID"],
            MAX_FORMATION["MID"] + 1,
        ):

            forward_count = (
                10
                - defenders_count
                - midfield_count
            )

            if not (
                MIN_FORMATION["FWD"]
                <= forward_count
                <= MAX_FORMATION["FWD"]
            ):
                continue

            if (
                defenders_count
                > len(defenders)
                or midfield_count
                > len(midfielders)
                or forward_count
                > len(forwards)
            ):
                continue

            xi = (
                goalkeepers[:1]
                + defenders[
                    :defenders_count
                ]
                + midfielders[
                    :midfield_count
                ]
                + forwards[
                    :forward_count
                ]
            )

            total = sum(
                score(player)
                for player
                in xi
            )

            if (
                best is None
                or total
                > best[
                    "total"
                ]
            ):

                best = {
                    "xi": xi,
                    "total": total,
                    "formation": (
                        f"{defenders_count}-"
                        f"{midfield_count}-"
                        f"{forward_count}"
                    ),
                }

    selected_ids = {
        player["id"]
        for player
        in best["xi"]
    }

    bench = [
        player
        for player in squad
        if player["id"]
        not in selected_ids
    ]

    bench.sort(
        key=score,
        reverse=True,
    )

    captain = max(
        best["xi"],
        key=score,
    )

    vice_candidates = [
        player
        for player
        in best["xi"]
        if player["id"]
        != captain["id"]
    ]

    vice = max(
        vice_candidates,
        key=score,
    )

    return {
        "formation":
            best["formation"],

        "expected_points":
            round(
                best["total"],
                3,
            ),

        "xi":
            best["xi"],

        "bench":
            bench,

        "captain":
            captain,

        "vice_captain":
            vice,
    }


def optimise_squad(
    projections,
    budget=100.0,
):

    players = projections

    count = len(players)

    objective = -np.array(
        [
            player[
                "strategy_score"
            ]
            for player
            in players
        ]
    )

    constraints = []

    # Exactly 15 players
    row = np.ones(
        count
    )

    constraints.append(
        LinearConstraint(
            row,
            15,
            15,
        )
    )

    # Exact position requirements
    for position, required in (
        SQUAD_POSITION_COUNTS.items()
    ):

        row = np.array(
            [
                1
                if player[
                    "position"
                ]
                == position
                else 0
                for player
                in players
            ]
        )

        constraints.append(
            LinearConstraint(
                row,
                required,
                required,
            )
        )

    # Budget
    row = np.array(
        [
            player["price"]
            for player
            in players
        ]
    )

    constraints.append(
        LinearConstraint(
            row,
            0,
            budget,
        )
    )

    # Maximum 3 per club
    club_ids = sorted(
        {
            player["team_id"]
            for player
            in players
        }
    )

    for club_id in club_ids:

        row = np.array(
            [
                1
                if player[
                    "team_id"
                ]
                == club_id
                else 0
                for player
                in players
            ]
        )

        constraints.append(
            LinearConstraint(
                row,
                0,
                MAX_PLAYERS_PER_TEAM,
            )
        )

    result = milp(
        c=objective,
        integrality=np.ones(
            count
        ),
        bounds=Bounds(
            np.zeros(count),
            np.ones(count),
        ),
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(
            "Squad optimisation failed: "
            + str(
                result.message
            )
        )

    selected = [
        player
        for index, player
        in enumerate(players)
        if result.x[index] > 0.5
    ]

    total_cost = sum(
        player["price"]
        for player
        in selected
    )

    return {
        "squad":
            selected,

        "cost":
            round(
                total_cost,
                1,
            ),

        "money_remaining":
            round(
                budget
                - total_cost,
                1,
            ),

        "strategy_score":
            round(
                sum(
                    player[
                        "strategy_score"
                    ]
                    for player
                    in selected
                ),
                3,
            ),

        "optimal_xi":
            optimal_xi(
                selected
            ),
    }


def club_counts(
    squad,
):

    counts = {}

    for player in squad:

        counts[
            player["team_id"]
        ] = (
            counts.get(
                player[
                    "team_id"
                ],
                0,
            )
            + 1
        )

    return counts


def transfer_candidates(
    projections,
    owned_ids,
):

    result = {}

    for position in (
        "GK",
        "DEF",
        "MID",
        "FWD",
    ):

        candidates = [
            player
            for player
            in projections
            if (
                player[
                    "position"
                ]
                == position
                and player[
                    "id"
                ]
                not in owned_ids
            )
        ]

        candidates.sort(
            key=lambda player:
            player[
                "strategy_score"
            ],
            reverse=True,
        )

        result[position] = (
            candidates[
                :
                TRANSFER_CANDIDATES_PER_POSITION
            ]
        )

    return result


def best_transfers(
    squad,
    projections,
    bank=0.0,
    free_transfers=1,
    max_moves=2,
):

    owned_ids = {
        player["id"]
        for player
        in squad
    }

    candidates = (
        transfer_candidates(
            projections,
            owned_ids,
        )
    )

    results = []

    # Single transfers
    for outgoing in squad:

        for incoming in candidates[
            outgoing[
                "position"
            ]
        ]:

            available_money = (
                bank
                + outgoing[
                    "price"
                ]
            )

            if (
                incoming[
                    "price"
                ]
                > available_money
            ):
                continue

            new_squad = [
                player
                for player in squad
                if player["id"]
                != outgoing["id"]
            ] + [
                incoming
            ]

            if not is_valid_squad(
                new_squad
            ):
                continue

            gain = (
                incoming[
                    "strategy_score"
                ]
                - outgoing[
                    "strategy_score"
                ]
            )

            hits = max(
                0,
                1
                - free_transfers,
            )

            hit_cost = (
                hits
                * TRANSFER_HIT_COST
            )

            results.append(
                {
                    "moves": [
                        {
                            "out":
                                outgoing[
                                    "name"
                                ],

                            "out_id":
                                outgoing[
                                    "id"
                                ],

                            "in":
                                incoming[
                                    "name"
                                ],

                            "in_id":
                                incoming[
                                    "id"
                                ],
                        }
                    ],

                    "transfers": 1,

                    "gross_gain":
                        round(
                            gain,
                            3,
                        ),

                    "hit_cost":
                        hit_cost,

                    "net_gain":
                        round(
                            gain
                            - hit_cost,
                            3,
                        ),

                    "bank_after":
                        round(
                            available_money
                            - incoming[
                                "price"
                            ],
                            1,
                        ),
                }
            )

    if max_moves >= 2:

        outgoing_pairs = (
            combinations(
                squad,
                2,
            )
        )

        for (
            outgoing_a,
            outgoing_b,
        ) in outgoing_pairs:

            candidate_a = candidates[
                outgoing_a[
                    "position"
                ]
            ]

            candidate_b = candidates[
                outgoing_b[
                    "position"
                ]
            ]

            for incoming_a in (
                candidate_a
            ):

                for incoming_b in (
                    candidate_b
                ):

                    if (
                        incoming_a[
                            "id"
                        ]
                        == incoming_b[
                            "id"
                        ]
                    ):
                        continue

                    money = (
                        bank
                        + outgoing_a[
                            "price"
                        ]
                        + outgoing_b[
                            "price"
                        ]
                    )

                    cost = (
                        incoming_a[
                            "price"
                        ]
                        + incoming_b[
                            "price"
                        ]
                    )

                    if cost > money:
                        continue

                    removed = {
                        outgoing_a[
                            "id"
                        ],
                        outgoing_b[
                            "id"
                        ],
                    }

                    new_squad = [
                        player
                        for player
                        in squad
                        if player[
                            "id"
                        ]
                        not in removed
                    ] + [
                        incoming_a,
                        incoming_b,
                    ]

                    if not is_valid_squad(
                        new_squad
                    ):
                        continue

                    gain = (
                        incoming_a[
                            "strategy_score"
                        ]
                        + incoming_b[
                            "strategy_score"
                        ]
                        - outgoing_a[
                            "strategy_score"
                        ]
                        - outgoing_b[
                            "strategy_score"
                        ]
                    )

                    hits = max(
                        0,
                        2
                        - free_transfers,
                    )

                    hit_cost = (
                        hits
                        * TRANSFER_HIT_COST
                    )

                    results.append(
                        {
                            "moves": [
                                {
                                    "out":
                                        outgoing_a[
                                            "name"
                                        ],
                                    "out_id":
                                        outgoing_a[
                                            "id"
                                        ],
                                    "in":
                                        incoming_a[
                                            "name"
                                        ],
                                    "in_id":
                                        incoming_a[
                                            "id"
                                        ],
                                },
                                {
                                    "out":
                                        outgoing_b[
                                            "name"
                                        ],
                                    "out_id":
                                        outgoing_b[
                                            "id"
                                        ],
                                    "in":
                                        incoming_b[
                                            "name"
                                        ],
                                    "in_id":
                                        incoming_b[
                                            "id"
                                        ],
                                },
                            ],

                            "transfers":
                                2,

                            "gross_gain":
                                round(
                                    gain,
                                    3,
                                ),

                            "hit_cost":
                                hit_cost,

                            "net_gain":
                                round(
                                    gain
                                    - hit_cost,
                                    3,
                                ),

                            "bank_after":
                                round(
                                    money
                                    - cost,
                                    1,
                                ),
                        }
                    )

    results.sort(
        key=lambda item:
        item["net_gain"],
        reverse=True,
    )

    return results[
        :
        TRANSFER_RESULTS_RETURNED
    ]


def squad_gameweek_score(
    squad,
    gameweek,
):

    xi_result = optimal_xi(
        squad,
        gw=gameweek,
    )

    xi_points = (
        xi_result[
            "expected_points"
        ]
    )

    captain = (
        xi_result[
            "captain"
        ]
    )

    captain_points = (
        captain[
            "gw_xpts"
        ].get(
            str(gameweek),
            0,
        )
    )

    # Captain's score is counted twice.
    return (
        xi_points
        + captain_points
    )


def apply_transfer_plan(
    squad,
    move,
    player_lookup,
):

    removed_ids = {
        item["out_id"]
        for item
        in move["moves"]
    }

    new_squad = [
        player
        for player
        in squad
        if player["id"]
        not in removed_ids
    ]

    for item in move[
        "moves"
    ]:

        new_squad.append(
            player_lookup[
                item["in_id"]
            ]
        )

    return new_squad


def plan_transfer_path(
    squad,
    projections,
    start_gameweek,
    horizon=6,
    bank=0.0,
    free_transfers=1,
):

    player_lookup = (
        projection_map(
            projections
        )
    )

    states = [
        {
            "squad":
                squad,

            "bank":
                bank,

            "free_transfers":
                free_transfers,

            "score":
                0.0,

            "path":
                [],
        }
    ]

    for step in range(
        horizon
    ):

        gameweek = (
            start_gameweek
            + step
        )

        new_states = []

        for state in states:

            # Option 1: roll
            gw_score = (
                squad_gameweek_score(
                    state["squad"],
                    gameweek,
                )
            )

            discounted = (
                gw_score
                * (
                    GAMEWEEK_DECAY
                    ** step
                )
            )

            new_states.append(
                {
                    "squad":
                        state[
                            "squad"
                        ],

                    "bank":
                        state[
                            "bank"
                        ],

                    "free_transfers":
                        min(
                            MAX_FREE_TRANSFERS,
                            state[
                                "free_transfers"
                            ]
                            + 1,
                        ),

                    "score":
                        state[
                            "score"
                        ]
                        + discounted,

                    "path":
                        state[
                            "path"
                        ]
                        + [
                            {
                                "gameweek":
                                    gameweek,

                                "action":
                                    "ROLL",

                                "expected_points":
                                    round(
                                        gw_score,
                                        3,
                                    ),
                            }
                        ],
                }
            )

            transfer_options = (
                best_transfers(
                    state[
                        "squad"
                    ],
                    projections,
                    bank=state[
                        "bank"
                    ],
                    free_transfers=state[
                        "free_transfers"
                    ],
                    max_moves=(
                        PATH_MAX_TRANSFERS_PER_GW
                    ),
                )
            )

            for option in (
                transfer_options[:10]
            ):

                updated_squad = (
                    apply_transfer_plan(
                        state[
                            "squad"
                        ],
                        option,
                        player_lookup,
                    )
                )

                transfer_count = (
                    option[
                        "transfers"
                    ]
                )

                hits = max(
                    0,
                    transfer_count
                    - state[
                        "free_transfers"
                    ],
                )

                hit_cost = (
                    hits
                    * TRANSFER_HIT_COST
                )

                gw_score = (
                    squad_gameweek_score(
                        updated_squad,
                        gameweek,
                    )
                    - hit_cost
                )

                next_free_transfers = (
                    min(
                        MAX_FREE_TRANSFERS,
                        max(
                            0,
                            state[
                                "free_transfers"
                            ]
                            - transfer_count,
                        )
                        + 1,
                    )
                )

                discounted = (
                    gw_score
                    * (
                        GAMEWEEK_DECAY
                        ** step
                    )
                )

                new_states.append(
                    {
                        "squad":
                            updated_squad,

                        "bank":
                            option[
                                "bank_after"
                            ],

                        "free_transfers":
                            next_free_transfers,

                        "score":
                            state[
                                "score"
                            ]
                            + discounted,

                        "path":
                            state[
                                "path"
                            ]
                            + [
                                {
                                    "gameweek":
                                        gameweek,

                                    "action":
                                        "TRANSFER",

                                    "moves":
                                        option[
                                            "moves"
                                        ],

                                    "hit_cost":
                                        hit_cost,

                                    "expected_points":
                                        round(
                                            gw_score,
                                            3,
                                        ),
                                }
                            ],
                    }
                )

        new_states.sort(
            key=lambda state:
            state["score"],
            reverse=True,
        )

        states = (
            new_states[
                :
                PATH_BEAM_WIDTH
            ]
        )

    best = states[0]

    return {
        "weighted_expected_points":
            round(
                best["score"],
                3,
            ),

        "bank_final":
            round(
                best["bank"],
                1,
            ),

        "free_transfers_final":
            best[
                "free_transfers"
            ],

        "path":
            best[
                "path"
            ],

        "final_squad":
            best[
                "squad"
            ],
    }