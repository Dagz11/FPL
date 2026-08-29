import json
import math
from pathlib import Path

import numpy as np

from .config import (
    ASSIST_POINTS,
    BASE_GOALS_PER_TEAM,
    BONUS90_PRIORS,
    CLEAN_SHEET_POINTS,
    DC_RETURN_PRIOR,
    DEFAULT_CONGESTION_RISK,
    DEFAULT_INJURY_RISK,
    DEFAULT_ROTATION_RISK,
    DEFAULT_START_MINUTES,
    DEFAULT_SUB_APPEARANCE_PROBABILITY,
    DEFAULT_SUB_MINUTES,
    GAMEWEEK_DECAY,
    GK_PENALTY_SAVE90_PRIOR,
    GK_SAVE90_PRIOR,
    GOAL_POINTS,
    HOME_ATTACK_MULTIPLIER,
    AWAY_ATTACK_MULTIPLIER,
    HORIZON_WEIGHTS,
    MAX_ATTACK_FACTOR,
    MAX_DEFENCE_FACTOR,
    MIN_ATTACK_FACTOR,
    MIN_DEFENCE_FACTOR,
    MODEL_WEIGHTS,
    NORMALISE_HORIZONS,
    PENALTY_TAKER_XG_MULTIPLIER,
    RATE_SHRINKAGE_MINUTES,
    SET_PIECE_XA_MULTIPLIER,
    START_PRIORS,
    START_PRIOR_STRENGTH,
    XG90_PRIORS,
    XA90_PRIORS,
    YELLOW90_PRIOR,
    RED90_PRIOR,
)

from .fpl_data import (
    get_bootstrap,
    get_fixtures,
    get_next_gameweek,
)


ROOT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
)

OVERRIDES_FILE = (
    ROOT_DIRECTORY
    / "data"
    / "manual_overrides.json"
)


def safe_float(
    value,
    default=0.0,
):

    try:
        if value in (
            None,
            "",
        ):
            return default

        return float(value)

    except (
        ValueError,
        TypeError,
    ):
        return default


def clip(
    value,
    low,
    high,
):

    return max(
        low,
        min(
            value,
            high,
        ),
    )


def sigmoid(value):

    return (
        1.0
        /
        (
            1.0
            + math.exp(-value)
        )
    )


def load_overrides():

    if not OVERRIDES_FILE.exists():
        return {}

    try:
        with open(
            OVERRIDES_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return data.get(
            "players",
            {},
        )

    except Exception:
        return {}


def build_team_maps(
    bootstrap,
):

    teams = bootstrap["teams"]

    team_by_id = {
        team["id"]: team
        for team in teams
    }

    return team_by_id


def build_position_map(
    bootstrap,
):

    return {
        item["id"]:
        item["singular_name_short"]
        for item
        in bootstrap["element_types"]
    }


def matches_played_by_team(
    fixtures,
):

    counts = {}

    for fixture in fixtures:

        if not fixture.get(
            "finished",
            False,
        ):
            continue

        home = fixture["team_h"]
        away = fixture["team_a"]

        counts[home] = (
            counts.get(home, 0)
            + 1
        )

        counts[away] = (
            counts.get(away, 0)
            + 1
        )

    return counts


def get_player_override(
    player,
    overrides,
):

    id_key = str(
        player["id"]
    )

    if id_key in overrides:
        return overrides[id_key]

    web_name = player.get(
        "web_name",
        "",
    )

    if web_name in overrides:
        return overrides[web_name]

    full_name = (
        f"{player.get('first_name', '')} "
        f"{player.get('second_name', '')}"
    ).strip()

    if full_name in overrides:
        return overrides[full_name]

    return {}


def availability_factor(
    player,
):

    chance = player.get(
        "chance_of_playing_next_round"
    )

    if chance is None:
        chance_factor = 1.0

    else:
        chance_factor = (
            safe_float(chance, 100)
            / 100
        )

    status = player.get(
        "status",
        "a",
    )

    status_factor = {
        "a": 1.00,
        "d": 0.75,
        "i": 0.05,
        "s": 0.00,
        "u": 0.00,
        "n": 0.00,
    }.get(
        status,
        0.90,
    )

    return min(
        chance_factor,
        status_factor,
    )


def calculate_start_profile(
    player,
    position,
    team_matches,
    override,
):

    starts = safe_float(
        player.get("starts")
    )

    minutes = safe_float(
        player.get("minutes")
    )

    prior = START_PRIORS[
        position
    ]

    start_probability = (
        starts
        + (
            START_PRIOR_STRENGTH
            * prior
        )
    ) / (
        team_matches
        + START_PRIOR_STRENGTH
    )

    start_probability *= (
        availability_factor(player)
    )

    if starts > 0:

        observed_start_minutes = (
            minutes
            / starts
        )

        observed_start_minutes = clip(
            observed_start_minutes,
            45,
            90,
        )

        start_minutes = (
            0.70
            * observed_start_minutes
            + 0.30
            * DEFAULT_START_MINUTES[
                position
            ]
        )

    else:
        start_minutes = (
            DEFAULT_START_MINUTES[
                position
            ]
        )

    appearances = safe_float(
        player.get("appearances")
    )

    if (
        appearances > starts
        and team_matches > 0
    ):

        sub_rate = (
            appearances
            - starts
        ) / team_matches

        sub_probability = clip(
            sub_rate,
            0,
            0.75,
        )

    else:
        sub_probability = (
            DEFAULT_SUB_APPEARANCE_PROBABILITY[
                position
            ]
        )

    if "start_probability" in override:

        start_probability = clip(
            safe_float(
                override[
                    "start_probability"
                ]
            ),
            0,
            1,
        )

    if "expected_start_minutes" in override:

        start_minutes = clip(
            safe_float(
                override[
                    "expected_start_minutes"
                ]
            ),
            1,
            95,
        )

    if (
        "sub_appearance_probability"
        in override
    ):

        sub_probability = clip(
            safe_float(
                override[
                    "sub_appearance_probability"
                ]
            ),
            0,
            1,
        )

    expected_minutes = (
        start_probability
        * start_minutes
        + (
            1
            - start_probability
        )
        * sub_probability
        * DEFAULT_SUB_MINUTES[
            position
        ]
    )

    p_play = (
        start_probability
        + (
            1
            - start_probability
        )
        * sub_probability
    )

    probability_60 = (
        start_probability
        * sigmoid(
            (
                start_minutes
                - 60
            )
            / 6
        )
    )

    return {
        "start_probability":
            start_probability,

        "sub_probability":
            sub_probability,

        "expected_start_minutes":
            start_minutes,

        "expected_minutes":
            expected_minutes,

        "play_probability":
            p_play,

        "p60":
            probability_60,
    }


def shrunk_rate(
    observed,
    minutes,
    prior,
):

    if minutes <= 0:
        return prior

    observed_per_90 = (
        observed
        * 90
        / minutes
    )

    weight = (
        minutes
        /
        (
            minutes
            + RATE_SHRINKAGE_MINUTES
        )
    )

    return (
        weight
        * observed_per_90
        + (
            1
            - weight
        )
        * prior
    )


def team_strength_averages(
    teams,
):

    fields = [
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]

    averages = {}

    for field in fields:

        values = [
            safe_float(
                team.get(field),
                1000,
            )
            for team
            in teams.values()
        ]

        averages[field] = (
            float(
                np.mean(values)
            )
            if values
            else 1000
        )

    return averages


def fixture_multipliers(
    team,
    opponent,
    is_home,
    averages,
):

    if is_home:

        team_attack = safe_float(
            team.get(
                "strength_attack_home"
            ),
            averages[
                "strength_attack_home"
            ],
        )

        opponent_defence = (
            safe_float(
                opponent.get(
                    "strength_defence_away"
                ),
                averages[
                    "strength_defence_away"
                ],
            )
        )

        attack_average = (
            averages[
                "strength_attack_home"
            ]
        )

        defence_average = (
            averages[
                "strength_defence_away"
            ]
        )

        venue_multiplier = (
            HOME_ATTACK_MULTIPLIER
        )

    else:

        team_attack = safe_float(
            team.get(
                "strength_attack_away"
            ),
            averages[
                "strength_attack_away"
            ],
        )

        opponent_defence = (
            safe_float(
                opponent.get(
                    "strength_defence_home"
                ),
                averages[
                    "strength_defence_home"
                ],
            )
        )

        attack_average = (
            averages[
                "strength_attack_away"
            ]
        )

        defence_average = (
            averages[
                "strength_defence_home"
            ]
        )

        venue_multiplier = (
            AWAY_ATTACK_MULTIPLIER
        )

    attack_factor = (
        (
            team_attack
            / max(
                attack_average,
                1,
            )
        )
        *
        (
            defence_average
            / max(
                opponent_defence,
                1,
            )
        )
        *
        venue_multiplier
    )

    attack_factor = clip(
        attack_factor,
        MIN_ATTACK_FACTOR,
        MAX_ATTACK_FACTOR,
    )

    return attack_factor


def expected_opponent_goals(
    team,
    opponent,
    is_home,
    averages,
):

    if is_home:

        opponent_attack = (
            safe_float(
                opponent.get(
                    "strength_attack_away"
                ),
                averages[
                    "strength_attack_away"
                ],
            )
        )

        team_defence = (
            safe_float(
                team.get(
                    "strength_defence_home"
                ),
                averages[
                    "strength_defence_home"
                ],
            )
        )

        attack_average = (
            averages[
                "strength_attack_away"
            ]
        )

        defence_average = (
            averages[
                "strength_defence_home"
            ]
        )

        venue = (
            AWAY_ATTACK_MULTIPLIER
        )

    else:

        opponent_attack = (
            safe_float(
                opponent.get(
                    "strength_attack_home"
                ),
                averages[
                    "strength_attack_home"
                ],
            )
        )

        team_defence = (
            safe_float(
                team.get(
                    "strength_defence_away"
                ),
                averages[
                    "strength_defence_away"
                ],
            )
        )

        attack_average = (
            averages[
                "strength_attack_home"
            ]
        )

        defence_average = (
            averages[
                "strength_defence_away"
            ]
        )

        venue = (
            HOME_ATTACK_MULTIPLIER
        )

    goal_lambda = (
        BASE_GOALS_PER_TEAM
        *
        (
            opponent_attack
            / max(
                attack_average,
                1,
            )
        )
        *
        (
            defence_average
            / max(
                team_defence,
                1,
            )
        )
        *
        venue
    )

    return clip(
        goal_lambda,
        0.35,
        3.40,
    )


def expected_goals_conceded_penalty(
    goal_lambda,
):

    probabilities = []

    for goals in range(0, 9):

        probability = (
            math.exp(
                -goal_lambda
            )
            * (
                goal_lambda
                ** goals
            )
            / math.factorial(
                goals
            )
        )

        probabilities.append(
            probability
        )

    expectation = 0.0

    for goals, probability in enumerate(
        probabilities
    ):

        expectation += (
            math.floor(
                goals
                / 2
            )
            * probability
        )

    return expectation


def defensive_contribution_probability(
    player,
    position,
    expected_minutes,
):

    if position == "GK":
        return 0.0

    observed_dc_points = (
        safe_float(
            player.get(
                "defensive_contribution"
            )
        )
    )

    minutes = safe_float(
        player.get("minutes")
    )

    prior = DC_RETURN_PRIOR[
        position
    ]

    if (
        observed_dc_points > 0
        and minutes > 0
    ):

        returns = (
            observed_dc_points
            / 2
        )

        matches_90 = (
            minutes
            / 90
        )

        rate = (
            returns
            / max(
                matches_90,
                1,
            )
        )

        probability = (
            0.60
            * clip(
                rate,
                0,
                1,
            )
            + 0.40
            * prior
        )

    else:
        probability = prior

    minutes_factor = clip(
        expected_minutes
        / 80,
        0,
        1,
    )

    return (
        probability
        * minutes_factor
    )


def project_fixture(
    player,
    position,
    team,
    opponent,
    fixture,
    is_home,
    start_profile,
    averages,
    override,
):

    minutes = safe_float(
        player.get("minutes")
    )

    xg90 = shrunk_rate(
        safe_float(
            player.get(
                "expected_goals"
            )
        ),
        minutes,
        XG90_PRIORS[
            position
        ],
    )

    xa90 = shrunk_rate(
        safe_float(
            player.get(
                "expected_assists"
            )
        ),
        minutes,
        XA90_PRIORS[
            position
        ],
    )

    bonus90 = shrunk_rate(
        safe_float(
            player.get("bonus")
        ),
        minutes,
        BONUS90_PRIORS[
            position
        ],
    )

    penalty_taker = bool(
        override.get(
            "penalty_taker",
            False,
        )
    )

    set_piece_taker = bool(
        override.get(
            "set_piece_taker",
            False,
        )
    )

    if penalty_taker:
        xg90 *= (
            PENALTY_TAKER_XG_MULTIPLIER
        )

    if set_piece_taker:
        xa90 *= (
            SET_PIECE_XA_MULTIPLIER
        )

    attack_factor = fixture_multipliers(
        team,
        opponent,
        is_home,
        averages,
    )

    opponent_goal_lambda = (
        expected_opponent_goals(
            team,
            opponent,
            is_home,
            averages,
        )
    )

    clean_sheet_probability = (
        math.exp(
            -opponent_goal_lambda
        )
    )

    expected_minutes = (
        start_profile[
            "expected_minutes"
        ]
    )

    goal_lambda = (
        xg90
        * expected_minutes
        / 90
        * attack_factor
    )

    assist_lambda = (
        xa90
        * expected_minutes
        / 90
        * attack_factor
    )

    appearance_points = (
        start_profile[
            "play_probability"
        ]
        +
        start_profile[
            "p60"
        ]
    )

    goal_points = (
        goal_lambda
        * GOAL_POINTS[
            position
        ]
    )

    assist_points = (
        assist_lambda
        * ASSIST_POINTS
    )

    clean_sheet_points = (
        start_profile[
            "p60"
        ]
        * clean_sheet_probability
        * CLEAN_SHEET_POINTS[
            position
        ]
    )

    bonus_points = (
        bonus90
        * expected_minutes
        / 90
        * MODEL_WEIGHTS[
            "bonus"
        ]
    )

    dc_probability = (
        defensive_contribution_probability(
            player,
            position,
            expected_minutes,
        )
    )

    dc_points = (
        dc_probability
        * 2
        * MODEL_WEIGHTS[
            "defensive_contributions"
        ]
    )

    yellow90 = shrunk_rate(
        safe_float(
            player.get(
                "yellow_cards"
            )
        ),
        minutes,
        YELLOW90_PRIOR[
            position
        ],
    )

    red90 = shrunk_rate(
        safe_float(
            player.get(
                "red_cards"
            )
        ),
        minutes,
        RED90_PRIOR[
            position
        ],
    )

    discipline_points = (
        -yellow90
        * expected_minutes
        / 90
        +
        -3
        * red90
        * expected_minutes
        / 90
    )

    concession_points = 0.0
    save_points = 0.0
    penalty_save_points = 0.0

    if position in (
        "GK",
        "DEF",
    ):

        expected_concession_penalty = (
            expected_goals_conceded_penalty(
                opponent_goal_lambda
            )
        )

        concession_points = (
            -expected_concession_penalty
            * start_profile[
                "p60"
            ]
        )

    if position == "GK":

        saves = safe_float(
            player.get("saves")
        )

        saves90 = shrunk_rate(
            saves,
            minutes,
            GK_SAVE90_PRIOR,
        )

        expected_saves = (
            saves90
            * expected_minutes
            / 90
        )

        save_points = (
            expected_saves
            / 3
        )

        penalty_saves = (
            safe_float(
                player.get(
                    "penalties_saved"
                )
            )
        )

        penalty_save90 = (
            shrunk_rate(
                penalty_saves,
                minutes,
                GK_PENALTY_SAVE90_PRIOR,
            )
        )

        penalty_save_points = (
            penalty_save90
            * expected_minutes
            / 90
            * 5
        )

    raw_xpts = (
        appearance_points
        + goal_points
        + assist_points
        + clean_sheet_points
        + bonus_points
        + dc_points
        + discipline_points
        + concession_points
        + save_points
        + penalty_save_points
    )

    form = safe_float(
        player.get("form")
    )

    form_adjustment = (
        MODEL_WEIGHTS[
            "form"
        ]
        * form
        * 0.10
    )

    rotation_risk = clip(
        safe_float(
            override.get(
                "rotation_risk",
                DEFAULT_ROTATION_RISK,
            )
        ),
        0,
        1,
    )

    congestion_risk = clip(
        safe_float(
            override.get(
                "congestion_risk",
                DEFAULT_CONGESTION_RISK,
            )
        ),
        0,
        1,
    )

    injury_risk = clip(
        safe_float(
            override.get(
                "injury_risk",
                DEFAULT_INJURY_RISK,
            )
        ),
        0,
        1,
    )

    risk_penalty = (
        rotation_risk
        * MODEL_WEIGHTS[
            "rotation_penalty"
        ]
        +
        congestion_risk
        * MODEL_WEIGHTS[
            "congestion_penalty"
        ]
        +
        injury_risk
        * MODEL_WEIGHTS[
            "injury_penalty"
        ]
    )

    xpts = (
        raw_xpts
        + form_adjustment
    ) * (
        1
        - risk_penalty
    )

    xpts = max(
        0,
        xpts,
    )

    opponent_name = opponent[
        "short_name"
    ]

    fixture_label = (
        f"{opponent_name} "
        f"({'H' if is_home else 'A'})"
    )

    return {
        "gameweek":
            fixture.get("event"),

        "fixture":
            fixture_label,

        "home":
            is_home,

        "xpts":
            round(xpts, 3),

        "expected_minutes":
            round(
                expected_minutes,
                1,
            ),

        "start_probability":
            round(
                start_profile[
                    "start_probability"
                ],
                4,
            ),

        "goal_lambda":
            round(
                goal_lambda,
                4,
            ),

        "assist_lambda":
            round(
                assist_lambda,
                4,
            ),

        "clean_sheet_probability":
            round(
                clean_sheet_probability,
                4,
            ),

        "opponent_goal_lambda":
            round(
                opponent_goal_lambda,
                4,
            ),

        "dc_probability":
            round(
                dc_probability,
                4,
            ),

        "bonus_expectation":
            round(
                bonus_points,
                3,
            ),

        "xg90":
            round(
                xg90,
                4,
            ),

        "xa90":
            round(
                xa90,
                4,
            ),

        "attack_factor":
            round(
                attack_factor,
                4,
            ),

        "rotation_risk":
            round(
                rotation_risk,
                4,
            ),

        "congestion_risk":
            round(
                congestion_risk,
                4,
            ),

        "injury_risk":
            round(
                injury_risk,
                4,
            ),
    }


def horizon_value(
    fixture_projections,
    horizon,
):

    selected = (
        fixture_projections[
            :horizon
        ]
    )

    if not selected:
        return 0.0

    total = sum(
        item["xpts"]
        for item
        in selected
    )

    if NORMALISE_HORIZONS:

        return (
            total
            / len(selected)
        )

    return total


def build_player_projections(
    start_gameweek=None,
):

    bootstrap = get_bootstrap()

    fixtures = get_fixtures()

    teams = build_team_maps(
        bootstrap
    )

    positions = build_position_map(
        bootstrap
    )

    averages = (
        team_strength_averages(
            teams
        )
    )

    team_match_counts = (
        matches_played_by_team(
            fixtures
        )
    )

    overrides = load_overrides()

    if start_gameweek is None:
        start_gameweek = (
            get_next_gameweek()
        )

    upcoming_fixtures = [
        fixture
        for fixture
        in fixtures
        if (
            fixture.get("event")
            is not None
            and fixture["event"]
            >= start_gameweek
        )
    ]

    projections = []

    for player in bootstrap[
        "elements"
    ]:

        team_id = player["team"]

        team = teams[
            team_id
        ]

        position = positions[
            player["element_type"]
        ]

        override = (
            get_player_override(
                player,
                overrides,
            )
        )

        team_matches = (
            team_match_counts.get(
                team_id,
                0,
            )
        )

        start_profile = (
            calculate_start_profile(
                player,
                position,
                team_matches,
                override,
            )
        )

        player_fixtures = [
            fixture
            for fixture
            in upcoming_fixtures
            if (
                fixture["team_h"]
                == team_id
                or
                fixture["team_a"]
                == team_id
            )
        ]

        player_fixtures.sort(
            key=lambda item:
            (
                item.get(
                    "event",
                    999,
                ),
                item.get(
                    "kickoff_time",
                    "",
                ),
            )
        )

        projected_fixtures = []

        for fixture in (
            player_fixtures[:10]
        ):

            is_home = (
                fixture[
                    "team_h"
                ]
                == team_id
            )

            opponent_id = (
                fixture[
                    "team_a"
                ]
                if is_home
                else fixture[
                    "team_h"
                ]
            )

            opponent = teams[
                opponent_id
            ]

            projected = (
                project_fixture(
                    player,
                    position,
                    team,
                    opponent,
                    fixture,
                    is_home,
                    start_profile,
                    averages,
                    override,
                )
            )

            projected_fixtures.append(
                projected
            )

        gw1_value = horizon_value(
            projected_fixtures,
            1,
        )

        gw3_value = horizon_value(
            projected_fixtures,
            3,
        )

        gw6_value = horizon_value(
            projected_fixtures,
            6,
        )

        gw10_value = horizon_value(
            projected_fixtures,
            10,
        )

        strategy_score = (
            HORIZON_WEIGHTS[1]
            * gw1_value
            +
            HORIZON_WEIGHTS[3]
            * gw3_value
            +
            HORIZON_WEIGHTS[6]
            * gw6_value
            +
            HORIZON_WEIGHTS[10]
            * gw10_value
        )

        gw_xpts = {
            str(item["gameweek"]):
            item["xpts"]
            for item
            in projected_fixtures
        }

        projections.append(
            {
                "id":
                    player["id"],

                "name":
                    player[
                        "web_name"
                    ],

                "full_name":
                    (
                        f"{player.get('first_name', '')} "
                        f"{player.get('second_name', '')}"
                    ).strip(),

                "team_id":
                    team_id,

                "team":
                    team[
                        "short_name"
                    ],

                "position":
                    position,

                "price":
                    (
                        player[
                            "now_cost"
                        ]
                        / 10
                    ),

                "selected_by_percent":
                    safe_float(
                        player.get(
                            "selected_by_percent"
                        )
                    ),

                "form":
                    safe_float(
                        player.get(
                            "form"
                        )
                    ),

                "total_points":
                    player.get(
                        "total_points",
                        0,
                    ),

                "start_probability":
                    round(
                        start_profile[
                            "start_probability"
                        ],
                        4,
                    ),

                "expected_minutes":
                    round(
                        start_profile[
                            "expected_minutes"
                        ],
                        1,
                    ),

                "gw1_xpts":
                    round(
                        gw1_value,
                        3,
                    ),

                "gw3_avg_xpts":
                    round(
                        gw3_value,
                        3,
                    ),

                "gw6_avg_xpts":
                    round(
                        gw6_value,
                        3,
                    ),

                "gw10_avg_xpts":
                    round(
                        gw10_value,
                        3,
                    ),

                "strategy_score":
                    round(
                        strategy_score,
                        4,
                    ),

                "gw_xpts":
                    gw_xpts,

                "fixtures":
                    projected_fixtures,

                "penalty_taker":
                    bool(
                        override.get(
                            "penalty_taker",
                            False,
                        )
                    ),

                "set_piece_taker":
                    bool(
                        override.get(
                            "set_piece_taker",
                            False,
                        )
                    ),
            }
        )

    projections.sort(
        key=lambda player:
        player[
            "strategy_score"
        ],
        reverse=True,
    )

    return projections