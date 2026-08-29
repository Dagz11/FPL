import numpy as np

from .config import (
    ASSIST_POINTS,
    CAPTAIN_10_PLUS_WEIGHT,
    CAPTAIN_15_PLUS_WEIGHT,
    CAPTAIN_MEAN_WEIGHT,
    CLEAN_SHEET_POINTS,
    DEFAULT_SIMULATION_RUNS,
    DEFAULT_SUB_MINUTES,
    GOAL_POINTS,
    MAX_START_MINUTES,
    MAX_SUB_MINUTES,
    MIN_START_MINUTES,
    MIN_SUB_MINUTES,
    RANDOM_SEED,
    START_MINUTES_STD,
)


def simulate_player(
    player_projection,
    runs=DEFAULT_SIMULATION_RUNS,
    seed=RANDOM_SEED,
):

    if not player_projection[
        "fixtures"
    ]:

        return {
            "mean": 0,
            "median": 0,
            "std": 0,
            "p_blank": 1,
            "p_10_plus": 0,
            "p_15_plus": 0,
        }

    fixture = (
        player_projection[
            "fixtures"
        ][0]
    )

    position = (
        player_projection[
            "position"
        ]
    )

    rng = np.random.default_rng(
        seed
        + player_projection["id"]
    )

    start_probability = (
        fixture[
            "start_probability"
        ]
    )

    expected_minutes = (
        fixture[
            "expected_minutes"
        ]
    )

    starts = (
        rng.random(runs)
        < start_probability
    )

    start_minutes_mean = max(
        60,
        expected_minutes
        / max(
            start_probability,
            0.35,
        ),
    )

    start_minutes = (
        rng.normal(
            start_minutes_mean,
            START_MINUTES_STD,
            runs,
        )
    )

    start_minutes = np.clip(
        start_minutes,
        MIN_START_MINUTES,
        MAX_START_MINUTES,
    )

    sub_probability = np.clip(
        0.30
        + (
            0.25
            * (
                1
                - start_probability
            )
        ),
        0.15,
        0.65,
    )

    sub_appearances = (
        (~starts)
        &
        (
            rng.random(runs)
            < sub_probability
        )
    )

    sub_minutes = rng.normal(
        DEFAULT_SUB_MINUTES[
            position
        ],
        7,
        runs,
    )

    sub_minutes = np.clip(
        sub_minutes,
        MIN_SUB_MINUTES,
        MAX_SUB_MINUTES,
    )

    minutes = np.where(
        starts,
        start_minutes,
        np.where(
            sub_appearances,
            sub_minutes,
            0,
        ),
    )

    played = (
        minutes > 0
    )

    played_60 = (
        minutes >= 60
    )

    points = np.zeros(
        runs,
        dtype=float,
    )

    # Appearance points
    points += np.where(
        played_60,
        2,
        np.where(
            played,
            1,
            0,
        ),
    )

    minute_factor = (
        minutes
        / max(
            expected_minutes,
            1,
        )
    )

    goal_lambda = (
        fixture[
            "goal_lambda"
        ]
        * minute_factor
    )

    assist_lambda = (
        fixture[
            "assist_lambda"
        ]
        * minute_factor
    )

    goals = rng.poisson(
        np.maximum(
            goal_lambda,
            0,
        )
    )

    assists = rng.poisson(
        np.maximum(
            assist_lambda,
            0,
        )
    )

    points += (
        goals
        * GOAL_POINTS[
            position
        ]
    )

    points += (
        assists
        * ASSIST_POINTS
    )

    opponent_goal_lambda = (
        fixture[
            "opponent_goal_lambda"
        ]
    )

    goals_conceded = (
        rng.poisson(
            opponent_goal_lambda,
            runs,
        )
    )

    clean_sheet = (
        played_60
        &
        (
            goals_conceded
            == 0
        )
    )

    points += (
        clean_sheet
        * CLEAN_SHEET_POINTS[
            position
        ]
    )

    if position in (
        "GK",
        "DEF",
    ):

        points -= (
            played_60
            * (
                goals_conceded
                // 2
            )
        )

    if position == "GK":

        save_lambda = (
            3.2
            * minutes
            / 90
        )

        saves = rng.poisson(
            save_lambda
        )

        points += (
            saves
            // 3
        )

    dc_probability = (
        fixture[
            "dc_probability"
        ]
        * np.minimum(
            minutes
            / 75,
            1,
        )
    )

    dc_returns = (
        rng.random(runs)
        <
        dc_probability
    )

    points += (
        dc_returns
        * 2
    )

    yellow_probability = np.minimum(
        0.13
        * minutes
        / 90,
        0.30,
    )

    yellow_cards = (
        rng.random(runs)
        <
        yellow_probability
    )

    points -= yellow_cards

    red_probability = np.minimum(
        0.006
        * minutes
        / 90,
        0.03,
    )

    red_cards = (
        rng.random(runs)
        <
        red_probability
    )

    points -= (
        red_cards
        * 3
    )

    bonus_mean = max(
        fixture[
            "bonus_expectation"
        ],
        0,
    )

    bonus = rng.poisson(
        bonus_mean,
        runs,
    )

    bonus = np.clip(
        bonus,
        0,
        3,
    )

    points += bonus

    mean = float(
        np.mean(points)
    )

    median = float(
        np.median(points)
    )

    std = float(
        np.std(points)
    )

    p_blank = float(
        np.mean(
            points <= 2
        )
    )

    p_10_plus = float(
        np.mean(
            points >= 10
        )
    )

    p_15_plus = float(
        np.mean(
            points >= 15
        )
    )

    captain_score = (
        CAPTAIN_MEAN_WEIGHT
        * mean
        +
        CAPTAIN_10_PLUS_WEIGHT
        * p_10_plus
        +
        CAPTAIN_15_PLUS_WEIGHT
        * p_15_plus
    )

    return {
        "player_id":
            player_projection[
                "id"
            ],

        "name":
            player_projection[
                "name"
            ],

        "mean":
            round(
                mean,
                3,
            ),

        "median":
            round(
                median,
                3,
            ),

        "std":
            round(
                std,
                3,
            ),

        "p_blank":
            round(
                p_blank,
                4,
            ),

        "p_10_plus":
            round(
                p_10_plus,
                4,
            ),

        "p_15_plus":
            round(
                p_15_plus,
                4,
            ),

        "p_zero_minutes":
            round(
                float(
                    np.mean(
                        minutes == 0
                    )
                ),
                4,
            ),

        "mean_minutes":
            round(
                float(
                    np.mean(
                        minutes
                    )
                ),
                2,
            ),

        "captain_score":
            round(
                captain_score,
                4,
            ),

        "q10":
            round(
                float(
                    np.quantile(
                        points,
                        0.10,
                    )
                ),
                2,
            ),

        "q90":
            round(
                float(
                    np.quantile(
                        points,
                        0.90,
                    )
                ),
                2,
            ),

        "runs":
            runs,
    }