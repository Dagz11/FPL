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
    GK_PENALTY_SAVE90_PRIOR,
    GK_SAVE90_PRIOR,
    GOAL_POINTS,
    HOME_ATTACK_MULTIPLIER,
    AWAY_ATTACK_MULTIPLIER,
    HORIZON_WEIGHTS,
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


# ============================================================
# PATHS
# ============================================================

ROOT_DIRECTORY = Path(__file__).resolve().parents[2]

OVERRIDES_FILE = (
    ROOT_DIRECTORY
    / "data"
    / "manual_overrides.json"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(
    value,
    default=0.0,
):
    """
    Convert an arbitrary value to float.

    None, blank strings and invalid values return the supplied default.
    """

    try:
        if value in (
            None,
            "",
        ):
            return float(default)

        return float(value)

    except (
        ValueError,
        TypeError,
    ):
        return float(default)


def positive_float(
    value,
    default,
):
    """
    FPL occasionally supplies zero / missing strength values.

    For team-strength fields, zero is not a useful football strength
    estimate, so treat non-positive values as missing.
    """

    result = safe_float(
        value,
        default,
    )

    if (
        not math.isfinite(result)
        or result <= 0
    ):
        return float(default)

    return result


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


def sigmoid(
    value,
):
    # Protect against pathological overflow.
    value = clip(
        value,
        -60,
        60,
    )

    return (
        1.0
        /
        (
            1.0
            + math.exp(-value)
        )
    )


def poisson_probability(
    lam,
    value,
):
    """
    P(X=value) for a Poisson variable.
    """

    if lam < 0:
        return 0.0

    return (
        math.exp(-lam)
        * (lam ** value)
        / math.factorial(value)
    )


# ============================================================
# MANUAL OVERRIDES
# ============================================================

def load_overrides():
    """
    Load user-entered team-news / role overrides.

    Failure to read the optional file should never crash the engine.
    """

    if not OVERRIDES_FILE.exists():
        return {}

    try:
        with open(
            OVERRIDES_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        players = data.get(
            "players",
            {},
        )

        if not isinstance(
            players,
            dict,
        ):
            return {}

        return players

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
    ):
        return {}


def get_player_override(
    player,
    overrides,
):
    """
    Overrides can be addressed by:
    1. FPL player ID
    2. web_name
    3. full name
    """

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


# ============================================================
# FPL LOOKUP MAPS
# ============================================================

def build_team_maps(
    bootstrap,
):
    return {
        team["id"]: team
        for team
        in bootstrap["teams"]
    }


def build_position_map(
    bootstrap,
):
    """
    Convert FPL position labels to the engine's internal labels.

    FPL:
        GKP, DEF, MID, FWD

    Engine:
        GK, DEF, MID, FWD
    """

    position_map = {}

    for item in bootstrap[
        "element_types"
    ]:

        position = item.get(
            "singular_name_short",
            "",
        )

        if position == "GKP":
            position = "GK"

        if position not in (
            "GK",
            "DEF",
            "MID",
            "FWD",
        ):
            raise ValueError(
                f"Unknown FPL position: {position}"
            )

        position_map[
            item["id"]
        ] = position

    return position_map


# ============================================================
# MATCH / MINUTES INFORMATION
# ============================================================

def matches_played_by_team(
    fixtures,
):
    """
    Count completed league fixtures for each club.
    """

    counts = {}

    for fixture in fixtures:

        if not fixture.get(
            "finished",
            False,
        ):
            continue

        home = fixture[
            "team_h"
        ]

        away = fixture[
            "team_a"
        ]

        counts[home] = (
            counts.get(home, 0)
            + 1
        )

        counts[away] = (
            counts.get(away, 0)
            + 1
        )

    return counts


def availability_factor(
    player,
):
    """
    Combine the FPL availability percentage with player status.

    This is kept conservative because FPL's status flag is often
    more useful than treating every unflagged player as 100% fit.
    """

    chance = player.get(
        "chance_of_playing_next_round"
    )

    if chance is None:
        chance_factor = 1.0

    else:
        chance_factor = clip(
            safe_float(
                chance,
                100,
            )
            / 100,
            0,
            1,
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

    return clip(
        min(
            chance_factor,
            status_factor,
        ),
        0,
        1,
    )


def calculate_start_profile(
    player,
    position,
    team_matches,
    override,
):
    """
    Estimate:

    - probability of starting
    - expected minutes when starting
    - probability of appearing from bench
    - overall appearance probability
    - probability of reaching 60 minutes
    - unconditional expected minutes

    All probabilities are explicitly bounded [0, 1].
    Expected minutes are explicitly bounded [0, 90].
    """

    starts = max(
        0.0,
        safe_float(
            player.get("starts")
        ),
    )

    minutes = max(
        0.0,
        safe_float(
            player.get("minutes")
        ),
    )

    appearances = max(
        starts,
        safe_float(
            player.get(
                "appearances"
            )
        ),
    )

    prior = START_PRIORS[
        position
    ]

    # --------------------------------------------------------
    # START PROBABILITY
    # --------------------------------------------------------

    if team_matches > 0:

        observed_start_rate = (
            starts
            / team_matches
        )

        observed_start_rate = clip(
            observed_start_rate,
            0.0,
            1.0,
        )

        start_probability = (
            (
                observed_start_rate
                * team_matches
            )
            +
            (
                prior
                * START_PRIOR_STRENGTH
            )
        ) / (
            team_matches
            + START_PRIOR_STRENGTH
        )

    else:
        start_probability = prior

    start_probability *= (
        availability_factor(
            player
        )
    )

    start_probability = clip(
        start_probability,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # EXPECTED MINUTES IF STARTING
    # --------------------------------------------------------

    if starts > 0:

        # Aggregate FPL minutes include substitute minutes, meaning
        # minutes / starts can overstate true start minutes.
        #
        # The value is therefore bounded and blended back towards
        # a positional prior.
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
            +
            0.30
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

    start_minutes = clip(
        start_minutes,
        45,
        90,
    )

    # --------------------------------------------------------
    # SUBSTITUTE APPEARANCE PROBABILITY
    # --------------------------------------------------------

    if team_matches > 0:

        substitute_appearances = max(
            0,
            appearances - starts,
        )

        observed_sub_rate = (
            substitute_appearances
            / team_matches
        )

        # Blend observed information with a prior to avoid one
        # Gameweek creating an extreme estimate.
        prior_sub_rate = (
            DEFAULT_SUB_APPEARANCE_PROBABILITY[
                position
            ]
        )

        sub_probability_given_bench = (
            (
                observed_sub_rate
                * team_matches
            )
            +
            (
                prior_sub_rate
                * START_PRIOR_STRENGTH
            )
        ) / (
            team_matches
            + START_PRIOR_STRENGTH
        )

    else:
        sub_probability_given_bench = (
            DEFAULT_SUB_APPEARANCE_PROBABILITY[
                position
            ]
        )

    sub_probability_given_bench = clip(
        sub_probability_given_bench,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # MANUAL OVERRIDES
    # --------------------------------------------------------

    if (
        "start_probability"
        in override
    ):
        start_probability = clip(
            safe_float(
                override[
                    "start_probability"
                ]
            ),
            0.0,
            1.0,
        )

    if (
        "expected_start_minutes"
        in override
    ):
        start_minutes = clip(
            safe_float(
                override[
                    "expected_start_minutes"
                ]
            ),
            1,
            90,
        )

    if (
        "sub_appearance_probability"
        in override
    ):
        sub_probability_given_bench = (
            clip(
                safe_float(
                    override[
                        "sub_appearance_probability"
                    ]
                ),
                0.0,
                1.0,
            )
        )

    # --------------------------------------------------------
    # UNCONDITIONAL MINUTES
    # --------------------------------------------------------

    probability_benched = (
        1.0
        - start_probability
    )

    substitute_appearance_probability = (
        probability_benched
        * sub_probability_given_bench
    )

    expected_sub_minutes = clip(
        DEFAULT_SUB_MINUTES[
            position
        ],
        0,
        45,
    )

    expected_minutes = (
        start_probability
        * start_minutes
        +
        substitute_appearance_probability
        * expected_sub_minutes
    )

    expected_minutes = clip(
        expected_minutes,
        0.0,
        90.0,
    )

    play_probability = (
        start_probability
        +
        substitute_appearance_probability
    )

    play_probability = clip(
        play_probability,
        0.0,
        1.0,
    )

    # Smooth approximation of reaching the 60-minute threshold.
    probability_60_given_start = sigmoid(
        (
            start_minutes
            - 60
        )
        / 6
    )

    probability_60 = (
        start_probability
        * probability_60_given_start
    )

    probability_60 = clip(
        probability_60,
        0.0,
        1.0,
    )

    return {
        "start_probability":
            start_probability,

        "sub_probability":
            sub_probability_given_bench,

        "sub_appearance_probability":
            substitute_appearance_probability,

        "expected_start_minutes":
            start_minutes,

        "expected_minutes":
            expected_minutes,

        "play_probability":
            play_probability,

        "p60":
            probability_60,
    }


# ============================================================
# BAYESIAN / SHRUNK PLAYER RATES
# ============================================================

def shrunk_rate(
    observed,
    minutes,
    prior,
):
    """
    Shrink an observed cumulative statistic toward a positional
    per-90 prior.

    This is particularly important in GW1-GW5, where tiny samples
    otherwise create absurd player projections.
    """

    observed = max(
        0.0,
        safe_float(
            observed
        ),
    )

    minutes = max(
        0.0,
        safe_float(
            minutes
        ),
    )

    prior = max(
        0.0,
        safe_float(
            prior
        ),
    )

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

    result = (
        weight
        * observed_per_90
        +
        (
            1
            - weight
        )
        * prior
    )

    return max(
        0.0,
        result,
    )


# ============================================================
# TEAM STRENGTH
# ============================================================

def team_strength_averages(
    teams,
):
    """
    Calculate league averages using positive strength values only.

    This fixes the previous bug where a season with zero-valued
    strength fields could make the whole model collapse to its
    minimum fixture multiplier.
    """

    fields = [
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]

    averages = {}

    for field in fields:

        values = []

        for team in teams.values():

            value = safe_float(
                team.get(field),
                0,
            )

            if (
                math.isfinite(value)
                and value > 0
            ):
                values.append(
                    value
                )

        if values:
            averages[field] = float(
                np.mean(values)
            )

        else:
            averages[field] = 1000.0

    return averages


def valid_team_strength(
    team,
    field,
):
    value = safe_float(
        team.get(field),
        0,
    )

    return (
        math.isfinite(value)
        and value > 0
    )


# ============================================================
# FIXTURE DIFFICULTY
# ============================================================

def fixture_difficulty(
    fixture,
    is_home,
):
    """
    Get FPL FDR from the player's team's perspective.
    """

    if is_home:
        raw = fixture.get(
            "team_h_difficulty",
            3,
        )

    else:
        raw = fixture.get(
            "team_a_difficulty",
            3,
        )

    difficulty = int(
        round(
            safe_float(
                raw,
                3,
            )
        )
    )

    return int(
        clip(
            difficulty,
            1,
            5,
        )
    )


def fixture_multipliers(
    team,
    opponent,
    is_home,
    averages,
    fixture,
):
    """
    Estimate the player's attacking multiplier.

    The model blends:
    - FPL team attack strengths
    - opponent defensive strengths
    - home/away
    - official FDR

    If FPL strength fields are unavailable or zero, it falls back
    cleanly to FDR rather than every fixture hitting the same floor.
    """

    difficulty = fixture_difficulty(
        fixture,
        is_home,
    )

    fdr_attack_factor = {
        1: 1.25,
        2: 1.12,
        3: 1.00,
        4: 0.86,
        5: 0.72,
    }[
        difficulty
    ]

    if is_home:

        attack_field = (
            "strength_attack_home"
        )

        defence_field = (
            "strength_defence_away"
        )

        venue_multiplier = (
            HOME_ATTACK_MULTIPLIER
        )

    else:

        attack_field = (
            "strength_attack_away"
        )

        defence_field = (
            "strength_defence_home"
        )

        venue_multiplier = (
            AWAY_ATTACK_MULTIPLIER
        )

    strength_is_available = (
        valid_team_strength(
            team,
            attack_field,
        )
        and valid_team_strength(
            opponent,
            defence_field,
        )
    )

    if strength_is_available:

        team_attack = positive_float(
            team.get(
                attack_field
            ),
            averages[
                attack_field
            ],
        )

        opponent_defence = (
            positive_float(
                opponent.get(
                    defence_field
                ),
                averages[
                    defence_field
                ],
            )
        )

        attack_average = max(
            averages[
                attack_field
            ],
            1,
        )

        defence_average = max(
            averages[
                defence_field
            ],
            1,
        )

        strength_factor = (
            (
                team_attack
                / attack_average
            )
            *
            (
                defence_average
                / opponent_defence
            )
            *
            venue_multiplier
        )

        # FPL team strengths carry more weight when available.
        attack_factor = (
            0.70
            * strength_factor
            +
            0.30
            * fdr_attack_factor
        )

    else:

        # Fallback: FDR + venue only.
        attack_factor = (
            fdr_attack_factor
            * venue_multiplier
        )

    return clip(
        attack_factor,
        0.60,
        1.50,
    )


def expected_opponent_goals(
    team,
    opponent,
    is_home,
    averages,
    fixture,
):
    """
    Estimate expected goals conceded by the player's team.

    This drives:
    - clean-sheet probability
    - GK/DEF concession penalties

    Crucially, Arsenal away can no longer produce the same lambda
    as Coventry away unless the underlying inputs genuinely imply it.
    """

    difficulty = fixture_difficulty(
        fixture,
        is_home,
    )

    # Expected opponent goals implied by FDR.
    fdr_goal_lambda = {
        1: 0.78,
        2: 1.05,
        3: 1.38,
        4: 1.78,
        5: 2.20,
    }[
        difficulty
    ]

    if is_home:

        opponent_attack_field = (
            "strength_attack_away"
        )

        team_defence_field = (
            "strength_defence_home"
        )

        venue_multiplier = (
            AWAY_ATTACK_MULTIPLIER
        )

    else:

        opponent_attack_field = (
            "strength_attack_home"
        )

        team_defence_field = (
            "strength_defence_away"
        )

        venue_multiplier = (
            HOME_ATTACK_MULTIPLIER
        )

    strength_is_available = (
        valid_team_strength(
            opponent,
            opponent_attack_field,
        )
        and valid_team_strength(
            team,
            team_defence_field,
        )
    )

    if strength_is_available:

        opponent_attack = (
            positive_float(
                opponent.get(
                    opponent_attack_field
                ),
                averages[
                    opponent_attack_field
                ],
            )
        )

        team_defence = (
            positive_float(
                team.get(
                    team_defence_field
                ),
                averages[
                    team_defence_field
                ],
            )
        )

        attack_average = max(
            averages[
                opponent_attack_field
            ],
            1,
        )

        defence_average = max(
            averages[
                team_defence_field
            ],
            1,
        )

        strength_lambda = (
            BASE_GOALS_PER_TEAM
            *
            (
                opponent_attack
                / attack_average
            )
            *
            (
                defence_average
                / team_defence
            )
            *
            venue_multiplier
        )

        # Blend two independent signals.
        goal_lambda = (
            0.70
            * strength_lambda
            +
            0.30
            * fdr_goal_lambda
        )

    else:

        goal_lambda = (
            fdr_goal_lambda
        )

    return clip(
        goal_lambda,
        0.35,
        3.25,
    )


# ============================================================
# EXPECTED FPL SCORING HELPERS
# ============================================================

def expected_goals_conceded_penalty(
    goal_lambda,
):
    """
    GK/DEF lose one point for every two goals conceded.

    Compute E[floor(goals / 2)] from a Poisson distribution.
    """

    goal_lambda = max(
        0.0,
        goal_lambda,
    )

    expectation = 0.0

    # 0-10 covers virtually all realistic football score mass.
    for goals in range(
        0,
        11,
    ):

        probability = (
            poisson_probability(
                goal_lambda,
                goals,
            )
        )

        expectation += (
            math.floor(
                goals / 2
            )
            * probability
        )

    return expectation


def expected_save_points(
    save_lambda,
):
    """
    Goalkeepers receive one point for every three saves.

    This uses the Poisson distribution rather than simply doing
    expected_saves / 3, which slightly overstates low save volumes.
    """

    save_lambda = max(
        0.0,
        save_lambda,
    )

    expectation = 0.0

    # 0-20 is more than enough for normal PL goalkeeper save counts.
    for saves in range(
        0,
        21,
    ):

        probability = (
            poisson_probability(
                save_lambda,
                saves,
            )
        )

        expectation += (
            math.floor(
                saves / 3
            )
            * probability
        )

    return expectation


def defensive_contribution_probability(
    player,
    position,
    expected_minutes,
):
    """
    Estimate probability of earning the 2-point defensive
    contribution return.
    """

    if position == "GK":
        return 0.0

    observed_dc_points = max(
        0.0,
        safe_float(
            player.get(
                "defensive_contribution"
            )
        ),
    )

    minutes = max(
        0.0,
        safe_float(
            player.get("minutes")
        ),
    )

    prior = DC_RETURN_PRIOR[
        position
    ]

    if (
        observed_dc_points > 0
        and minutes > 0
    ):

        observed_returns = (
            observed_dc_points
            / 2
        )

        matches_90 = (
            minutes
            / 90
        )

        observed_rate = (
            observed_returns
            / max(
                matches_90,
                1,
            )
        )

        probability = (
            0.60
            * clip(
                observed_rate,
                0,
                1,
            )
            +
            0.40
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

    return clip(
        probability
        * minutes_factor,
        0,
        1,
    )


# ============================================================
# SINGLE FIXTURE PROJECTION
# ============================================================

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
    """
    Produce a complete expected-points projection for one fixture.
    """

    minutes_played_so_far = max(
        0.0,
        safe_float(
            player.get("minutes")
        ),
    )

    # --------------------------------------------------------
    # PLAYER ATTACKING RATES
    # --------------------------------------------------------

    xg90 = shrunk_rate(
        player.get(
            "expected_goals"
        ),
        minutes_played_so_far,
        XG90_PRIORS[
            position
        ],
    )

    xa90 = shrunk_rate(
        player.get(
            "expected_assists"
        ),
        minutes_played_so_far,
        XA90_PRIORS[
            position
        ],
    )

    bonus90 = shrunk_rate(
        player.get("bonus"),
        minutes_played_so_far,
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

    # --------------------------------------------------------
    # FIXTURE STRENGTH
    # --------------------------------------------------------

    difficulty = fixture_difficulty(
        fixture,
        is_home,
    )

    attack_factor = (
        fixture_multipliers(
            team,
            opponent,
            is_home,
            averages,
            fixture,
        )
    )

    opponent_goal_lambda = (
        expected_opponent_goals(
            team,
            opponent,
            is_home,
            averages,
            fixture,
        )
    )

    # --------------------------------------------------------
    # MINUTES
    # --------------------------------------------------------

    expected_minutes = clip(
        start_profile[
            "expected_minutes"
        ],
        0,
        90,
    )

    expected_start_minutes = clip(
        start_profile[
            "expected_start_minutes"
        ],
        0,
        90,
    )

    # --------------------------------------------------------
    # ATTACKING RETURNS
    # --------------------------------------------------------

    goal_lambda = (
        xg90
        * (
            expected_minutes
            / 90
        )
        * attack_factor
    )

    assist_lambda = (
        xa90
        * (
            expected_minutes
            / 90
        )
        * attack_factor
    )

    goal_lambda = max(
        0.0,
        goal_lambda,
    )

    assist_lambda = max(
        0.0,
        assist_lambda,
    )

    # --------------------------------------------------------
    # APPEARANCE POINTS
    # --------------------------------------------------------

    appearance_points = (
        start_profile[
            "play_probability"
        ]
        +
        start_profile[
            "p60"
        ]
    )

    # --------------------------------------------------------
    # GOAL / ASSIST POINTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CLEAN SHEET
    # --------------------------------------------------------

    # A player subbed after 60 keeps a clean sheet if no goal was
    # conceded while he was on the pitch. Use expected start minutes
    # rather than blindly assuming he plays all 90.
    cs_exposure_fraction = clip(
        expected_start_minutes
        / 90,
        0,
        1,
    )

    clean_sheet_lambda = (
        opponent_goal_lambda
        * cs_exposure_fraction
    )

    clean_sheet_probability_on_pitch = (
        math.exp(
            -clean_sheet_lambda
        )
    )

    clean_sheet_points = (
        start_profile[
            "p60"
        ]
        * clean_sheet_probability_on_pitch
        * CLEAN_SHEET_POINTS[
            position
        ]
    )

    # --------------------------------------------------------
    # BONUS
    # --------------------------------------------------------

    bonus_points = (
        bonus90
        * expected_minutes
        / 90
        * MODEL_WEIGHTS[
            "bonus"
        ]
    )

    # Nobody can receive more than 3 bonus in a match.
    bonus_points = clip(
        bonus_points,
        0,
        3
        * start_profile[
            "play_probability"
        ],
    )

    # --------------------------------------------------------
    # DEFENSIVE CONTRIBUTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DISCIPLINE
    # --------------------------------------------------------

    yellow90 = shrunk_rate(
        player.get(
            "yellow_cards"
        ),
        minutes_played_so_far,
        YELLOW90_PRIOR[
            position
        ],
    )

    red90 = shrunk_rate(
        player.get(
            "red_cards"
        ),
        minutes_played_so_far,
        RED90_PRIOR[
            position
        ],
    )

    discipline_points = (
        -yellow90
        * expected_minutes
        / 90
        -
        3
        * red90
        * expected_minutes
        / 90
    )

    # --------------------------------------------------------
    # GOALS CONCEDED DEDUCTIONS
    # --------------------------------------------------------

    concession_points = 0.0

    if position in (
        "GK",
        "DEF",
    ):

        concession_lambda_on_pitch = (
            opponent_goal_lambda
            * cs_exposure_fraction
        )

        expected_penalty_units = (
            expected_goals_conceded_penalty(
                concession_lambda_on_pitch
            )
        )

        concession_points = (
            -expected_penalty_units
            * start_profile[
                "p60"
            ]
        )

    # --------------------------------------------------------
    # GOALKEEPER SAVES
    # --------------------------------------------------------

    save_points = 0.0
    penalty_save_points = 0.0
    expected_saves = 0.0

    if position == "GK":

        saves90 = shrunk_rate(
            player.get("saves"),
            minutes_played_so_far,
            GK_SAVE90_PRIOR,
        )

        # Tougher fixtures generally create more shots/saves.
        # Modest adjustment only; do not let saves cancel out all
        # clean-sheet downside.
        save_fixture_factor = clip(
            opponent_goal_lambda
            / BASE_GOALS_PER_TEAM,
            0.70,
            1.45,
        )

        expected_saves = (
            saves90
            * expected_minutes
            / 90
            * save_fixture_factor
        )

        save_points = (
            expected_save_points(
                expected_saves
            )
        )

        penalty_save90 = shrunk_rate(
            player.get(
                "penalties_saved"
            ),
            minutes_played_so_far,
            GK_PENALTY_SAVE90_PRIOR,
        )

        penalty_save_points = (
            penalty_save90
            * expected_minutes
            / 90
            * 5
        )

    # --------------------------------------------------------
    # BASE EXPECTED POINTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------

    form = max(
        0.0,
        safe_float(
            player.get("form")
        ),
    )

    # Very small modifier only.
    # Form is already partly represented in current-season stats.
    form_adjustment = (
        MODEL_WEIGHTS[
            "form"
        ]
        * form
        * 0.10
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

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

    # Never allow risk settings to create a negative multiplier.
    risk_penalty = clip(
        risk_penalty,
        0.0,
        0.85,
    )

    xpts = (
        raw_xpts
        + form_adjustment
    ) * (
        1
        - risk_penalty
    )

    xpts = max(
        0.0,
        xpts,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

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

        "opponent":
            opponent_name,

        "home":
            is_home,

        "fdr":
            difficulty,

        "xpts":
            round(
                xpts,
                3,
            ),

        "raw_xpts":
            round(
                raw_xpts,
                3,
            ),

        "expected_minutes":
            round(
                expected_minutes,
                1,
            ),

        "expected_start_minutes":
            round(
                expected_start_minutes,
                1,
            ),

        "start_probability":
            round(
                start_profile[
                    "start_probability"
                ],
                4,
            ),

        "play_probability":
            round(
                start_profile[
                    "play_probability"
                ],
                4,
            ),

        "p60":
            round(
                start_profile[
                    "p60"
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
                clean_sheet_probability_on_pitch,
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

        "expected_saves":
            round(
                expected_saves,
                3,
            ),

        "save_points_expectation":
            round(
                save_points,
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

        "risk_penalty":
            round(
                risk_penalty,
                4,
            ),
    }


# ============================================================
# GAMEWEEK AGGREGATION
# ============================================================

def aggregate_gameweek_xpts(
    fixture_projections,
    start_gameweek,
    gameweeks=10,
):
    """
    Aggregate fixture xPts into Gameweeks.

    This fixes two important issues:

    1. Double Gameweeks:
       both fixture projections are summed.

    2. Blank Gameweeks:
       the Gameweek remains present with 0 xPts.
    """

    gw_totals = {
        gameweek: 0.0
        for gameweek
        in range(
            start_gameweek,
            start_gameweek
            + gameweeks,
        )
    }

    for fixture in fixture_projections:

        gameweek = fixture.get(
            "gameweek"
        )

        if gameweek not in gw_totals:
            continue

        gw_totals[
            gameweek
        ] += fixture[
            "xpts"
        ]

    return gw_totals


def horizon_value(
    gameweek_totals,
    start_gameweek,
    horizon,
):
    """
    Calculate horizon value over actual Gameweeks.

    Example at GW2:

        horizon=3 -> GW2 + GW3 + GW4

    A blank counts as zero.
    A double includes both fixtures.
    """

    values = [
        gameweek_totals.get(
            gameweek,
            0.0,
        )
        for gameweek
        in range(
            start_gameweek,
            start_gameweek
            + horizon,
        )
    ]

    if not values:
        return 0.0

    total = sum(
        values
    )

    if NORMALISE_HORIZONS:

        return (
            total
            / horizon
        )

    return total


# ============================================================
# SANITY CHECKS
# ============================================================

def build_projection_flags(
    start_profile,
    projected_fixtures,
):
    """
    Add diagnostic warnings rather than silently trusting unusual
    model output.
    """

    flags = []

    if not (
        0
        <= start_profile[
            "start_probability"
        ]
        <= 1
    ):
        flags.append(
            "INVALID_START_PROBABILITY"
        )

    if not (
        0
        <= start_profile[
            "expected_minutes"
        ]
        <= 90
    ):
        flags.append(
            "INVALID_EXPECTED_MINUTES"
        )

    if projected_fixtures:

        for fixture in projected_fixtures:

            if not (
                0
                <= fixture[
                    "clean_sheet_probability"
                ]
                <= 1
            ):
                flags.append(
                    "INVALID_CLEAN_SHEET_PROBABILITY"
                )

                break

        if len(
            projected_fixtures
        ) >= 3:

            xpts_values = [
                item["xpts"]
                for item
                in projected_fixtures
            ]

            fdr_values = {
                item["fdr"]
                for item
                in projected_fixtures
            }

            # Different fixture difficulties but almost perfectly
            # identical model outputs is suspicious.
            if (
                len(fdr_values) > 1
                and (
                    max(xpts_values)
                    - min(xpts_values)
                ) < 0.01
            ):
                flags.append(
                    "SUSPICIOUSLY_FLAT_FIXTURE_PROJECTIONS"
                )

    return sorted(
        set(flags)
    )


# ============================================================
# FULL PLAYER PROJECTION BUILD
# ============================================================

def build_player_projections(
    start_gameweek=None,
):
    """
    Build complete projections for every FPL player.

    Output remains compatible with optimiser.py while also exposing
    clearer next-GW / next-3 / next-6 / next-10 names.
    """

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

    start_gameweek = int(
        start_gameweek
    )

    final_projection_gameweek = (
        start_gameweek
        + 9
    )

    # Only fixtures occurring inside the next ten Gameweeks.
    #
    # DO NOT use [:10], because a Double Gameweek means ten fixtures
    # is not necessarily ten Gameweeks.
    upcoming_fixtures = [
        fixture
        for fixture
        in fixtures
        if (
            fixture.get(
                "event"
            )
            is not None
            and start_gameweek
            <= fixture["event"]
            <= final_projection_gameweek
        )
    ]

    projections = []

    for player in bootstrap[
        "elements"
    ]:

        team_id = player[
            "team"
        ]

        team = teams[
            team_id
        ]

        position = positions[
            player[
                "element_type"
            ]
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

        # ----------------------------------------------------
        # PLAYER FIXTURES
        # ----------------------------------------------------

        player_fixtures = [
            fixture
            for fixture
            in upcoming_fixtures
            if (
                fixture[
                    "team_h"
                ]
                == team_id
                or
                fixture[
                    "team_a"
                ]
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
                    "kickoff_time"
                )
                or "",
                item.get(
                    "id",
                    0,
                ),
            )
        )

        projected_fixtures = []

        for fixture in player_fixtures:

            is_home = (
                fixture[
                    "team_h"
                ]
                == team_id
            )

            if is_home:

                opponent_id = (
                    fixture[
                        "team_a"
                    ]
                )

            else:

                opponent_id = (
                    fixture[
                        "team_h"
                    ]
                )

            opponent = teams[
                opponent_id
            ]

            projected_fixture = (
                project_fixture(
                    player=player,
                    position=position,
                    team=team,
                    opponent=opponent,
                    fixture=fixture,
                    is_home=is_home,
                    start_profile=(
                        start_profile
                    ),
                    averages=averages,
                    override=override,
                )
            )

            projected_fixtures.append(
                projected_fixture
            )

        # ----------------------------------------------------
        # AGGREGATE BY GAMEWEEK
        # ----------------------------------------------------

        gameweek_totals = (
            aggregate_gameweek_xpts(
                projected_fixtures,
                start_gameweek,
                gameweeks=10,
            )
        )

        next_gw_value = (
            horizon_value(
                gameweek_totals,
                start_gameweek,
                1,
            )
        )

        next_3_value = (
            horizon_value(
                gameweek_totals,
                start_gameweek,
                3,
            )
        )

        next_6_value = (
            horizon_value(
                gameweek_totals,
                start_gameweek,
                6,
            )
        )

        next_10_value = (
            horizon_value(
                gameweek_totals,
                start_gameweek,
                10,
            )
        )

        # ----------------------------------------------------
        # OUR MULTI-GW STRATEGY SCORE
        # ----------------------------------------------------

        strategy_score = (
            HORIZON_WEIGHTS[
                1
            ]
            * next_gw_value
            +
            HORIZON_WEIGHTS[
                3
            ]
            * next_3_value
            +
            HORIZON_WEIGHTS[
                6
            ]
            * next_6_value
            +
            HORIZON_WEIGHTS[
                10
            ]
            * next_10_value
        )

        # optimiser.py currently expects string GW keys.
        gw_xpts = {
            str(gameweek):
                round(
                    xpts,
                    3,
                )
            for (
                gameweek,
                xpts,
            )
            in gameweek_totals.items()
        }

        flags = (
            build_projection_flags(
                start_profile,
                projected_fixtures,
            )
        )

        full_name = (
            f"{player.get('first_name', '')} "
            f"{player.get('second_name', '')}"
        ).strip()

        projection = {
            "id":
                player["id"],

            "name":
                player[
                    "web_name"
                ],

            "full_name":
                full_name,

            "team_id":
                team_id,

            "team":
                team[
                    "short_name"
                ],

            "position":
                position,

            "price":
                round(
                    player[
                        "now_cost"
                    ]
                    / 10,
                    1,
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

            # -----------------------------------------------
            # MINUTES
            # -----------------------------------------------

            "start_probability":
                round(
                    start_profile[
                        "start_probability"
                    ],
                    4,
                ),

            "sub_appearance_probability":
                round(
                    start_profile[
                        "sub_appearance_probability"
                    ],
                    4,
                ),

            "play_probability":
                round(
                    start_profile[
                        "play_probability"
                    ],
                    4,
                ),

            "expected_start_minutes":
                round(
                    start_profile[
                        "expected_start_minutes"
                    ],
                    1,
                ),

            "expected_minutes":
                round(
                    start_profile[
                        "expected_minutes"
                    ],
                    1,
                ),

            "p60":
                round(
                    start_profile[
                        "p60"
                    ],
                    4,
                ),

            # -----------------------------------------------
            # CLEAR NEW HORIZON NAMES
            # -----------------------------------------------

            "next_gameweek":
                start_gameweek,

            "next_gw_xpts":
                round(
                    next_gw_value,
                    3,
                ),

            "next_3_avg_xpts":
                round(
                    next_3_value,
                    3,
                ),

            "next_6_avg_xpts":
                round(
                    next_6_value,
                    3,
                ),

            "next_10_avg_xpts":
                round(
                    next_10_value,
                    3,
                ),

            # -----------------------------------------------
            # LEGACY KEYS
            #
            # Keep these temporarily because optimiser.py is
            # already using them.
            # -----------------------------------------------

            "gw1_xpts":
                round(
                    next_gw_value,
                    3,
                ),

            "gw3_avg_xpts":
                round(
                    next_3_value,
                    3,
                ),

            "gw6_avg_xpts":
                round(
                    next_6_value,
                    3,
                ),

            "gw10_avg_xpts":
                round(
                    next_10_value,
                    3,
                ),

            # -----------------------------------------------
            # STRATEGY
            # -----------------------------------------------

            "strategy_score":
                round(
                    strategy_score,
                    4,
                ),

            "gw_xpts":
                gw_xpts,

            "fixtures":
                projected_fixtures,

            # -----------------------------------------------
            # ROLES
            # -----------------------------------------------

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

            # -----------------------------------------------
            # VALIDATION
            # -----------------------------------------------

            "model_flags":
                flags,

            "model_valid":
                len(flags) == 0,
        }

        projections.append(
            projection
        )

    # --------------------------------------------------------
    # RANK BY OUR FULL STRATEGY SCORE
    # --------------------------------------------------------

    projections.sort(
        key=lambda player:
        (
            player[
                "strategy_score"
            ],
            player[
                "next_gw_xpts"
            ],
        ),
        reverse=True,
    )

    return projections