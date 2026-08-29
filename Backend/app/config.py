"""
Central configuration for the FPL Decision Engine.

IMPORTANT:
All model tuning should happen here rather than being scattered
throughout the projection, simulation and optimisation code.
"""

# ============================================================
# MODEL VERSION
# ============================================================

MODEL_VERSION = "1.0.0"


# ============================================================
# FPL RULES
# ============================================================

MAX_FREE_TRANSFERS = 5
TRANSFER_HIT_COST = 4
MAX_PLAYERS_PER_TEAM = 3

SQUAD_SIZE = 15

SQUAD_POSITION_COUNTS = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

MIN_FORMATION = {
    "DEF": 3,
    "MID": 2,
    "FWD": 1,
}

MAX_FORMATION = {
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}


# ============================================================
# FPL SCORING
# ============================================================

GOAL_POINTS = {
    "GK": 10,
    "DEF": 6,
    "MID": 5,
    "FWD": 4,
}

ASSIST_POINTS = 3

CLEAN_SHEET_POINTS = {
    "GK": 4,
    "DEF": 4,
    "MID": 1,
    "FWD": 0,
}

YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2
PENALTY_MISS_POINTS = -2
PENALTY_SAVE_POINTS = 5

DEFENSIVE_CONTRIBUTION_POINTS = 2

DEFENSIVE_CONTRIBUTION_THRESHOLD = {
    "DEF": 10,
    "MID": 12,
    "FWD": 12,
}


# ============================================================
# YOUR MULTI-GAMEWEEK WEIGHTING
# ============================================================

# These are the explicit horizon weights retained from our model.

HORIZON_WEIGHTS = {
    1: 1.00,
    3: 0.80,
    6: 0.55,
    10: 0.30,
}

# Average points over each horizon rather than simply summing them.
# This stops the 10-GW horizon overwhelming everything simply
# because it contains more matches.

NORMALISE_HORIZONS = True


# ============================================================
# GAMEWEEK DECAY
# ============================================================

# Used by the legal transfer-path planner.
# Immediate points matter more than distant points.

GAMEWEEK_DECAY = 0.94


# ============================================================
# START / MINUTES MODEL
# ============================================================

# Prior probability that a healthy FPL player is a starter before
# enough current-season evidence exists.

START_PRIORS = {
    "GK": 0.72,
    "DEF": 0.70,
    "MID": 0.67,
    "FWD": 0.65,
}

# Number of pseudo-matches used to prevent GW1/GW2 samples from
# wildly changing projected starts.

START_PRIOR_STRENGTH = 2.0

DEFAULT_START_MINUTES = {
    "GK": 90,
    "DEF": 76,
    "MID": 74,
    "FWD": 72,
}

DEFAULT_SUB_APPEARANCE_PROBABILITY = {
    "GK": 0.02,
    "DEF": 0.25,
    "MID": 0.33,
    "FWD": 0.34,
}

DEFAULT_SUB_MINUTES = {
    "GK": 5,
    "DEF": 18,
    "MID": 20,
    "FWD": 21,
}


# ============================================================
# EARLY-SEASON SHRINKAGE
# ============================================================

# Player xG/xA rates are blended towards positional priors.
# Once a player has accumulated lots of minutes their observed
# numbers dominate.

RATE_SHRINKAGE_MINUTES = 450

XG90_PRIORS = {
    "GK": 0.001,
    "DEF": 0.045,
    "MID": 0.190,
    "FWD": 0.330,
}

XA90_PRIORS = {
    "GK": 0.005,
    "DEF": 0.075,
    "MID": 0.170,
    "FWD": 0.130,
}

BONUS90_PRIORS = {
    "GK": 0.30,
    "DEF": 0.35,
    "MID": 0.42,
    "FWD": 0.40,
}


# ============================================================
# FIXTURE MODEL
# ============================================================

BASE_GOALS_PER_TEAM = 1.45

HOME_ATTACK_MULTIPLIER = 1.08
AWAY_ATTACK_MULTIPLIER = 0.92

MIN_ATTACK_FACTOR = 0.60
MAX_ATTACK_FACTOR = 1.55

MIN_DEFENCE_FACTOR = 0.60
MAX_DEFENCE_FACTOR = 1.55


# ============================================================
# MODEL COMPONENT WEIGHTS
# ============================================================

# These coefficients are deliberately centralised.
# They can be calibrated later from actual prediction error.

MODEL_WEIGHTS = {

    # Underlying attacking production
    "xg": 1.00,
    "xa": 1.00,

    # Availability
    "minutes_security": 1.00,
    "start_security": 1.00,

    # Fixture/team strength
    "fixture": 1.00,

    # Current FPL form contributes information, but we deliberately
    # don't let short-term form dominate underlying performance.
    "form": 0.12,

    # Bonus-point expectation
    "bonus": 1.00,

    # Defensive contributions
    "defensive_contributions": 1.00,

    # Explicit penalties for uncertainty
    "rotation_penalty": 0.30,
    "congestion_penalty": 0.16,
    "injury_penalty": 0.55,

    # Role bonuses supplied via manual overrides
    "penalty_role": 0.18,
    "set_piece_role": 0.08,
}


# ============================================================
# ROLE MULTIPLIERS
# ============================================================

PENALTY_TAKER_XG_MULTIPLIER = 1.14
SET_PIECE_XA_MULTIPLIER = 1.08


# ============================================================
# DEFENSIVE CONTRIBUTION PRIORS
# ============================================================

DC_RETURN_PRIOR = {
    "GK": 0.00,
    "DEF": 0.25,
    "MID": 0.14,
    "FWD": 0.035,
}


# ============================================================
# DISCIPLINE PRIORS
# ============================================================

YELLOW90_PRIOR = {
    "GK": 0.04,
    "DEF": 0.16,
    "MID": 0.14,
    "FWD": 0.09,
}

RED90_PRIOR = {
    "GK": 0.002,
    "DEF": 0.008,
    "MID": 0.007,
    "FWD": 0.005,
}


# ============================================================
# GOALKEEPER MODEL
# ============================================================

GK_SAVE90_PRIOR = 3.2
GK_PENALTY_SAVE90_PRIOR = 0.025


# ============================================================
# CAPTAINCY
# ============================================================

# We previously wanted captaincy to consider the haul distribution,
# not merely raw mean xPts.

CAPTAIN_MEAN_WEIGHT = 1.00
CAPTAIN_10_PLUS_WEIGHT = 1.00
CAPTAIN_15_PLUS_WEIGHT = 1.50

CAPTAIN_SIMULATION_RUNS = 20000


# ============================================================
# MONTE CARLO
# ============================================================

DEFAULT_SIMULATION_RUNS = 20000

START_MINUTES_STD = 12

MIN_START_MINUTES = 45
MAX_START_MINUTES = 95

MIN_SUB_MINUTES = 1
MAX_SUB_MINUTES = 35

RANDOM_SEED = 42


# ============================================================
# TRANSFER OPTIMISER
# ============================================================

# Number of top possible replacements per position that the search
# keeps. Increasing this improves exhaustive coverage but costs CPU.

TRANSFER_CANDIDATES_PER_POSITION = 18

TRANSFER_RESULTS_RETURNED = 20

PATH_BEAM_WIDTH = 45

PATH_MAX_TRANSFERS_PER_GW = 2


# ============================================================
# MANUAL RISK ADJUSTMENTS
# ============================================================

DEFAULT_ROTATION_RISK = 0.0
DEFAULT_CONGESTION_RISK = 0.0
DEFAULT_INJURY_RISK = 0.0