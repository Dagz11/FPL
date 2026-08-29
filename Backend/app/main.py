from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import BaseModel

from .config import (
    MODEL_VERSION,
)

from .fpl_data import (
    get_latest_public_squad,
    get_next_gameweek,
)

from .optimiser import (
    best_transfers,
    optimal_xi,
    optimise_squad,
    plan_transfer_path,
)

from .projections import (
    build_player_projections,
)

from .simulation import (
    simulate_player,
)


app = FastAPI(
    title="FPL Decision Engine",
    description=(
        "Custom weighted FPL projection, "
        "Monte Carlo and multi-Gameweek "
        "optimisation engine."
    ),
    version=MODEL_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SquadRequest(BaseModel):

    player_ids: list[int]


class TransferRequest(BaseModel):

    player_ids: list[int]

    bank: float = 0.0

    free_transfers: int = 1

    max_moves: int = 2


class PathRequest(BaseModel):

    player_ids: list[int]

    bank: float = 0.0

    free_transfers: int = 1

    horizon: int = 6

    start_gameweek: int | None = None


def get_projection_lookup(
    start_gameweek=None,
):

    projections = (
        build_player_projections(
            start_gameweek
        )
    )

    lookup = {
        player["id"]: player
        for player
        in projections
    }

    return (
        projections,
        lookup,
    )


def resolve_squad(
    player_ids,
    lookup,
):

    missing = [
        player_id
        for player_id
        in player_ids
        if player_id
        not in lookup
    ]

    if missing:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unknown player IDs: "
                + str(missing)
            ),
        )

    return [
        lookup[player_id]
        for player_id
        in player_ids
    ]


@app.get("/")
def root():

    return {
        "name":
            "FPL Decision Engine",

        "status":
            "running",

        "version":
            MODEL_VERSION,

        "next_gameweek":
            get_next_gameweek(),

        "features": [
            "custom projections",
            "minutes model",
            "fixture modelling",
            "Monte Carlo",
            "optimal XI",
            "captaincy modelling",
            "squad optimisation",
            "transfer optimisation",
            "legal multi-GW transfer paths",
        ],
    }


@app.get("/players")
def players(
    limit: int = Query(
        100,
        ge=1,
        le=1000,
    ),
    start_gameweek: int | None = None,
):

    projections = (
        build_player_projections(
            start_gameweek
        )
    )

    return projections[
        :limit
    ]


@app.get(
    "/players/{player_id}"
)
def player(
    player_id: int,
):

    projections, lookup = (
        get_projection_lookup()
    )

    if player_id not in lookup:

        raise HTTPException(
            status_code=404,
            detail="Player not found",
        )

    return lookup[
        player_id
    ]


@app.get(
    "/simulate/{player_id}"
)
def simulate(
    player_id: int,
    runs: int = Query(
        20000,
        ge=1000,
        le=200000,
    ),
):

    projections, lookup = (
        get_projection_lookup()
    )

    if player_id not in lookup:

        raise HTTPException(
            status_code=404,
            detail="Player not found",
        )

    return simulate_player(
        lookup[player_id],
        runs=runs,
    )


@app.get(
    "/optimal-squad"
)
def optimal_squad(
    budget: float = Query(
        100.0,
        ge=70,
        le=150,
    ),
):

    projections = (
        build_player_projections()
    )

    return optimise_squad(
        projections,
        budget=budget,
    )


@app.post(
    "/optimal-xi"
)
def optimise_existing_xi(
    request: SquadRequest,
):

    projections, lookup = (
        get_projection_lookup()
    )

    squad = resolve_squad(
        request.player_ids,
        lookup,
    )

    if len(squad) != 15:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly 15 players "
                "are required."
            ),
        )

    return optimal_xi(
        squad
    )


@app.post(
    "/transfers"
)
def transfers(
    request: TransferRequest,
):

    projections, lookup = (
        get_projection_lookup()
    )

    squad = resolve_squad(
        request.player_ids,
        lookup,
    )

    if len(squad) != 15:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly 15 players "
                "are required."
            ),
        )

    return best_transfers(
        squad,
        projections,
        bank=request.bank,
        free_transfers=(
            request.free_transfers
        ),
        max_moves=request.max_moves,
    )


@app.post(
    "/transfer-path"
)
def transfer_path(
    request: PathRequest,
):

    start_gameweek = (
        request.start_gameweek
        or get_next_gameweek()
    )

    projections, lookup = (
        get_projection_lookup(
            start_gameweek
        )
    )

    squad = resolve_squad(
        request.player_ids,
        lookup,
    )

    if len(squad) != 15:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly 15 players "
                "are required."
            ),
        )

    return plan_transfer_path(
        squad,
        projections,
        start_gameweek=(
            start_gameweek
        ),
        horizon=request.horizon,
        bank=request.bank,
        free_transfers=(
            request.free_transfers
        ),
    )


@app.get(
    "/manager/{entry_id}"
)
def manager(
    entry_id: int,
):

    try:

        imported = (
            get_latest_public_squad(
                entry_id
            )
        )

        projections, lookup = (
            get_projection_lookup()
        )

        squad = resolve_squad(
            imported[
                "player_ids"
            ],
            lookup,
        )

        return {
            "imported_from_gameweek":
                imported[
                    "gameweek"
                ],

            "player_ids":
                imported[
                    "player_ids"
                ],

            "squad":
                squad,

            "optimal_xi":
                optimal_xi(
                    squad
                ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )