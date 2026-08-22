from fastapi import FastAPI

from .fpl_data import get_players
from .optimiser import rank_players


app = FastAPI(
    title="FPL Decision Engine",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "name": "FPL Decision Engine",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/players")
def players():
    players_data = get_players()

    return rank_players(players_data)

