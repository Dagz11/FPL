import time
from typing import Any

import requests


BOOTSTRAP_URL = (
    "https://fantasy.premierleague.com/api/bootstrap-static/"
)

FIXTURES_URL = (
    "https://fantasy.premierleague.com/api/fixtures/"
)

ENTRY_URL = (
    "https://fantasy.premierleague.com/api/entry/{entry_id}/"
)

PICKS_URL = (
    "https://fantasy.premierleague.com/api/entry/"
    "{entry_id}/event/{event}/picks/"
)


CACHE_SECONDS = 300


class SimpleCache:

    def __init__(self):
        self.data = {}

    def get(self, key):
        cached = self.data.get(key)

        if not cached:
            return None

        value, timestamp = cached

        if time.time() - timestamp > CACHE_SECONDS:
            del self.data[key]
            return None

        return value

    def set(self, key, value):
        self.data[key] = (value, time.time())


cache = SimpleCache()

session = requests.Session()

session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 FPL-Decision-Engine/1.0"
        )
    }
)


def _request_json(url: str) -> Any:

    response = session.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_bootstrap(force_refresh=False):

    key = "bootstrap"

    if not force_refresh:
        cached = cache.get(key)

        if cached is not None:
            return cached

    data = _request_json(BOOTSTRAP_URL)

    cache.set(key, data)

    return data


def get_fixtures(force_refresh=False):

    key = "fixtures"

    if not force_refresh:
        cached = cache.get(key)

        if cached is not None:
            return cached

    data = _request_json(FIXTURES_URL)

    cache.set(key, data)

    return data


def get_players():

    return get_bootstrap()["elements"]


def get_teams():

    return get_bootstrap()["teams"]


def get_element_types():

    return get_bootstrap()["element_types"]


def get_events():

    return get_bootstrap()["events"]


def get_next_gameweek():

    events = get_events()

    unfinished = [
        event
        for event in events
        if not event.get("finished", False)
    ]

    if unfinished:
        return min(
            event["id"]
            for event in unfinished
        )

    return max(
        event["id"]
        for event in events
    )


def get_last_finished_gameweek():

    events = get_events()

    finished = [
        event["id"]
        for event in events
        if event.get("finished", False)
    ]

    if not finished:
        return None

    return max(finished)


def get_entry(entry_id: int):

    return _request_json(
        ENTRY_URL.format(
            entry_id=entry_id,
        )
    )


def get_entry_picks(
    entry_id: int,
    event: int,
):

    return _request_json(
        PICKS_URL.format(
            entry_id=entry_id,
            event=event,
        )
    )


def get_latest_public_squad(entry_id: int):

    gameweek = get_last_finished_gameweek()

    if gameweek is None:
        raise ValueError(
            "No completed Gameweek is available "
            "for public squad import yet."
        )

    data = get_entry_picks(
        entry_id,
        gameweek,
    )

    player_ids = [
        pick["element"]
        for pick in data["picks"]
    ]

    return {
        "gameweek": gameweek,
        "player_ids": player_ids,
        "entry_history": data.get(
            "entry_history",
            {},
        ),
    }