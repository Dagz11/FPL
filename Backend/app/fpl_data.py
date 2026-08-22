import requests


BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def get_fpl_data():
    response = requests.get(BOOTSTRAP_URL, timeout=20)
    response.raise_for_status()

    return response.json()


def get_players():
    data = get_fpl_data()

    return data["elements"]
