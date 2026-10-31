from .config import GAMES_DIR


def load_games() -> list[dict]:
    games: list[dict] = []
    if not GAMES_DIR.exists():
        return games
    for path in sorted(GAMES_DIR.glob("*/profile.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        data.setdefault("id", path.parent.name)
        data.setdefault("name", data["id"])
        data.setdefault("genre", "游戏")
        data.setdefault("aliases", [])
        data.setdefault("window_keywords", [])
        data.setdefault("quick_questions", [])
        data.setdefault("terms", [])
        data.setdefault("intro", "")
        data["id"] = str(data["id"])
        games.append(data)
    return games


def get_game(game_id: str | None, games: list[dict] | None = None) -> dict | None:
    if not game_id:
        return None
    for game in games if games is not None else load_games():
        if game.get("id") == game_id:
            return game
    return None


def public_game(game: dict) -> dict:
    keys = ("id", "name", "genre", "intro", "aliases", "quick_questions", "terms", "voice")
    return {key: game.get(key) for key in keys}
