"""LabirinthAi: лабиринты в тумане и ИИ, который научился их проходить.

Вход игры по контракту PythonAiGames: GAME_INFO и create_widget(ctx). Qt загружается
только в create_widget - лабиринты, ИИ и обучение работают без него.
"""
from game.version import VERSION

GAME_INFO = {
    "id": "labirinth",
    "title": {"ru": "Лабиринт", "en": "Labyrinth"},
    "summary": {"ru": "Лабиринт в тумане: пройди сам, наперегонки с ИИ или смотри, как ИИ учится",
                "en": "Mazes in the fog: solve them yourself, race the AI or watch it learn"},
    "preview": "docs/preview.png",
    "version": VERSION,
}

APP_NAME = "LabirinthAi"


def create_widget(ctx):
    """ctx: aikit.Context. Виджет с меню, полем и панелями."""
    from game.ui.widget import GameWidget
    return GameWidget(ctx)
