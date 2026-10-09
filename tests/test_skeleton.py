import game


def test_game_info_follows_contract():
    info = game.GAME_INFO
    assert info["id"] == "labirinth"
    assert set(info["title"]) == {"ru", "en"} and set(info["summary"]) == {"ru", "en"}
    assert info["version"] == game.VERSION
    assert callable(game.create_widget)
