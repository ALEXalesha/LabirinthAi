import json
import os
import time

from aikit import pick_data_root


def test_data_root_rules(tmp_path):
    env = {"APPDATA": r"C:\Users\u\AppData\Roaming"}
    assert pick_data_root("Game", False, "x.exe", env) == os.path.join(env["APPDATA"], "Game")
    assert pick_data_root("Game", False, "x.exe", {**env, "GAME_HOME": "D:/t"}) == "D:/t"
    exe = tmp_path / "Game.exe"
    assert pick_data_root("Game", True, str(exe), env) == os.path.join(env["APPDATA"], "Game")
    (tmp_path / "portable.txt").write_text("", encoding="utf-8")
    assert pick_data_root("Game", True, str(exe), env) == os.path.join(str(tmp_path), "saves")


def test_context_emits_changes_and_remembers_them(qapp, tmp_path):
    from aikit import settings
    from aikit.host import Context, host_schema
    host = settings.Settings(host_schema(), tmp_path / "host.json")
    ctx = Context(tmp_path / "game", host_settings=host)
    seen = []
    ctx.changed.connect(seen.append)
    ctx.set_theme("light")
    ctx.set_theme("light")          # то же самое - без сигнала
    ctx.set_lang("en")
    ctx.set_lang("de")              # нет такого языка - пропуск
    assert seen == ["theme", "lang"]
    saved = json.loads((tmp_path / "host.json").read_text(encoding="utf-8"))
    assert (saved["theme"], saved["lang"]) == ("light", "en")
    assert ctx.file("a", "b.json").startswith(str(tmp_path / "game"))


def test_standalone_window_calls_shutdown_and_saves_place(qapp, tmp_path):
    from PySide6.QtWidgets import QLabel

    from aikit.host import make_window

    class Game(QLabel):
        stopped = False

        def shutdown(self):
            Game.stopped = True

    info = {"id": "demo", "title": {"ru": "Демо", "en": "Demo"}, "version": "1.2.3"}
    window, ctx = make_window(info, lambda c: Game("игра"), "DemoApp", root=str(tmp_path))
    window.show()
    qapp.processEvents()
    assert window.windowTitle() == "Демо 1.2.3"
    ctx.set_lang("en")
    assert window.windowTitle() == "Demo 1.2.3"
    started = time.time()
    window.close()
    assert Game.stopped and time.time() - started < 2
    assert (tmp_path / "window.json").exists()
    assert os.path.isdir(tmp_path / "demo")
