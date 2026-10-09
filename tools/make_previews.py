"""Кадры для README: окно без экрана в нескольких состояниях, обе темы.

    python tools/make_previews.py [папка]     # по умолчанию docs/
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aikit import offscreen  # noqa: E402

offscreen.setup(force=True)
os.environ["LABIRINTHAI_HOME"] = tempfile.mkdtemp(prefix="labirinthai-previews-")


def settle(app, ms=200):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def walk(g, steps):
    """Бот (правая рука) делает steps шагов за человека."""
    from game.ai.agents import RightHand
    bot = RightHand()
    bot.reset(g.session.human.memory)
    for _ in range(steps):
        if g.session.over:
            break
        g.on_key(bot.act(g.session.human.memory))


def main(out):
    from PySide6.QtWidgets import QApplication

    from aikit.host import make_window
    from game import APP_NAME, GAME_INFO, create_widget

    os.makedirs(out, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    window, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=os.environ["LABIRINTHAI_HOME"])
    window.resize(1280, 800)
    window.show()
    g = window.game
    shots = []
    for theme_name in ("dark", "light"):
        ctx.set_theme(theme_name)
        sfx = "" if theme_name == "dark" else "_light"
        g.show_page("menu")
        settle(app)
        shots.append((window.grab(), f"preview_menu{sfx}.png"))
        g.change("size", "l")
        g.change("show_thoughts", True)
        g.start("solo", seed=21)
        walk(g, 40)
        settle(app)
        shots.append((window.grab(), f"preview_solo{sfx}.png"))
        g.change("level", "strong")
        g.start("race", seed=22)
        walk(g, 1)
        for _ in range(35):
            g.on_ai_tick()
            walk(g, 1)
        settle(app)
        shots.append((window.grab(), f"preview_race{sfx}.png"))
        g.change("size", "m")
        g.start("watch", seed=23)
        for _ in range(60):
            g.on_ai_tick()
        settle(app)
        shots.append((window.grab(), f"preview_watch{sfx}.png"))
        g.show_page("training")
        settle(app)
        shots.append((window.grab(), f"preview_training{sfx}.png"))
    # картинка для списка игр в ИИ-Игротеке: 16:9
    ctx.set_theme("dark")
    window.resize(1280, 720)
    g.change("size", "m")
    g.start("watch", seed=23)
    for _ in range(60):
        g.on_ai_tick()
    settle(app)
    shots.append((window.grab(), "preview.png"))
    g.change("show_thoughts", False)
    for pix, name in shots:
        pix.save(os.path.join(out, name))
        print(name)
    g.shutdown()
    window.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs"))
