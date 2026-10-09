"""Окно без экрана (Qt offscreen): проход ботом через окно, гонка, «Смотреть ИИ», клавиши,
темы, языки, статистика, остановка за 2 секунды. Ожидания - с тайм-аутом; клавиши -
собранными событиями, настоящая клавиатура и курсор не трогаются."""
import json
import os
import time

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from aikit.widgets import Combo
from game import APP_NAME, GAME_INFO, create_widget
from game.ai import agents as A
from game.core.fog import Memory
from game.ui import widget as W
from game.ui.i18n import EN, RU

QWIDGETSIZE_MAX = (1 << 24) - 1


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, tmp_path):
    from aikit.host import make_window
    w, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=str(tmp_path / "home"))
    w.resize(1280, 800)
    w.show()
    yield w
    w.game.shutdown()
    w.close()


def solve_by_bot(app, g, limit=60.0):
    """Бот (правая рука) играет за человека через окно - по той же памяти, что видна в окне."""
    bot = A.RightHand()
    bot.reset(g.session.human.memory)
    end = time.time() + limit
    while not g.session.over:
        assert time.time() < end, "проход не кончился вовремя"
        g.on_key(bot.act(g.session.human.memory))
        app.processEvents()


def wait(app, cond, limit=30.0):
    end = time.time() + limit
    while not cond() and time.time() < end:
        app.processEvents()
        time.sleep(0.002)
    return cond()


def test_translations_complete():
    assert set(RU) == set(EN)


def test_solo_to_the_exit_records_history(app, win):
    g = win.game
    g.change("size", "s")
    g.start("solo", seed=3)
    assert g.current_page() == "play" and not g.play.boards[1].isVisible()
    solve_by_bot(app, g)
    assert g.play.status.text() == RU["solo_done"]
    hist = g.history()
    assert len(hist) == 1 and hist[0]["mode"] == "solo" and hist[0]["done"]
    assert hist[0]["steps"] >= hist[0]["shortest"] > 0
    assert not g.clock.isActive()


def test_race_records_win_or_loss(app, win):
    g = win.game
    g.change("size", "s")
    g.change("level", "novice")
    g.change("ai_speed", 30)
    g.start("race", seed=4)
    assert g.play.boards[1].isVisible()
    assert not g.ai_timer.isActive() or g.session.started is None
    g.on_ai_tick()
    assert g.session.runners[1].memory.steps == 0, "ИИ не стартует до первого хода человека"
    bot = A.RightHand()
    bot.reset(g.session.human.memory)
    g.on_key(bot.act(g.session.human.memory))
    assert wait(app, lambda: g.session.over, 30)
    e = g.ctx.stats.entry("race/novice")
    assert e["games"] == 1 and e["wins"] + e["losses"] == 1
    assert g.history()[-1]["result"] in ("win", "loss")


def test_watch_all_agents_finish_with_places(app, win):
    g = win.game
    g.change("size", "s")
    for i, name in enumerate(("right_hand", "tremaux", "optimist", "astar")):
        g.change(f"watch{i}", name)
    g.change("watch_speed", 60)
    g.start("watch", seed=7)
    assert g.current_page() == "watch"
    assert wait(app, lambda: g.session.over, 60)
    places = sorted(r.finish_order for r in g.session.runners)
    assert places[0] >= 1 and all(r.done for r in g.session.runners)
    astar = g.session.runners[3]
    assert astar.memory.steps == g.session.maze.shortest()
    assert "A*" in g.watch.boards[3].title


def test_ai_gets_only_memory(app, win, monkeypatch):
    seen = []
    orig = A.Tremaux.act

    def spy(self, mem):
        seen.append(mem)
        return orig(self, mem)

    monkeypatch.setattr(A.Tremaux, "act", spy)
    g = win.game
    g.change("size", "s")
    g.change("watch0", "tremaux")
    g.start("watch", seed=2)
    for _ in range(10):
        g.on_ai_tick()
    assert seen and all(isinstance(m, Memory) for m in seen)
    mem = seen[-1]
    assert np.all(mem.cells[~mem.known] == 0)


def test_keys_move_the_player(app, win):
    g = win.game
    g.change("size", "s")
    g.start("solo", seed=5)
    win.activateWindow()
    assert QTest.qWaitForWindowActive(win, 2000)
    mem = g.session.human.memory
    keys = {0: Qt.Key.Key_Up, 1: Qt.Key.Key_Right, 2: Qt.Key.Key_Down, 3: Qt.Key.Key_Left}
    d = mem.moves()[0]
    QTest.keyClick(g, keys[d])
    app.processEvents()
    assert mem.steps == 1
    back = (d + 2) % 4
    QTest.keyClick(g, {0: Qt.Key.Key_W, 1: Qt.Key.Key_D, 2: Qt.Key.Key_S, 3: Qt.Key.Key_A}[back])
    app.processEvents()
    assert mem.steps == 2 and mem.pos == mem.start
    wall = [x for x in range(4) if x not in mem.moves()][0]
    QTest.keyClick(g, keys[wall])
    assert mem.steps == 2, "шаг в стену не считается"


def test_thoughts_show_arrows_or_nothing_without_net(app, win, monkeypatch):
    g = win.game
    g.change("size", "s")
    g.change("show_thoughts", True)
    g.start("solo", seed=6)
    th = g.thoughts_for(g.session.human, human=True)
    if W.load_model(W.STRONG) is None:
        assert th["probs"] is None
    else:
        p = th["probs"]
        assert abs(p.sum() - 1) < 1e-5
        assert all(p[d] == 0 for d in range(4) if d not in g.session.human.memory.moves())
    g.change("show_thoughts", False)
    assert g.thoughts_for(g.session.human, human=True) is None


def test_theme_and_language_on_the_fly(app, win):
    g = win.game
    g.start("solo", seed=5)
    win.ctx.set_theme("light")
    assert g.play.boards[0].theme == "light"
    assert not g.play.boards[0].grab().toImage().isNull()
    win.ctx.set_lang("en")
    assert g.play.again_button.text() == EN["again"]
    assert g.play.status.text() == EN["solo_status"]
    win.ctx.set_lang("ru")
    win.ctx.set_theme("dark")
    assert g.play.again_button.text() == RU["again"]


def test_pages_render_in_both_themes(app, win):
    g = win.game
    for theme_name in ("dark", "light"):
        win.ctx.set_theme(theme_name)
        for page in ("menu", "stats", "training", "settings"):
            g.show_page(page)
            app.processEvents()
            assert not win.grab().toImage().isNull()
    win.ctx.set_theme("dark")


def test_wrapped_labels_have_no_height_cap_and_combos_draw_arrow(app, win):
    g = win.game
    g.start("solo", seed=5)
    for w in g.findChildren(QLabel):
        if w.wordWrap():
            assert w.maximumHeight() == QWIDGETSIZE_MAX, w.text()
    assert len(g.findChildren(Combo)) >= 7


def test_shutdown_within_two_seconds_while_watching(app, win):
    g = win.game
    g.change("size", "xl")
    g.start("watch", seed=12)
    assert g.ai_timer.isActive()
    t0 = time.time()
    assert g.shutdown()
    assert time.time() - t0 < 2.0
    assert not g.ai_timer.isActive() and not g.clock.isActive()


def test_menu_page_stops_timers(app, win):
    g = win.game
    g.start("watch", seed=1)
    g.show_page("menu")
    assert not g.ai_timer.isActive()


def test_pause_in_watch(app, win):
    g = win.game
    g.start("watch", seed=1)
    g.toggle_pause()
    assert not g.ai_timer.isActive()
    steps = [r.memory.steps for r in g.session.runners]
    g.on_ai_tick()
    assert [r.memory.steps for r in g.session.runners] == steps
    g.toggle_pause()
    assert g.ai_timer.isActive()


def test_same_maze_again(app, win):
    g = win.game
    g.start("solo", seed=9)
    cells = g.session.maze.cells.copy()
    g.on_key(g.session.human.memory.moves()[0])
    g.restart_same()
    assert np.array_equal(g.session.maze.cells, cells) and g.session.human.memory.steps == 0


def test_settings_persist(app, tmp_path):
    from aikit.host import make_window
    root = str(tmp_path / "home3")
    w, _ = make_window(GAME_INFO, create_widget, APP_NAME, root=root)
    w.game.change("size", "l")
    w.game.change("watch2", "astar")
    w.close()
    w2, _ = make_window(GAME_INFO, create_widget, APP_NAME, root=root)
    assert w2.game.settings["size"] == "l" and w2.game.settings["watch2"] == "astar"
    assert w2.game.menu.size_buttons["l"].isChecked()
    w2.close()


def test_window_remembers_size(app, tmp_path):
    from aikit.host import make_window
    root = str(tmp_path / "home2")
    w, _ = make_window(GAME_INFO, create_widget, APP_NAME, root=root)
    w.resize(1111, 777)
    w.show()
    app.processEvents()
    w.close()
    app.processEvents()
    data = json.load(open(os.path.join(root, "window.json"), encoding="utf-8"))
    assert 1111 in data.values() or data.get("width") == 1111


def test_broken_history_file_is_empty_history(app, win):
    g = win.game
    with open(g.ctx.file("history.json"), "w", encoding="utf-8") as f:
        f.write("{мусор")
    assert g.history() == []
    g.show_page("stats")
    assert g.stats.empty.isVisible() or g.stats.solo.rowCount() == 0
