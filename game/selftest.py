"""Самопроверка без экрана: python main.py --selftest итог.txt.

Окно создаётся offscreen, данные игрока - во временной папке. Проверяется: генераторы
(законы на случайных лабиринтах), решатели, окно - проход человека ботом через окно,
гонка и «Смотреть ИИ» до конца.
"""
import os
import tempfile
import time
import traceback


def selftest(out_path):
    lines = []
    code = 0
    try:
        from aikit import offscreen
        offscreen.setup(force=True)
        home = tempfile.mkdtemp(prefix="labirinthai-selftest-")
        os.environ["LABIRINTHAI_HOME"] = home
        from PySide6.QtWidgets import QApplication

        import aikit
        from aikit.host import make_window
        from game import APP_NAME, GAME_INFO, create_widget
        from game.ai.agents import RightHand
        from game.ai.net import STRONG, load_model
        from game.core.fog import play
        from game.core.generators import KINDS, generate
        from game.version import VERSION

        lines.append(f"LabirinthAi {VERSION} самопроверка (aikit {aikit.__version__})")
        t0 = time.time()
        for i in range(120):
            m = generate(KINDS[i % len(KINDS)], 3 + i % 15, 4 + i % 11, i)
            if not (m.symmetric() and m.connected() and m.passages() == m.h * m.w - 1):
                raise RuntimeError(f"лабиринт {i} нарушает законы")
            if not play(RightHand(), m).done:
                raise RuntimeError(f"правая рука не прошла лабиринт {i}")
        lines.append(f"генераторы и правая рука: 120 лабиринтов, {time.time() - t0:.2f} с")
        lines.append("сеть: " + ("загружена" if load_model(STRONG) is not None else "нет (сильный - Оптимист)"))

        app = QApplication.instance() or QApplication([])
        window, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=home)
        window.show()
        g = window.game
        for mode in ("solo", "race", "watch"):
            t0 = time.time()
            g.start(mode, seed=5)
            bot = RightHand()
            if mode != "watch":
                bot.reset(g.session.human.memory)
            end = time.time() + 60
            while not g.session.over:
                if time.time() > end:
                    raise RuntimeError(f"{mode}: не кончилось за 60 с")
                if mode != "watch":
                    g.on_key(bot.act(g.session.human.memory))
                if mode != "solo":
                    g.on_ai_tick()
                app.processEvents()
            image = window.grab().toImage()
            if image.isNull() or image.width() < 300:
                raise RuntimeError("окно не нарисовалось")
            lines.append(f"{mode}: «{g.play.status.text() if mode != 'watch' else 'все дошли'}», "
                         f"кадр {image.width()}x{image.height()}, {time.time() - t0:.2f} с")
        t0 = time.time()
        window.close()
        app.processEvents()
        lines.append(f"окно закрыто за {time.time() - t0:.2f} с")
        lines.append("ИТОГ: работает")
    except Exception:  # noqa: BLE001 - любая ошибка должна попасть в файл
        lines.append("ИТОГ: ошибка")
        lines.append(traceback.format_exc())
        code = 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return code
