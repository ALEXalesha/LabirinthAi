"""Context и окно отдельного запуска. Отдельно от aikit/__init__, чтобы обучение
(чистый numpy, aikit.nn) не тянуло за собой Qt."""
import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from aikit import LANGS, data_root


class Context(QObject):
    """Что игра получает от оболочки. changed(str) - 'theme' или 'lang': игра
    перерисовывается на лету. Игра пишет файлы только в data_dir."""

    changed = Signal(str)

    def __init__(self, data_dir, theme="dark", lang="ru", volume=70, host_settings=None):
        super().__init__()
        from aikit.sound import Sounds
        from aikit.stats import Stats
        self.data_dir = str(data_dir)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self._theme = theme
        self._lang = lang if lang in LANGS else "ru"
        self.host_settings = host_settings
        self.stats = Stats(os.path.join(self.data_dir, "stats.json"))
        self.sound = Sounds(os.path.join(self.data_dir, "sounds"), volume)

    @property
    def theme(self):
        return self._theme

    @property
    def lang(self):
        return self._lang

    def palette(self):
        from aikit import theme
        return theme.palette(self._theme)

    def file(self, *parts):
        path = os.path.join(self.data_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def set_theme(self, name):
        from aikit import theme
        if name not in theme.THEMES or name == self._theme:
            return
        self._theme = name
        self._remember("theme", name)
        self.changed.emit("theme")

    def set_lang(self, lang):
        if lang not in LANGS or lang == self._lang:
            return
        self._lang = lang
        self._remember("lang", lang)
        self.changed.emit("lang")

    def set_volume(self, volume):
        self.sound.set_volume(volume)
        self._remember("volume", volume)

    def _remember(self, key, value):
        if self.host_settings is not None:
            try:
                self.host_settings.set(key, value)
            except ValueError:
                pass


def host_schema():
    from aikit import settings, theme
    return {"theme": ("dark", settings.choice(*theme.THEMES)),
            "lang": ("ru", settings.choice(*LANGS)),
            "volume": (70, settings.int_range(0, 100))}


WINDOW_OPTS = {"width": 1180, "height": 780, "minWidth": 900, "minHeight": 700}


def make_window(game_info, create_widget, app_name, root=None):
    """Окно отдельного запуска: заголовок, тема, место окна, виджет игры в центре.
    Возвращает (окно, ctx). QApplication должно уже быть."""
    from PySide6.QtWidgets import QMainWindow

    from aikit import settings, theme
    from aikit.window_state import WindowMemory

    root = root or data_root(app_name)
    host = settings.Settings(host_schema(), os.path.join(root, "host.json"))
    ctx = Context(os.path.join(root, game_info["id"]), host["theme"], host["lang"], host["volume"], host)

    class HostWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.ctx = ctx
            self.game = create_widget(ctx)
            self.setCentralWidget(self.game)
            self.apply()
            ctx.changed.connect(self.apply)
            need = self.game.minimumSizeHint()
            opts = {**WINDOW_OPTS, "minWidth": max(WINDOW_OPTS["minWidth"], need.width()),
                    "minHeight": max(WINDOW_OPTS["minHeight"], need.height())}
            self.window_opts = opts
            self.setMinimumSize(opts["minWidth"], opts["minHeight"])
            self.memory = WindowMemory(self, os.path.join(root, "window.json"), opts)

        def apply(self, *_):
            title = game_info["title"].get(ctx.lang, game_info["title"]["en"])
            self.setWindowTitle(f"{title} {game_info.get('version', '')}".strip())
            self.setStyleSheet(theme.qss(ctx.theme))

        def moveEvent(self, event):
            super().moveEvent(event)
            if hasattr(self, "memory"):
                self.memory.track()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if hasattr(self, "memory"):
                self.memory.track()

        def closeEvent(self, event):
            shutdown = getattr(self.game, "shutdown", None)
            if shutdown is not None:
                shutdown()
            self.memory.save()
            super().closeEvent(event)

    return HostWindow(), ctx


def run_standalone(game_info, create_widget, app_name, argv=None, icon=None):
    """python main.py: своё окно для одной игры."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName(app_name)
    if icon is not None:
        app.setWindowIcon(icon)
    window, _ = make_window(game_info, create_widget, app_name)
    if icon is not None:
        window.setWindowIcon(icon)
    window.memory.show()
    return app.exec()
