"""aikit - общий набор игр с ИИ (контракт PythonAiGames/docs/contract.md).

Игра отдаёт GAME_INFO и create_widget(ctx); окно ей даёт оболочка или run_standalone.
Здесь - Context (тема, язык, настройки, статистика, звук, папка данных) и отдельный запуск.

Версия набора - в файле VERSION рядом. aikit не правится под одну игру: улучшения
переносятся во все копии и в PythonAiGames/aikit.
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def version():
    try:
        return (_HERE / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "0"


__version__ = version()

LANGS = ("ru", "en")


def frozen():
    return getattr(sys, "frozen", False)


def pick_data_root(app_name, frozen_, exe, env):
    """Папка данных программы по признакам запуска - отдельно от sys и os.environ, чтобы
    проверять. <APP>_HOME подменяет её целиком (тесты, самопроверка); у собранного exe с
    portable.txt рядом - папка saves рядом; иначе %APPDATA%\\<программа>."""
    var = app_name.upper() + "_HOME"
    if env.get(var):
        return env[var]
    if frozen_:
        beside = os.path.dirname(os.path.abspath(exe))
        if os.path.exists(os.path.join(beside, "portable.txt")):
            return os.path.join(beside, "saves")
    base = env.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, app_name)


def data_root(app_name):
    return pick_data_root(app_name, frozen(), sys.executable, os.environ)


def __getattr__(name):
    """Context, make_window, run_standalone - из aikit.host, по первому обращению: так
    import aikit.nn в обучении не загружает Qt."""
    if name in ("Context", "make_window", "run_standalone", "WINDOW_OPTS", "host_schema"):
        from aikit import host
        return getattr(host, name)
    raise AttributeError(name)
