"""Личное не публикуется: почта автора, адрес домашнего Gitea, личные пути - ни в одном
файле репозитория. Почта и адрес берутся из настроек git, а не пишутся строкой: иначе
проверка сама стала бы утечкой."""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_HOSTS = ("github.com", "gitlab.com", "codeberg.org")


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def tracked_files():
    out = git("ls-files")
    if not out:
        pytest.skip("нет git")
    return [f for f in out.splitlines() if not f.endswith((".npz", ".png", ".ico"))]


def test_no_private_strings_in_tracked_files():
    secrets = []
    email = git("config", "user.email")
    if email:
        secrets.append(re.escape(email))
    origin = git("remote", "get-url", "origin")
    m = re.match(r"^[a-z]+://([^/]+)/", origin)
    # домашний сервер - личное; github.com в клоне с GitHub - нет (на него ссылаются
    # значки README и скрипт выкладки), так же, как в DurakAi
    if m and not m.group(1).lower().endswith(PUBLIC_HOSTS):
        secrets.append(re.escape(m.group(1)))
    secrets.append(re.escape("Drive" + "\\" + "Alexey"))
    secrets.append(re.escape("Drive" + "/" + "Alexey"))
    pattern = re.compile("|".join(secrets), re.IGNORECASE)
    bad = []
    for f in tracked_files():
        path = os.path.join(ROOT, f)
        if not os.path.isfile(path):
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        if pattern.search(text):
            bad.append(f)
    assert not bad, bad


def test_public_origin_is_not_a_secret(monkeypatch):
    """В клоне с GitHub origin - github.com, а он законно есть в README и tools/:
    проверка не должна краснеть на CI GitHub."""
    real = git

    def fake(*args):
        if args[0] == "remote":
            return "https://github.com/someone/LabirinthAi.git"
        if args[0] == "config":
            return ""
        return real(*args)
    monkeypatch.setattr(sys.modules[__name__], "git", fake)
    test_no_private_strings_in_tracked_files()
