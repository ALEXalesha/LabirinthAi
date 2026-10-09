"""Личное не публикуется: почта автора, адрес домашнего Gitea, личные пути - ни в одном
файле репозитория. Почта и адрес берутся из настроек git, а не пишутся строкой: иначе
проверка сама стала бы утечкой."""
import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    if m:
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
