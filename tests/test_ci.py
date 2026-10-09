"""CI: на Gitea задачи идут на раннер ПК (метка windows), на GitHub - windows-latest;
шаги одинаковые."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def test_gitea_runs_on_pc_runner_and_github_on_hosted():
    gitea, github = read(".gitea", "workflows", "ci.yml"), read(".github", "workflows", "ci.yml")
    assert "runs-on: windows\n" in gitea and "uses: actions/setup-python" not in gitea
    assert "runs-on: windows-latest" in github
    for step in ("python -m pytest -q", "python main.py --selftest selftest.txt", "OPENBLAS_NUM_THREADS"):
        assert step in gitea and step in github
