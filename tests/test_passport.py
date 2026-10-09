"""Паспорт модели: первые лабиринты каждой проверки пересчитываются с теми же зёрнами и
сверяются с записанными шагами. Нет паспорта (сеть не обучена) - проверка пропускается."""
import pytest

from game.ai.net import load_passport
from game.train import passport as P

PASS = load_passport()
pytestmark = pytest.mark.skipif(PASS is None, reason="паспорта ещё нет")


def test_passport_has_every_check_and_agent():
    assert PASS["games_per_check"] >= 1000
    assert [(c["size"], c["kind"]) for c in PASS["checks"]] == P.CHECKS
    for c in PASS["checks"]:
        assert set(PASS["agents"]) <= set(c["agents"])


@pytest.mark.slow
@pytest.mark.parametrize("ci", range(len(P.CHECKS)))
def test_sample_recomputes_exactly(ci):
    check = PASS["checks"][ci]
    for agent, rec in check["agents"].items():
        for i, steps, done in rec["sample"][:4]:
            assert P.run_one(agent, check["size"], check["kind"], i) == (steps, done), (agent, i)


def test_check_mazes_are_not_training_or_curve_mazes():
    from game.train.ppo import VALID_SEED
    assert P.SEED > VALID_SEED + 10 ** 6 and P.SEED > 2 ** 30
