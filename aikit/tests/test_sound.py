import io
import wave

import numpy as np
import pytest

from aikit import sound


@pytest.mark.parametrize("name", sound.NAMES)
def test_every_sound_has_its_length(name):
    s = sound.synth(name)
    assert len(s) == round(sound.DURATIONS[name] * sound.RATE)
    assert s.dtype == np.float32


@pytest.mark.parametrize("name", sound.NAMES)
def test_sounds_stay_in_range_and_have_no_clicks_at_the_edges(name):
    s = sound.synth(name)
    assert np.abs(s).max() <= 1.0
    assert np.abs(s).max() > 0.2                  # слышно
    edge = int(0.0005 * sound.RATE)          # первые и последние полмиллисекунды
    assert np.abs(s[:edge]).max() < 0.1
    assert np.abs(s[-edge:]).max() < 0.1


def test_moves_are_short_and_endings_are_longer():
    assert sound.DURATIONS["move"] < 0.2 and sound.DURATIONS["capture"] < 0.2
    for name in ("win", "loss", "draw"):
        assert sound.DURATIONS[name] >= 0.3


def crossings(part):
    return int(np.count_nonzero(np.diff(np.signbit(part))))


def test_win_goes_up_and_loss_goes_down():
    for name, up in (("win", True), ("loss", False)):
        s = sound.synth(name)
        q = len(s) // 4
        first, last = crossings(s[:q]), crossings(s[-q:])
        assert (last > first) == up, name


def test_move_and_capture_sound_different():
    assert crossings(sound.synth("move")) != crossings(sound.synth("capture"))


def test_wav_bytes_are_a_valid_file():
    data = sound.to_wav(sound.synth("win"))
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == sound.RATE
        assert w.getnframes() == len(sound.synth("win"))


def test_player_without_sound_device_does_not_fail(qapp, tmp_path):
    player = sound.Sounds(tmp_path / "snd", volume=50)
    for name in sound.NAMES:
        player.play(name)
    player.set_volume(0)
    player.play("win")
    assert sorted(p.name for p in (tmp_path / "snd").iterdir()) == sorted(f"{n}.wav" for n in sound.NAMES)
