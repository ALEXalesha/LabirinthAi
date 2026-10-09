"""Звуки синтезом на numpy: ход, взятие, победа, поражение, ничья.

Волна считается один раз, пишется в WAV в папке данных и играется через QSoundEffect.
Нет звуковой карты или QtMultimedia - игра молчит, но работает.
"""
import io
import os
import wave
from pathlib import Path

import numpy as np

RATE = 22050
NAMES = ("move", "capture", "win", "loss", "draw")
DURATIONS = {"move": 0.09, "capture": 0.11, "win": 0.62, "loss": 0.62, "draw": 0.42}


def _envelope(n, attack=0.004, release=0.03):
    """Нарастание и затухание по краям - без щелчков в начале и в конце."""
    env = np.ones(n)
    a = max(1, int(attack * RATE))
    r = max(1, int(release * RATE))
    env[:a] = np.linspace(0, 1, a)
    env[-r:] *= np.linspace(1, 0, r)
    return env


def _tone(freqs, length, decay=0.0):
    """Тон с плавно меняющейся частотой: freqs - начальная и конечная, в герцах."""
    n = round(length * RATE)
    f = np.linspace(freqs[0], freqs[1], n)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    wave_ = np.sin(phase) + 0.25 * np.sin(2 * phase)
    if decay:
        wave_ *= np.exp(-decay * np.arange(n) / RATE)
    return wave_


def _notes(freqs, total):
    """Несколько нот подряд, всего total секунд."""
    n = round(total * RATE)
    part = n // len(freqs)
    out = np.zeros(n)
    for i, f in enumerate(freqs):
        seg = _tone((f, f), part / RATE, decay=3.0) * _envelope(part, release=0.02)
        out[i * part:(i + 1) * part] = seg
    return out


def synth(name):
    length = DURATIONS[name]
    n = round(length * RATE)
    if name == "move":
        s = _tone((880, 660), length, decay=25)
    elif name == "capture":
        s = _tone((440, 520), length, decay=20)
    elif name == "win":
        s = _notes([523, 659, 784, 1047], length)       # до-ми-соль-до вверх
    elif name == "loss":
        s = _notes([392, 330, 262, 196], length)        # вниз
    elif name == "draw":
        s = _notes([440, 440], length)
    else:
        raise ValueError(name)
    s = s[:n] * _envelope(n)
    return (0.6 * s / np.abs(s).max()).astype(np.float32)


def to_wav(samples):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class Sounds:
    """Проигрыватель. volume - 0..100."""

    def __init__(self, folder, volume=70):
        self.volume = volume
        self.effects = {}
        folder = Path(folder)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            for name in NAMES:
                (folder / f"{name}.wav").write_bytes(to_wav(synth(name)))
        except OSError:
            return
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return      # тесты, самопроверка, кадры README: без экрана - и без звука в колонках
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QSoundEffect
        except ImportError:
            return
        for name in NAMES:
            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(str(folder / f"{name}.wav")))
            self.effects[name] = effect
        self.set_volume(volume)

    def set_volume(self, volume):
        self.volume = volume
        for effect in self.effects.values():
            effect.setVolume(volume / 100)

    def play(self, name):
        effect = self.effects.get(name)
        if effect is None or self.volume <= 0:
            return
        effect.stop()
        effect.play()
