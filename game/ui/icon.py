"""Иконка программы: маленький лабиринт с фишкой и выходом, нарисованный кодом."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap

from aikit import theme
from game.core.generators import generate
from game.core.maze import BIT


def render(size=256):
    pal = theme.palette("dark")
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0, 0, size, size), size * 0.2, size * 0.2)
    p.setClipPath(clip)
    p.fillRect(QRectF(0, 0, size, size), QColor(pal["surface"]))
    n = 5
    maze = generate("dfs", n, n, 11)
    m = size * 0.12
    cs = (size - 2 * m) / n
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(pal["good"]))
    p.drawRoundedRect(QRectF(m + (n - 1) * cs + cs * 0.2, m + (n - 1) * cs + cs * 0.2, cs * 0.6, cs * 0.6),
                      cs * 0.12, cs * 0.12)
    pen = QPen(QColor(pal["text"]), max(1.2, size * 0.035))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    for r in range(n):
        for c in range(n):
            x, y = m + c * cs, m + r * cs
            if r > 0 and not maze.cells[r, c] & BIT[0]:
                p.drawLine(QPointF(x, y), QPointF(x + cs, y))
            if c > 0 and not maze.cells[r, c] & BIT[3]:
                p.drawLine(QPointF(x, y), QPointF(x, y + cs))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(m, m, cs * n, cs * n))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(pal["accent"]))
    p.drawEllipse(QPointF(m + cs / 2, m + cs / 2), cs * 0.3, cs * 0.3)
    p.end()
    return img


def app_icon():
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(QPixmap.fromImage(render(s)))
    return icon
