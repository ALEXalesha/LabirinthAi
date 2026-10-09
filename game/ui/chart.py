"""Кривая обучения: шаги / кратчайший путь по ходу обучения, рядом - классические решатели."""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from aikit import theme


class CurveChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme = "dark"
        self.points = []          # [(x, y)]
        self.refs = []            # [(подпись, y, ключ цвета)]
        self.x_label = ""
        self.net_label = ""
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self):
        return QSize(640, 300)

    def set_data(self, points, refs, x_label, net_label, theme_name):
        self.points = list(points)
        self.refs = list(refs)
        self.x_label = x_label
        self.net_label = net_label
        self.theme = theme_name
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = theme.palette(self.theme)
        f = QFont(self.font())
        f.setPixelSize(12)
        p.setFont(f)
        area = QRectF(52, 14, self.width() - 200, self.height() - 50)
        if not self.points or area.width() < 50:
            p.end()
            return
        xs = [x for x, _ in self.points]
        ys = [y for _, y in self.points] + [y for _, y, _ in self.refs]
        x_max = max(xs) or 1
        y_max = min(max(ys) * 1.08, 12.0)
        y_min = 1.0

        def pt(x, y):
            y = max(y_min, min(y_max, y))
            return QPointF(area.left() + area.width() * x / x_max,
                           area.bottom() - area.height() * (y - y_min) / (y_max - y_min))

        grid = QPen(QColor(pal["grid"]))
        p.setPen(grid)
        step = 1 if y_max - y_min <= 6 else 2
        y = y_min
        while y <= y_max + 1e-9:
            a = pt(0, y)
            p.drawLine(QPointF(area.left(), a.y()), QPointF(area.right(), a.y()))
            p.setPen(QColor(pal["muted"]))
            p.drawText(QRectF(0, a.y() - 8, 46, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{y:g}×")
            p.setPen(grid)
            y += step
        p.setPen(QColor(pal["muted"]))
        p.drawText(QRectF(area.left(), area.bottom() + 6, area.width(), 16), Qt.AlignmentFlag.AlignLeft,
                   "0")
        p.drawText(QRectF(area.left(), area.bottom() + 6, area.width(), 16), Qt.AlignmentFlag.AlignRight,
                   f"{x_max:,}".replace(",", " "))
        p.drawText(QRectF(area.left(), area.bottom() + 20, area.width(), 16), Qt.AlignmentFlag.AlignHCenter,
                   self.x_label)
        for label, y, key in self.refs:
            a = pt(0, y)
            pen = QPen(QColor(pal.get(key, pal["muted"])), 1.5, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(area.left(), a.y()), QPointF(area.right(), a.y()))
            p.drawText(QRectF(area.right() + 8, a.y() - 8, 140, 16), Qt.AlignmentFlag.AlignVCenter,
                       f"{label} {y:.2f}×")
        path = QPainterPath(pt(*self.points[0]))
        for x, y in self.points[1:]:
            path.lineTo(pt(x, y))
        pen = QPen(QColor(pal["good"]), 2.5)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        last = pt(*self.points[-1])
        p.setBrush(QColor(pal["good"]))
        p.drawEllipse(last, 4, 4)
        p.setPen(QColor(pal["good"]))
        p.drawText(QRectF(area.right() + 8, last.y() - 8, 140, 16), Qt.AlignmentFlag.AlignVCenter,
                   f"{self.net_label} {self.points[-1][1]:.2f}×")
        p.end()
