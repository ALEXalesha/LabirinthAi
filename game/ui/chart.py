"""Кривые обучения: шаги / кратчайший путь по ходу обучения, рядом - классические решатели."""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from aikit import theme


class CurveChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme = "dark"
        self.series = []          # [(подпись, [(x, y)], ключ цвета)]
        self.refs = []            # [(подпись, y, ключ цвета)]
        self.x_label = ""
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self):
        return QSize(640, 320)

    def set_data(self, series, refs, x_label, theme_name):
        self.series = [(lab, list(pts), key) for lab, pts, key in series if pts]
        self.refs = list(refs)
        self.x_label = x_label
        self.theme = theme_name
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = theme.palette(self.theme)
        f = QFont(self.font())
        f.setPixelSize(12)
        p.setFont(f)
        area = QRectF(52, 14, self.width() - 230, self.height() - 50)
        if not self.series or area.width() < 50:
            p.end()
            return
        xs = [x for _, pts, _ in self.series for x, _ in pts]
        ys = [y for _, pts, _ in self.series for _, y in pts] + [y for _, y, _ in self.refs]
        x_max = max(xs) or 1
        y_max = min(max(ys) * 1.08, 12.0)
        y_min = 1.0

        def pt(x, y):
            y = max(y_min, min(y_max, y))
            return QPointF(area.left() + area.width() * x / x_max,
                           area.bottom() - area.height() * (y - y_min) / (y_max - y_min))

        grid = QPen(QColor(pal["grid"]))
        step = 1 if y_max - y_min <= 6 else 2
        y = y_min
        while y <= y_max + 1e-9:
            a = pt(0, y)
            p.setPen(grid)
            p.drawLine(QPointF(area.left(), a.y()), QPointF(area.right(), a.y()))
            p.setPen(QColor(pal["muted"]))
            p.drawText(QRectF(0, a.y() - 8, 46, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{y:g}×")
            y += step
        p.setPen(QColor(pal["muted"]))
        p.drawText(QRectF(area.left(), area.bottom() + 6, area.width(), 16), Qt.AlignmentFlag.AlignLeft, "0")
        p.drawText(QRectF(area.left(), area.bottom() + 6, area.width(), 16), Qt.AlignmentFlag.AlignRight,
                   f"{x_max:,}".replace(",", " "))
        p.drawText(QRectF(area.left(), area.bottom() + 20, area.width(), 16), Qt.AlignmentFlag.AlignHCenter,
                   self.x_label)
        labels = []                       # (y, текст, цвет) - справа, без наложения
        for label, y, key in self.refs:
            a = pt(0, y)
            color = QColor(pal.get(key, pal["muted"]))
            p.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(area.left(), a.y()), QPointF(area.right(), a.y()))
            labels.append((a.y(), f"{label} {y:.2f}×", color))
        for label, pts, key in self.series:
            color = QColor(pal.get(key, pal["good"]))
            path = QPainterPath(pt(*pts[0]))
            for x, y in pts[1:]:
                path.lineTo(pt(x, y))
            p.setPen(QPen(color, 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            last = pt(*pts[-1])
            p.setBrush(color)
            p.drawEllipse(last, 4, 4)
            labels.append((last.y(), f"{label} {pts[-1][1]:.2f}×", color))
        labels.sort(key=lambda t: t[0])
        prev = -1e9
        for y, text, color in labels:
            y = max(y, prev + 16)
            prev = y
            p.setPen(color)
            p.drawText(QRectF(area.right() + 8, y - 8, 220, 16), Qt.AlignmentFlag.AlignVCenter, text)
        p.end()
