"""Общие виджеты: выпадающий список со своей стрелкой, подпись с переносом, кнопки.

Две ловушки Qt, на которые уже наступали:

- Подпись с переносом строк нельзя ограничивать по высоте (setFixedHeight/Maximum): на
  настоящем экране текст шире, чем в offscreen, переносится на лишнюю строку и обрезается
  (TicTacToeAi 1.0.0). Только setMinimumHeight. label(..., wrap=True) так и делает.
- У скруглённого QComboBox кнопка ::drop-down без стиля рисуется квадратом Fusion поверх
  скругления (AiCar 2.0.0). В теме ::drop-down прозрачный, ::down-arrow пустой, а стрелку
  рисует Combo.paintEvent: треугольник из рамок QSS Qt не умеет, выходит прямоугольник.
"""
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPolygonF
from PySide6.QtWidgets import QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget


class Combo(QComboBox):
    """QComboBox со стрелкой, нарисованной кодом в цвет текста темы."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.arrow_color = QColor("#e6e8ec")

    def set_arrow_color(self, color):
        self.arrow_color = QColor(color)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.arrow_color)
        cx = self.width() - 16
        cy = self.height() / 2 + 1
        p.drawPolygon(QPolygonF([QPointF(cx - 5, cy - 3), QPointF(cx + 5, cy - 3), QPointF(cx, cy + 3)]))
        p.end()


def label(text="", name=None, wrap=False):
    """Подпись. С переносом - без ограничения высоты сверху (см. начало файла)."""
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def card():
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def button(text, name=None, slot=None):
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


def segmented(options, slot):
    """Ряд кнопок-переключателей: {значение: подпись}. Возвращает (виджет, {значение: кнопка})."""
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    group = QButtonGroup(box)
    group.setExclusive(True)
    buttons = {}
    for value, text in options.items():
        b = button(text, "seg")
        b.setCheckable(True)
        b.clicked.connect(lambda _=False, v=value: slot(v))
        group.addButton(b)
        row.addWidget(b)
        buttons[value] = b
    return box, buttons


def restyle(widget, object_name):
    """Сменить objectName и перечитать стиль - так метка меняет цвет по QSS."""
    widget.setObjectName(object_name)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
