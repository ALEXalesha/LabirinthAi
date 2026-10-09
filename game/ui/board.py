"""Поле лабиринта: туман, память, след, выход, «Мысли ИИ». Рисуется своим кодом (QPainter).

Поле рисует только то, что знает участник: видимые и виденные клетки, их стены. Невиданное -
туман. reveal=True (конец прохода, читеры) - весь лабиринт.
"""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from aikit import theme
from game.core.maze import BIT, DC, DR

AGENT_COLORS = {"you": "accent", "ai": "warn", "right_hand": "muted", "tremaux": "accent2",
                "optimist": "select", "net_random": "bad", "net_early": "warn", "net_strong": "good",
                "bfs": "capture_hint", "astar": "capture_hint"}


def qcolor(hex_, alpha=255):
    c = QColor(hex_)
    c.setAlpha(alpha)
    return c


class MazeBoard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme = "dark"
        self.runner = None
        self.reveal = False
        self.thoughts = None
        self.title = ""
        self.subtitle = ""
        self.highlight = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(180, 180)

    def sizeHint(self):
        return QSize(560, 560)

    def set_theme(self, name):
        self.theme = name
        self.update()

    def show_runner(self, runner, reveal=False, thoughts=None, title="", subtitle="", highlight=False):
        self.runner = runner
        self.reveal = reveal
        self.thoughts = thoughts
        self.title = title
        self.subtitle = subtitle
        self.highlight = highlight
        self.update()

    # --- геометрия ---

    def layout_rect(self):
        top = 0
        if self.title:
            top += 24
        if self.subtitle:
            top += 20
        return QRectF(8, 8 + top, self.width() - 16, self.height() - 16 - top)

    def cell_geometry(self):
        mem = self.runner.memory
        area = self.layout_rect()
        cs = max(4.0, min(area.width() / mem.w, area.height() / mem.h))
        x0 = area.x() + (area.width() - cs * mem.w) / 2
        y0 = area.y() + (area.height() - cs * mem.h) / 2
        return x0, y0, cs

    def cell_rect(self, r, c):
        x0, y0, cs = self.cell_geometry()
        return QRectF(x0 + c * cs, y0 + r * cs, cs, cs)

    # --- рисование ---

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = theme.palette(self.theme)
        p.fillRect(self.rect(), QColor(pal["bg"]))
        y = 8
        if self.runner is None:
            if self.title:
                p.setPen(QColor(pal["muted"]))
                p.drawText(QRectF(8, 8, self.width() - 16, self.height() - 16), Qt.AlignmentFlag.AlignCenter,
                           "\n".join((self.title, self.subtitle)))
            p.end()
            return
        if self.title:
            f = QFont(self.font())
            f.setPixelSize(16)
            f.setBold(True)
            p.setFont(f)
            color = AGENT_COLORS.get(self.runner.name, "accent")
            p.setPen(QColor(pal.get(color, pal["accent"])) if self.highlight else QColor(pal["text"]))
            p.drawText(QRectF(8, y, self.width() - 16, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       self.title)
            y += 24
        if self.subtitle:
            f = QFont(self.font())
            f.setPixelSize(13)
            p.setFont(f)
            p.setPen(QColor(pal["muted"]))
            p.drawText(QRectF(8, y, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       self.subtitle)
        self._draw_maze(p, pal)
        p.end()

    def _draw_maze(self, p, pal):
        run = self.runner.run
        mem = run.memory
        maze = run.reveal() if self.reveal else None
        x0, y0, cs = self.cell_geometry()
        H, W = mem.h, mem.w
        fog = qcolor(pal["bg"]).darker(135)
        floor = QColor(pal["surface"])
        seen_old = QColor(pal["surface2"])
        p.setPen(Qt.PenStyle.NoPen)
        p.fillRect(QRectF(x0, y0, cs * W, cs * H), fog)
        r0, c0 = mem.pos
        heat = self.thoughts is not None
        explored = set(self.thoughts.get("explored") or ()) if heat else set()
        for r in range(H):
            for c in range(W):
                known = mem.known[r, c] or maze is not None
                if not known:
                    continue
                rect = QRectF(x0 + c * cs, y0 + r * cs, cs + 0.5, cs + 0.5)
                near = abs(r - r0) <= 1 and abs(c - c0) <= 1
                p.fillRect(rect, floor if (near or maze is not None) else seen_old)
                v = int(mem.visits[r, c])
                if v >= 1:
                    p.fillRect(rect, qcolor(pal["accent"], 50))
                if heat and v >= 2:
                    rgb = theme.heat(-min(v - 1, 3) / 3, self.theme)
                    p.fillRect(rect, QColor(*rgb, 150))
                if (r, c) in explored:
                    p.fillRect(rect, qcolor(pal["warn"], 70))
                if maze is not None and not mem.known[r, c]:
                    p.fillRect(rect, qcolor(pal["bg"], 90))
        # выход и вход
        er, ec = mem.exit
        p.setBrush(QColor(pal["good"]))
        m = cs * 0.18
        p.drawRoundedRect(QRectF(x0 + ec * cs + m, y0 + er * cs + m, cs - 2 * m, cs - 2 * m), cs * 0.12, cs * 0.12)
        sr, sc = mem.start
        p.setBrush(qcolor(pal["muted"], 140))
        p.drawEllipse(QPointF(x0 + (sc + 0.5) * cs, y0 + (sr + 0.5) * cs), cs * 0.16, cs * 0.16)
        # стены
        pen = QPen(QColor(pal["text"]))
        pen.setWidthF(max(1.5, cs * 0.09))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        for r in range(H):
            for c in range(W):
                for d in (0, 3):                      # север и запад; юг и восток - у соседей
                    r2, c2 = r + DR[d], c + DC[d]
                    if not (0 <= r2 < H and 0 <= c2 < W):
                        continue
                    if maze is not None:
                        state = 1 if maze.cells[r, c] & BIT[d] else 0
                    else:
                        state = mem.edge(r, c, d)
                    if state == 0:
                        if d == 0:
                            p.drawLine(QPointF(x0 + c * cs, y0 + r * cs), QPointF(x0 + (c + 1) * cs, y0 + r * cs))
                        else:
                            p.drawLine(QPointF(x0 + c * cs, y0 + r * cs), QPointF(x0 + c * cs, y0 + (r + 1) * cs))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(x0, y0, cs * W, cs * H))
        # след
        trail = mem.trail[-60:]
        if len(trail) > 1:
            col = AGENT_COLORS.get(self.runner.name, "accent")
            tpen = QPen(qcolor(pal.get(col, pal["accent"]), 150))
            tpen.setWidthF(max(1.0, cs * 0.12))
            tpen.setCapStyle(Qt.PenCapStyle.RoundCap)
            tpen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(tpen)
            path = QPainterPath(QPointF(x0 + (trail[0][1] + 0.5) * cs, y0 + (trail[0][0] + 0.5) * cs))
            for r, c in trail[1:]:
                path.lineTo(QPointF(x0 + (c + 0.5) * cs, y0 + (r + 0.5) * cs))
            p.drawPath(path)
        # фишка
        col = AGENT_COLORS.get(self.runner.name, "accent")
        center = QPointF(x0 + (c0 + 0.5) * cs, y0 + (r0 + 0.5) * cs)
        p.setPen(QPen(QColor(pal["bg"]), max(1.0, cs * 0.06)))
        p.setBrush(QColor(pal.get(col, pal["accent"])))
        p.drawEllipse(center, cs * 0.32, cs * 0.32)
        # стрелки «Мыслей»
        probs = self.thoughts.get("probs") if heat else None
        if probs is not None and not self.runner.done:
            self._arrows(p, pal, center, cs, probs)

    def _arrows(self, p, pal, center, cs, probs):
        f = QFont(self.font())
        f.setPixelSize(max(9, min(14, int(cs * 0.32))))
        f.setBold(True)
        p.setFont(f)
        best = max(range(4), key=lambda d: probs[d])
        for d in range(4):
            pr = float(probs[d])
            if pr < 0.02:
                continue
            length = cs * (0.45 + 0.75 * pr)
            dx, dy = DC[d], DR[d]
            tip = QPointF(center.x() + dx * length, center.y() + dy * length)
            color = qcolor(pal["good"] if d == best else pal["warn"], int(110 + 145 * pr))
            pen = QPen(color, max(2.0, cs * 0.12 * (0.5 + pr)))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(center, tip)
            head = cs * 0.2
            nx, ny = -dy, dx
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([
                QPointF(tip.x() + dx * head, tip.y() + dy * head),
                QPointF(tip.x() + nx * head * 0.7, tip.y() + ny * head * 0.7),
                QPointF(tip.x() - nx * head * 0.7, tip.y() - ny * head * 0.7)]))
            if cs >= 22:
                p.setPen(QColor(pal["text"]))
                lx = tip.x() + dx * cs * 0.35
                ly = tip.y() + dy * cs * 0.35
                p.drawText(QRectF(lx - cs * 0.6, ly - cs * 0.3, cs * 1.2, cs * 0.6), Qt.AlignmentFlag.AlignCenter,
                           f"{round(pr * 100)}%")
