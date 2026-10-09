"""Мутационная проверка: нарочно ломает продукт и смотрит, что проверки это ловят.

    python tools/mutate.py            # все мутанты
    python tools/mutate.py rules      # только с этим словом в имени

Каждый мутант - замена строки в файле. Файл правится на месте, гоняются указанные
проверки, затем файл восстанавливается (в finally - и при Ctrl+C). Мутант «выжил», если
проверки остались зелёными: значит, этот кусок продукта не проверен.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (имя, файл, было, стало, проверки)
MUTANTS = []          # все мутанты - в tools/mutants.py


def load_more():
    """Мутанты движка и ИИ лежат рядом, чтобы список не разрастался в одном месте."""
    extra = os.path.join(ROOT, "tools", "mutants.py")
    if os.path.exists(extra):
        ns = {}
        with open(extra, encoding="utf-8") as f:
            exec(f.read(), ns)
        return ns["MUTANTS"]
    return []


def run(mutant, python):
    name, rel, old, new, tests = mutant
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read()
    if src.count(old) != 1:
        return name, f"строка найдена {src.count(old)} раз"
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src.replace(old, new))
        r = subprocess.run([python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            *tests.split()], cwd=ROOT, capture_output=True, text=True,
                           timeout=900)
        return name, "пойман" if r.returncode != 0 else "ВЫЖИЛ"
    finally:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)


def main(argv):
    flt = argv[1] if len(argv) > 1 else ""
    mutants = [m for m in MUTANTS + load_more() if flt in m[0]]
    alive = 0
    for m in mutants:
        name, verdict = run(m, sys.executable)
        print(f"{verdict:8} {name}", flush=True)
        alive += verdict != "пойман"
    print(f"итого {len(mutants)}, не пойманы {alive}")
    return 1 if alive else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
