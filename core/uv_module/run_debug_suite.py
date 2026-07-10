#!/usr/bin/env python3
"""
Запуск pytest по uv_module/tests с максимальным логированием на stderr.

Использование из корня репозитория dutin:
  python uv_module/run_debug_suite.py
  python uv_module/run_debug_suite.py -k baker

Не вставляй в zsh отдельную строку «# ...» из доков (иначе zsh: command not found: #,
если не включён setopt interactivecomments).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    try:
        import pytest
    except ImportError:
        print("Нужен pytest: pip install pytest", file=sys.stderr)
        return 1

    suite = _REPO / "uv_module" / "tests" / "test_uv_module_suite.py"
    args = [
        str(suite),
        "-vv",
        "--tb=short",
        "--log-cli-level=DEBUG",
        "--color=yes",
        *sys.argv[1:],
    ]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
