"""Построение поискового индекса по руководствам из data/manuals/.

Запуск (Windows и Linux одинаково, из корня проекта):
    python scripts/build_index.py              # фрагменты + эмбеддинги (если модель настроена в .env)
    python scripts/build_index.py --no-embed   # только фрагменты (BM25), без обращения к API
Повторный запуск дешёвый: страницы и эмбеддинги берутся из кеша в index/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.retrieval.build import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-embed", action="store_true", help="не считать эмбеддинги")
    args = parser.parse_args()
    manifest = build_index(get_settings(), embeddings=not args.no_embed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
