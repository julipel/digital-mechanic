"""Рендер одной страницы руководства в PNG — для чипа ссылки «документ, стр. N» в интерфейсе.

Требование заказчика: полный PDF наружу не отдаётся, только картинка одной страницы
и только авторизованной сессии с ограничением частоты (ограничение — в app/api.py).
Документ ищется только в манифесте KB_DIR/kb.yaml по doc_id — путь из запроса не берётся.
Номер страницы — индекс страницы PDF с 1 (правило 2), тот же, что в ссылках.
Кеш PNG — в PAGE_CACHE_DIR (KB и индекс в контейнере смонтированы только на чтение).
"""
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import pymupdf

from app.config import Settings
from app.retrieval.manuals import manuals_by_id


class PageNotFound(LookupError):
    """Нет такого документа в манифесте или такой страницы в нём."""


_render_lock = threading.Lock()


def render_page(settings: Settings, doc_id: str, pdf_page: int) -> Path:
    """PNG страницы pdf_page (с 1) документа doc_id; возвращает путь к файлу в кеше."""
    manual = manuals_by_id(settings.require_kb_dir()).get(doc_id)
    if manual is None or not 1 <= pdf_page <= manual.pages:
        raise PageNotFound(f"{doc_id}: страница {pdf_page}")
    cache_dir = settings.page_cache_dir
    # doc_id из манифеста, не из запроса: имя файла кеша не может выйти из каталога
    target = cache_dir / f"{manual.doc_id}-p{pdf_page:04d}-{settings.page_render_dpi}.png"
    if target.exists():
        return target
    with _render_lock:                      # параллельные запросы одной страницы не рендерят её дважды
        if target.exists():
            return target
        kb_dir = settings.require_kb_dir().resolve()
        source = (kb_dir / manual.file).resolve()
        if not source.is_relative_to(kb_dir):     # манифест не может указать файл вне базы знаний
            raise PageNotFound(f"{doc_id}: файл вне KB_DIR")
        with pymupdf.open(source) as doc:
            if pdf_page > doc.page_count:
                raise PageNotFound(f"{doc_id}: страница {pdf_page}")
            pixmap = doc[pdf_page - 1].get_pixmap(dpi=settings.page_render_dpi)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        pixmap.save(str(tmp), output="png")
        tmp.replace(target)
    return target


class RateLimiter:
    """Скользящее окно: не больше limit запросов за window_s секунд на ключ (сессию)."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            if len(self._hits) > 1000:          # не копим ключи бесконечно
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True
