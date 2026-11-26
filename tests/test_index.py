"""Индексация руководств: номера страниц, колонтитулы, нарезка, применимость кодов.

Часть тестов работает на синтетических строках (без PDF). Тесты по настоящим PDF и
построенному индексу пропускаются, если их нет (PDF и index/ в git не публикуются):
    python scripts/build_index.py --no-embed
"""
import json
import re
from collections import defaultdict

import pymupdf
import pytest

from app.config import get_settings
from app.policy.applicability import load_table
from app.retrieval.chunk import chunk_document, is_heading, page_has_table
from app.retrieval.extract import Line, PageText, clean_page, merge_rows
from app.retrieval.manuals import Manual, load_manuals, manuals_by_id

S = get_settings()
MANUALS = load_manuals(S.config_dir)
CHUNKS_FILE = S.index_dir / "chunks.jsonl"
HAS_PDF = all((S.manuals_dir / m.file).exists() for m in MANUALS)
HAS_INDEX = CHUNKS_FILE.exists()
needs_pdf = pytest.mark.skipif(not HAS_PDF, reason="нет PDF в data/manuals/")
needs_index = pytest.mark.skipif(not HAS_INDEX, reason="индекс не построен: python scripts/build_index.py --no-embed")

H = 842.0  # высота страницы A4 в pt
RE536 = manuals_by_id(S.config_dir)["yamz536_re"]
RK536 = manuals_by_id(S.config_dir)["yamz536_rk"]
ZMZ = manuals_by_id(S.config_dir)["zmz5234_rk"]


def line(text, y, size=10.0, bold=False, x0=50.0, x1=None):
    return Line(text, y, y + size, size, bold, x0, x1 if x1 is not None else x0 + 6 * len(text))


@pytest.fixture(scope="module")
def chunks():
    return [json.loads(l) for l in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()]


# --- справочник -----------------------------------------------------------------------------

def test_manuals_config_matches_tz():
    by_family = defaultdict(set)
    for m in MANUALS:
        by_family[m.engine_family].add(m.manual)
    assert by_family["yamz-236"] == {"236-3902150-Б РЭ", "236.01.01 РК"}
    assert by_family["yamz-536"] == {"536.3902150 РЭ", "536.01.02 РК"}
    assert len(by_family["zmz-5234"]) == 1 and len(by_family["zmz-409"]) == 1
    assert {m.pages for m in MANUALS} == {172, 287, 285, 603, 158, 238}


@needs_pdf
@pytest.mark.parametrize("manual", MANUALS, ids=lambda m: m.doc_id)
def test_pdf_page_count(manual: Manual):
    with pymupdf.open(S.manuals_dir / manual.file) as doc:
        assert doc.page_count == manual.pages


# --- номер страницы не берётся из текста ---------------------------------------------------

def test_printed_number_with_offset_is_removed():
    """536 РЭ: на PDF-странице 187 напечатано «186» — вырезается; число из таблицы — нет."""
    page = clean_page([line("Текст страницы", 100), line("186", 800)], 187, H, RE536)
    assert page.pdf_page == 187 and "186" not in page.text
    kept = clean_page([line("Текст страницы", 100), line("37", 800)], 38, H, ZMZ)
    assert "37" in kept.text  # не номер этой страницы — содержимое (цифра таблицы)


def test_number_in_page_body_is_kept():
    page = clean_page([line("21", 400)], 21, H, ZMZ)
    assert page.text == "21"


def test_536_rk_header_footer_and_card_title():
    raw = [
        line("Руководство по ремонту", 35, 11, bold=True),
        line("двигателей ЯМЗ-536, ЯМЗ-5361,", 47, 11, bold=True),
        line("Снятие клапана заслонки EGR", 78, 10.1, x0=420),
        line("Порядок работ", 130, 13.9, bold=True),
        line("Отвернуть болты 1.", 160),
        line("536-20-010000-040", 756, x0=462),
        line("2013-02-12", 795, x0=426, x1=480),
        line("Страница 2", 795, x0=494),
    ]
    page = clean_page(raw, 101, H, RK536)
    assert page.title == "Снятие клапана заслонки EGR"
    assert page.text == "Порядок работ\nОтвернуть болты 1."


def test_merge_rows_joins_words_and_table_cells():
    merged = merge_rows([line("Гайки", 100, x0=50, x1=80), line("крепления", 100, x0=83, x1=130),
                         line("98…107,8", 100, x0=400, x1=450), line("Следующая", 115)])
    assert [l.text for l in merged] == ["Гайки крепления | 98…107,8", "Следующая"]


# --- нарезка ---------------------------------------------------------------------------------

def test_table_page_is_one_chunk_and_never_crosses_page():
    rows = [line(f"Болт {i} | {90 + i}…{100 + i},5 (9…10)", 100 + 12 * i) for i in range(40)]
    pages = [
        PageText(1, [line("Таблица 23", 80)] + rows),
        PageText(2, [line("ОБЩИЕ СВЕДЕНИЯ", 60, 14, bold=True)] + [line("текст " * 12, 80 + 12 * i) for i in range(40)]),
    ]
    out = chunk_document(ZMZ, pages)
    first = [c for c in out if c.pdf_page == 1]
    assert len(first) == 1 and first[0].is_table and "Болт 39" in first[0].text
    assert all(c.chunk_id.startswith(f"{ZMZ.doc_id}-p{c.pdf_page:03d}-") for c in out)
    assert {c.manual for c in out} == {ZMZ.manual} and {c.engine_family for c in out} == {"zmz-5234"}
    assert len([c for c in out if c.pdf_page == 2]) > 1  # обычный текст режется


def test_toc_line_is_not_heading():
    assert not is_heading(line("СИСТЕМА ОХЛАЖДЕНИЯ ..................... 116", 100, bold=True), 10)
    assert is_heading(line("СИСТЕМА ОХЛАЖДЕНИЯ", 100, 14, bold=True), 10)
    assert is_heading(line("4.7 Фильтр масляный", 100, 14), 14)


def test_fault_table_header_marks_table():
    page = PageText(214, [line("Причины неисправности", 60), line("Способ устранения", 60, x0=300)] +
                    [line("Подтянуть соединения", 80 + 10 * i) for i in range(8)])
    assert page_has_table(page)


# --- построенный индекс ----------------------------------------------------------------------

def _page_text(chunks, doc_id, page):
    return "\n".join(c["text"] for c in chunks if c["doc_id"] == doc_id and c["pdf_page"] == page)


@needs_index
def test_index_chunk_metadata(chunks):
    by_id = manuals_by_id(S.config_dir)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        m = by_id[c["doc_id"]]
        assert c["manual"] == m.manual and c["engine_family"] == m.engine_family
        assert 1 <= c["pdf_page"] <= m.pages
        assert c["chunk_id"].startswith(f"{m.doc_id}-p{c['pdf_page']:03d}-")


@needs_index
def test_known_pages(chunks):
    assert "0,20…0,30 мм" in _page_text(chunks, "zmz5234_rk", 21)
    assert "ПОСТОРОННИЙ СТУК" in _page_text(chunks, "yamz536_re", 186)
    p214 = _page_text(chunks, "yamz536_re", 214)
    assert "ПОСТОРОННИЙ СТУК" in p214 and "Негерметична система" in p214
    assert "98…107,8 (10…11)" in _page_text(chunks, "zmz5234_rk", 137)


@needs_index
def test_printed_page_numbers_and_536_rk_footers_removed(chunks):
    lines_by_page = defaultdict(set)
    for c in chunks:
        lines_by_page[(c["doc_id"], c["pdf_page"])].update(c["text"].splitlines())
    re536 = [p for (d, p) in lines_by_page if d == "yamz536_re"]
    leftover = [p for p in re536 if str(p - 1) in lines_by_page[("yamz536_re", p)]]
    assert len(leftover) <= 3, leftover  # единичные совпадения с цифрами таблиц допустимы
    rk = [c for c in chunks if c["doc_id"] == "yamz536_rk"]
    assert all(c["title"] for c in rk)
    assert not any(re.fullmatch(r"Страница \d+|Конец модуля данных", l) for c in rk for l in c["text"].splitlines())


@needs_index
def test_applicability_confirmed_by_manual_text(chunks):
    """Каждый «неприменимый» код подтверждён: цитата об устройстве есть на указанной странице,
    а характерных терминов узла нет ни в одном фрагменте документов этого двигателя."""
    norm = lambda s: re.sub(r"\s+", " ", s.replace(" | ", " ")).lower()
    for family, codes in load_table(S.config_dir).items():
        family_chunks = [c for c in chunks if c["engine_family"] == family]
        for code, entry in codes.items():
            assert entry["confirmed"] and entry["evidence"], (family, code)
            for ev in entry["evidence"]:
                assert norm(ev["quote"]) in norm(_page_text(chunks, ev["doc_id"], ev["pdf_page"])), (family, code, ev)
            for term in entry["absent_terms"]:
                hits = [c["chunk_id"] for c in family_chunks if re.search(term, c["text"], re.IGNORECASE)]
                assert not hits, (family, code, term, hits[:3])


# --- исправления по ревью этапа 2 -----------------------------------------------------------

def test_merged_fault_header_and_pipe_rows_mark_table():
    header = PageText(54, [line("Причина неисправности | Метод устранения", 60)] +
                      [line("Проверить зазор", 80 + 10 * i) for i in range(8)])
    assert page_has_table(header)
    rows = PageText(5, [line(f"Болты {i} | 10-12", 80 + 10 * i) for i in range(6)] +
                    [line("текст", 200 + 10 * i) for i in range(4)])
    assert page_has_table(rows)


def test_numbered_list_item_is_not_heading():
    assert not is_heading(line("1. Настроить и проверить центробежный датчик", 100, 14), 14)


def test_ocr_page_strips_page_number_and_footers():
    from app.retrieval.build import ocr_page
    page = ocr_page(RK536, 101, "Руководство по ремонту\nОтвернуть болты\nСтраница 2\n", title="Карта")
    assert page.pdf_page == 101 and page.source == "ocr" and page.text == "Отвернуть болты"
    re236 = manuals_by_id(S.config_dir)["yamz236_re"]
    page = ocr_page(re236, 69, "Гайка | 100-140 (10-14)\n69\n", title=None)
    assert page.text == "Гайка | 100-140 (10-14)"


def test_ocr_files_outside_scan_pages_are_ignored(tmp_path, monkeypatch):
    """Номер страницы OCR берётся из scan_pages, а не из имени файла: p000.txt и p999.txt
    не подменяют страницы."""
    from app.retrieval import build
    manual = Manual(doc_id="t", manual="T", file="t.pdf", engine_family="zmz-5234", kind="РК",
                    pages=3, scan_pages=(2,))
    ocr = tmp_path / "ocr" / "t"
    ocr.mkdir(parents=True)
    for name in ("p000.txt", "p002.txt", "p999.txt"):
        (ocr / name).write_text(f"текст {name}", encoding="utf-8")
    monkeypatch.setattr(build, "extract_pdf", lambda pdf, m: [PageText(i, [line(f"стр {i}", 100)]) for i in (1, 2, 3)])
    monkeypatch.setattr(build, "_file_hash", lambda pdf: "x")
    settings = S.model_copy(update={"index_dir": tmp_path, "manuals_dir": tmp_path})
    logged = []
    pages = build.load_pages(manual, settings, log=logged.append)
    assert [p.text for p in pages] == ["стр 1", "текст p002.txt", "стр 3"]
    assert [p.source for p in pages] == ["text", "ocr", "text"]
    assert len(logged) == 3  # извлечение + два пропущенных файла
