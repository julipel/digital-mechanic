"""Индексация руководств: номера страниц, колонтитулы, нарезка, применимость кодов.

Часть тестов работает на синтетических строках и синтетических описаниях документов (без базы
знаний). Тесты по настоящим PDF и индексу пропускаются без KB_DIR / INDEX_DIR (база знаний
внешняя, в репозитории её нет). В тестах — только doc_id, страницы и короткие ключевые слова.
"""
import json
import re
from collections import defaultdict

import pymupdf
import pytest

from app.policy.applicability import load_table
from app.retrieval.chunk import chunk_document, is_heading, page_has_table
from app.retrieval.extract import Line, PageText, clean_page, merge_rows
from app.retrieval.manuals import Manual, load_manuals, manuals_by_id
from tests.conftest import HAS_KB, kb_settings, needs_index, needs_kb

S = kb_settings()
MANUALS = load_manuals(S.kb_dir) if HAS_KB else ()
CHUNKS_FILE = S.index_dir / "chunks.jsonl" if S.index_dir else None
needs_pdf = needs_kb

H = 842.0  # высота страницы A4 в pt
# Синтетические описания документов: правила извлечения как у настоящих, без базы знаний.
RE536 = Manual(doc_id="yamz536_re", manual="536.3902150 РЭ", file="re536.pdf", engine_family="yamz-536",
               kind="РЭ", pages=285, page_number_offset=-1)
RK536 = Manual(doc_id="yamz536_rk", manual="536.01.02 РК", file="rk536.pdf", engine_family="yamz-536",
               kind="РК", pages=603, page_number_offset=None, title_band=(70, 100),
               strip_patterns=(r"^Руководство по ремонту$", r"^двигателей ЯМЗ-536", r"^\d{4}-\d\d-\d\d$",
                               r"^Страница \d+$", r"^\d{3}-\d\d-\d{6}-\d{3}$"))
ZMZ = Manual(doc_id="zmz5234_rk", manual="ЗМЗ-5234.10 РК", file="zmz.pdf", engine_family="zmz-5234",
             kind="РК", pages=158, page_number_offset=0)
RE236 = Manual(doc_id="yamz236_re", manual="236-3902150-Б РЭ", file="re236.pdf", engine_family="yamz-236",
               kind="РЭ", pages=172, page_number_offset=0)


def line(text, y, size=10.0, bold=False, x0=50.0, x1=None):
    return Line(text, y, y + size, size, bold, x0, x1 if x1 is not None else x0 + 6 * len(text))


@pytest.fixture(scope="module")
def chunks():
    return [json.loads(l) for l in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()]


# --- справочник -----------------------------------------------------------------------------

@needs_kb
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
    with pymupdf.open(S.kb_dir / manual.file) as doc:
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
        line("Снятие узла X", 78, 10.1, x0=420),
        line("Порядок работ", 130, 13.9, bold=True),
        line("Отвернуть болты 1.", 160),
        line("000-00-000000-000", 756, x0=462),
        line("2013-02-12", 795, x0=426, x1=480),
        line("Страница 2", 795, x0=494),
    ]
    page = clean_page(raw, 101, H, RK536)
    assert page.title == "Снятие узла X"
    assert page.text == "Порядок работ\nОтвернуть болты 1."


def test_merge_rows_joins_words_and_table_cells():
    merged = merge_rows([line("Гайки", 100, x0=50, x1=80), line("крепления", 100, x0=83, x1=130),
                         line("11…22,3", 100, x0=400, x1=450), line("Следующая", 115)])
    assert [l.text for l in merged] == ["Гайки крепления | 11…22,3", "Следующая"]


# --- нарезка ---------------------------------------------------------------------------------

def test_table_page_is_one_chunk_and_never_crosses_page():
    rows = [line(f"Болт {i} | {10 + i}…{20 + i},5 (1…2)", 100 + 12 * i) for i in range(40)]
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
    by_id = manuals_by_id(S.kb_dir)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        m = by_id[c["doc_id"]]
        assert c["manual"] == m.manual and c["engine_family"] == m.engine_family
        assert 1 <= c["pdf_page"] <= m.pages
        assert c["chunk_id"].startswith(f"{m.doc_id}-p{c['pdf_page']:03d}-")


@needs_index
def test_known_pages(chunks):
    p21 = _page_text(chunks, "zmz5234_rk", 21).lower()
    assert "зазор" in p21 and "коромысл" in p21
    assert "ПОСТОРОННИЙ СТУК" in _page_text(chunks, "yamz536_re", 186)
    p214 = _page_text(chunks, "yamz536_re", 214)
    assert "ПОСТОРОННИЙ СТУК" in p214 and "Негерметична система" in p214
    assert "Моменты затяжки" in _page_text(chunks, "zmz5234_rk", 137)


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
    """Каждый «неприменимый» код подтверждён: ключевые слова об устройстве есть на указанной
    странице, а характерных терминов узла нет ни в одном фрагменте документов этого двигателя."""
    norm = lambda s: re.sub(r"\s+", " ", s.replace(" | ", " ")).lower()
    for family, codes in load_table(S.config_dir).items():  # таблица применимости — в репозитории
        family_chunks = [c for c in chunks if c["engine_family"] == family]
        for code, entry in codes.items():
            assert entry["confirmed"] and entry["evidence"], (family, code)
            for ev in entry["evidence"]:
                page = norm(_page_text(chunks, ev["doc_id"], ev["pdf_page"]))
                assert ev["keywords"] and all(norm(k) in page for k in ev["keywords"]), (family, code, ev)
                assert all(len(k.split()) <= 3 for k in ev["keywords"]), "не цитаты, а ключевые слова"
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


def test_sentence_clause_is_not_carried_as_section():
    """Пункт-предложение («2. Запрещается…», «9.3 Проверить…») режет страницу, но разделом не становится:
    на следующей странице раздел — настоящий заголовок. Строки выдуманы, не из руководств."""
    body = [line("текст " * 12, 200 + 12 * i, 14) for i in range(30)]
    pages = [
        PageText(1, [line("УЗЕЛ ПОДАЧИ ВОЗДУХА", 60, 14, bold=True)] + body[:10]
                 + [line("2. Запрещается мыть корпус узла горячим", 400, 14, bold=True),
                    line("раствором, так как уплотнение теряет форму", 412, 14)] + body[10:20]),
        PageText(2, body[:10] + [line("9.3 Проверить люфт рычага привода, зазор между", 400, 14),
                                 line("рычагом и осью", 412, 14)] + body[10:20]),
        PageText(3, body[:5]),
    ]
    out = chunk_document(ZMZ, pages)
    assert {c.section for c in out} == {"УЗЕЛ ПОДАЧИ ВОЗДУХА"}
    assert any(c.text.startswith("2. Запрещается") for c in out)      # граница фрагмента осталась


def test_clause_from_other_subsection_ends_section():
    """4.6.3 после «4.2 Приёмка…» — раздел 4.2 закончился, откат к главе «4 …». Ближайший пункт
    остаётся в поле heading — это контекст для эмбеддинга."""
    body = [line("Текст абзаца " * 6, 200 + 12 * i, 14) for i in range(10)]
    pages = [
        PageText(1, [line("4 Требования к узлам стенда", 60, 18)] + body),
        PageText(2, [line("4.2 Приёмка узлов стенда", 60, 14, bold=True)] + body),
        PageText(3, [line("4.6.3 Болты крышки стенда затягивать после", 60, 14),
                     line("остывания корпуса", 72, 14)] + body),
    ]
    out = chunk_document(ZMZ, pages)
    assert {c.pdf_page: c.section for c in out} == {1: "4 Требования к узлам стенда",
                                                    2: "4.2 Приёмка узлов стенда",
                                                    3: "4 Требования к узлам стенда"}
    assert {c.pdf_page: c.heading for c in out}[3] == "4.6.3 Болты крышки стенда затягивать после"


def test_multilevel_rows_on_table_page_are_not_sections():
    rows = [line(f"Причина {i} | Устранить {i}", 100 + 12 * i, 10) for i in range(8)]
    pages = [
        PageText(1, [line("ПЕРЕЧЕНЬ ОТКАЗОВ УЗЛА", 60, 10, bold=True),
                     line("4.2 Не горит лампа стенда", 80, 10, bold=True)] + rows),
        PageText(2, [line("текст " * 12, 80 + 12 * i, 10) for i in range(10)]),
    ]
    assert {c.section for c in chunk_document(ZMZ, pages)} == {"ПЕРЕЧЕНЬ ОТКАЗОВ УЗЛА"}


def test_ocr_heading_is_upper_case_line():
    from app.retrieval.chunk import is_ocr_heading
    ocr = lambda t: Line(text=t, y0=0, y1=0, size=0, bold=False)
    assert is_ocr_heading(ocr("ПЕРЕЧЕНЬ ОТКАЗОВ УЗЛА"))
    assert is_ocr_heading(ocr("СТЕНДЫ"))
    assert not is_ocr_heading(ocr("ВНИМАНИЕ! НЕ ВКЛЮЧАТЬ СТЕНД БЕЗ КРЫШКИ"))
    assert not is_ocr_heading(ocr("ОТКАЗ | ДЕЙСТВИЕ"))
    assert not is_ocr_heading(ocr("Проверить крышку стенда"))
    assert not is_heading(line("А Б I В II Д Q k", 100, 14, bold=True), 14)    # подписи рисунка


def test_ocr_page_strips_page_number_and_footers():
    from app.retrieval.build import ocr_page
    page = ocr_page(RK536, 101, "Руководство по ремонту\nОтвернуть болты\nСтраница 2\n", title="Карта")
    assert page.pdf_page == 101 and page.source == "ocr" and page.text == "Отвернуть болты"
    page = ocr_page(RE236, 69, "Гайка | 11-22 (1-2)\n69\n", title=None)
    assert page.text == "Гайка | 11-22 (1-2)"


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
    settings = S.model_copy(update={"index_dir": tmp_path, "kb_dir": tmp_path})
    logged = []
    pages = build.load_pages(manual, settings, log=logged.append)
    assert [p.text for p in pages] == ["стр 1", "текст p002.txt", "стр 3"]
    assert [p.source for p in pages] == ["text", "ocr", "text"]
    assert len(logged) == 3  # извлечение + два пропущенных файла


# --- таблица поперёк книжного листа ----------------------------------------------------------

def _rotated_table_pdf(rotate: int) -> pymupdf.Page:
    """Книжная страница, на которой таблица набрана поперёк листа (как карты дефектации).
    Строки выдуманы, латиницей: в базовом шрифте PDF нет кириллицы. rotate=90 — текст снизу вверх, 270 — сверху вниз."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    rows = [("Pos", "Defect", "Verdict"), ("1", "Crack of stand body", "Reject"),
            ("2", "Axle wear 12,34", "Chrome")]
    for r, cells in enumerate(rows):
        for c, text in enumerate(cells):
            if rotate == 90:     # верх альбомного листа — левый край, строки читаются снизу вверх
                point = pymupdf.Point(100 + 40 * r, 780 - 200 * c)
            else:                # верх альбомного листа — правый край, строки читаются сверху вниз
                point = pymupdf.Point(495 - 40 * r, 60 + 200 * c)
            page.insert_text(point, text, fontsize=10, rotate=rotate, fontname="helv")
    page.insert_text(pymupdf.Point(540, 820), "146", fontsize=10)   # номер страницы — горизонтально
    return page


@pytest.mark.parametrize("rotate", [90, 270])
def test_rotated_table_rows_are_read_in_order(rotate):
    from app.retrieval.extract import _raw_lines
    page = _rotated_table_pdf(rotate)
    lines = merge_rows(_raw_lines(page))
    texts = [l.text for l in lines]
    assert texts[:3] == ["Pos | Defect | Verdict", "1 | Crack of stand body | Reject",
                         "2 | Axle wear 12,34 | Chrome"], texts
    assert texts[-1] == "146"      # строка другого направления — в конце, в своих координатах
    cleaned = clean_page(_raw_lines(page), 146, page.rect.height, ZMZ)
    assert "146" not in cleaned.text.splitlines() and "2 | Axle wear 12,34 | Chrome" in cleaned.text


def test_horizontal_page_order_unchanged():
    from app.retrieval.extract import _raw_lines
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, t in enumerate(["First stand line", "Second stand line"]):
        page.insert_text(pymupdf.Point(72, 100 + 20 * i), t, fontsize=10, fontname="helv")
    assert [l.text for l in _raw_lines(page)] == ["First stand line", "Second stand line"]
