"""Поиск: фильтр по двигателю ставит код (правило 3), BM25 работает без эмбеддингов.
Тексты фрагментов и числа выдуманы: нормативы руководств в репозиторий не кладём."""
import json

import numpy as np
import pytest

from app.config import Settings
from app.retrieval.search import SearchIndex, rrf, search, tokenize
from tests.conftest import kb_settings, needs_index


def _chunk(cid, family, page, text):
    return {"chunk_id": cid, "doc_id": f"doc_{family}", "manual": f"M {family}", "engine_family": family,
            "pdf_page": page, "section": None, "title": None, "text": text,
            "is_table": False, "is_fault_table": False, "source": "text"}


@pytest.fixture
def tiny_index(tmp_path):
    chunks = [
        _chunk("a-1", "fam-a", 1, "Зазор между коромыслом и клапаном 1,11-2,22 мм"),
        _chunk("a-2", "fam-a", 2, "Давление масла на прогретом двигателе 111-222 кПа"),
        _chunk("b-1", "fam-b", 7, "Зазор между коромыслом и клапаном 3,33-4,44 мм"),
        _chunk("b-2", "fam-b", 8, "Сигнализатор аварийного давления масла загорается при 33-44 кПа"),
    ]
    # фон: в семействе из 2 фрагментов IDF у BM25 Okapi равен нулю, нужен «корпус»
    fillers = ["Порядок работы цилиндров", "Заправочные ёмкости", "Электрооборудование", "Сцепление"]
    chunks += [_chunk(f"{fam[-1]}-f{i}", fam, 100 + i, t) for fam in ("fam-a", "fam-b")
               for i, t in enumerate(fillers)]
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"embedding_model": None}), encoding="utf-8")
    return SearchIndex(tmp_path)


@pytest.fixture
def no_llm(tmp_path):
    return Settings(_env_file=None, embedding_model="", openai_api_key=None, db_path=tmp_path / "t.sqlite")


def test_only_family_documents_returned(tiny_index, no_llm):
    for family in ("fam-a", "fam-b"):
        result = search("зазор коромысло клапан", family, no_llm, k=10, index=tiny_index)
        assert result.hits, family
        assert {h.engine_family for h in result.hits} == {family}


def test_other_family_never_leaks_even_on_exact_match(tiny_index, no_llm):
    # запрос дословно из документа fam-b, но машина с двигателем fam-a
    result = search("Сигнализатор аварийного давления масла загорается при 33-44 кПа", "fam-a",
                    no_llm, k=10, index=tiny_index)
    assert all(h.chunk_id.startswith("a-") for h in result.hits)


def test_engine_family_required(tiny_index, no_llm):
    with pytest.raises(ValueError):
        search("зазор", "", no_llm, index=tiny_index)
    with pytest.raises(ValueError):
        search("зазор", "fam-unknown", no_llm, index=tiny_index)


def test_bm25_mode_without_embeddings(tiny_index, no_llm):
    result = search("давление масла", "fam-a", no_llm, index=tiny_index)
    assert result.mode == "bm25"
    assert result.hits[0].chunk_id == "a-2"
    assert result.hits[0].pdf_page == 2      # страница — из метаданных фрагмента


def test_vector_ranking_filtered_by_family(tiny_index):
    n = len(tiny_index.chunks)
    tiny_index.embeddings = np.eye(n, dtype=np.float32)
    # вектор запроса совпадает с фрагментом b-1, но ищем в fam-a
    query = np.zeros(n, dtype=np.float32)
    query[next(i for i, c in enumerate(tiny_index.chunks) if c["chunk_id"] == "b-1")] = 1
    rows = tiny_index.vector_ranking(query, "fam-a")
    assert {tiny_index.chunks[r]["engine_family"] for r in rows} == {"fam-a"}


def test_rrf_prefers_agreement():
    scores = rrf([[1, 2, 3], [2, 1, 4]])
    assert max(scores, key=scores.get) in (1, 2)
    assert scores[4] < scores[3] or scores[4] == pytest.approx(scores[3])


def test_tokenize_stems_russian_forms():
    assert tokenize("зазоры")[0] == tokenize("зазоров")[0] == tokenize("зазор")[0]
    assert "1,11" in tokenize("зазор 1,11 мм")


@needs_index
def test_real_index_bm25_known_page(no_llm):
    env = kb_settings()
    settings = no_llm.model_copy(update={"index_dir": env.index_dir, "kb_dir": env.kb_dir})
    result = search("зазор между коромыслом и стержнем клапана на холодном двигателе", "zmz-5234",
                    settings, k=5)
    assert ("zmz5234_rk", 21) in {(h.doc_id, h.pdf_page) for h in result.hits}
    assert all(h.engine_family == "zmz-5234" for h in result.hits)


def test_embedding_failure_falls_back_to_bm25(tiny_index, no_llm, monkeypatch):
    import app.retrieval.search as mod
    tiny_index.embeddings = np.eye(len(tiny_index.chunks), dtype=np.float32)
    tiny_index.embedding_model = "emb"
    settings = no_llm.model_copy(update={"embedding_model": "emb"})

    def boom(*a, **kw):
        raise RuntimeError("провайдер недоступен")
    monkeypatch.setattr(mod, "embed_query", boom)
    result = search("давление масла", "fam-a", settings, index=tiny_index)
    assert result.mode == "bm25" and result.hits[0].chunk_id == "a-2"


def test_index_must_match_manuals_registry(tiny_index):
    from app.retrieval.search import check_families
    with pytest.raises(ValueError):
        check_families(tiny_index, {})


def test_unknown_family_in_manifest_rejected():
    from app.retrieval.manuals import Manual, check_families_known
    m = Manual(doc_id="d", manual="M", file="d.pdf", engine_family="yamz-999", kind="РЭ", pages=1)
    with pytest.raises(ValueError):
        check_families_known((m,), {"yamz-236"})


# --- расширение запроса словарём config/query_expansions.json ------------------------------------

def test_expansion_catches_word_forms():
    from app.retrieval.search import expand_query, load_expansions
    groups = load_expansions(Settings(_env_file=None).config_dir)
    for q in ("Можно ли своим ходом перегнать машину на базу?", "Перегоню её в гараж?",
              "Доеду сам до парка?", "Можно ли продолжать движение?", "Вызывать эвакуатор?"):
        expanded, names = expand_query(q, groups)
        assert names == ["движение с неисправностью"], q
        assert expanded.startswith(q) and "прекратить эксплуатацию" in expanded


def test_expansion_not_triggered_by_unrelated_words():
    from app.retrieval.search import expand_query, load_expansions
    groups = load_expansions(Settings(_env_file=None).config_dir)
    # «ход поршня», «переход», «недоход» не должны превращать вопрос в «можно ли ехать»
    for q in ("Какой ход поршня?", "Переход на другое масло", "Зазор клапанов на холодном двигателе"):
        assert expand_query(q, groups) == (q, []), q


def test_expansion_keeps_family_filter_and_original_query(tiny_index, no_llm):
    result = search("Можно ли своим ходом доехать? давление масла", "fam-b", no_llm, k=10, index=tiny_index)
    assert result.query == "Можно ли своим ходом доехать? давление масла"
    assert result.expanded_by == ["движение с неисправностью"]
    assert {h.engine_family for h in result.hits} == {"fam-b"}


def test_missing_or_broken_expansions_file(tmp_path):
    from app.retrieval.search import load_expansions
    assert load_expansions(tmp_path) == ()
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "query_expansions.json").write_text('{"groups": [{"name": "x", "triggers": [" "], "add": "y"}]}',
                                               encoding="utf-8")
    with pytest.raises(ValueError):
        load_expansions(bad)


def test_real_expansions_file_is_valid():
    from app.retrieval.search import load_expansions
    groups = load_expansions(Settings(_env_file=None).config_dir)
    assert groups and all(e.add and e.triggers for e in groups)


@pytest.mark.parametrize("content", [
    "{не json", '{"items": []}', '{"groups": [{"name": "x", "add": "y"}]}',
    '{"groups": [{"name": "x", "triggers": ["перегн"], "add": 5}]}',
    '{"groups": [{"name": "x", "triggers": ["перегн"], "add": "давление 111 кПа"}]}',
])
def test_broken_expansions_file_gives_value_error(tmp_path, content):
    from app.retrieval.search import load_expansions
    (tmp_path / "query_expansions.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_expansions(tmp_path)
