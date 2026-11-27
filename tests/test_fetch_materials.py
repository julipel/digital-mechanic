"""Загрузка материалов по ссылке: PDF сопоставляются с манифестом по sha256, неизвестные
пропускаются, уже скачанные не перекачиваются, vehicles.json берётся из архива заглушек.
Сеть не нужна: список и загрузка подменяются. Содержимое файлов выдумано."""
import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("fetch_materials",
                                              Path(__file__).resolve().parent.parent / "scripts" / "fetch_materials.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

PDF_A, PDF_B, PDF_X = b"%PDF stand A", b"%PDF stand B", b"%PDF unknown"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mock_2026/data/vehicles.json", '{"vehicles": []}')
        z.writestr("mock_2026/data/fault_types.json", "[]")
    return buf.getvalue()


def setup(tmp_path):
    manifest = tmp_path / "kb.yaml"
    manifest.write_text(f"manuals:\n- doc_id: a\n  file: a.pdf\n  sha256: {sha(PDF_A)}\n"
                        f"- doc_id: b\n  file: b.pdf\n  sha256: {sha(PDF_B)}\n", encoding="utf-8")
    ft = tmp_path / "fault_types.json"
    ft.write_text("[]", encoding="utf-8")
    remote = {"/Руководство А.pdf": PDF_A, "/Руководство Б.pdf": PDF_B, "/Чужое.pdf": PDF_X,
              "/mock.zip": make_zip(), "/ТЗ.docx": b"docx"}
    calls = []

    def list_fn(url):
        return [{"type": "file", "name": p.lstrip("/"), "path": p, "sha256": sha(b)} for p, b in remote.items()]

    def download_fn(url, path, dest):
        calls.append(path)
        dest.write_bytes(remote[path])
    return manifest, ft, list_fn, download_fn, calls


def test_fetch_places_files_by_sha_and_extracts_vehicles(tmp_path):
    manifest, ft, list_fn, download_fn, calls = setup(tmp_path)
    kb, veh = tmp_path / "kb", tmp_path / "data" / "vehicles.json"
    rc = fm.fetch("https://example.test/d/x", kb, veh, manifest=manifest, fault_types_repo=ft,
                  list_fn=list_fn, download_fn=download_fn, log=lambda *a: None)
    assert rc == 0
    assert (kb / "a.pdf").read_bytes() == PDF_A and (kb / "b.pdf").read_bytes() == PDF_B
    assert sorted(p.name for p in kb.iterdir()) == ["a.pdf", "b.pdf", "kb.yaml"]     # чужой PDF и docx — нет
    assert veh.read_text(encoding="utf-8") == '{"vehicles": []}'
    assert "/ТЗ.docx" not in calls

    calls.clear()                                                                       # повторный запуск
    rc = fm.fetch("https://example.test/d/x", kb, veh, manifest=manifest, fault_types_repo=ft,
                  list_fn=list_fn, download_fn=download_fn, log=lambda *a: None)
    assert rc == 0 and "/Руководство А.pdf" not in calls and "/Руководство Б.pdf" not in calls


def test_missing_manual_is_reported(tmp_path):
    manifest, ft, list_fn, download_fn, _ = setup(tmp_path)
    only_a = lambda url: [i for i in list_fn(url) if i["path"] != "/Руководство Б.pdf"]
    log = []
    rc = fm.fetch("https://example.test/d/x", tmp_path / "kb", tmp_path / "v.json", manifest=manifest,
                  fault_types_repo=ft, list_fn=only_a, download_fn=download_fn, log=log.append)
    assert rc == 1 and any("b.pdf" in m for m in log)


def test_real_manifest_lists_all_manuals():
    files = fm.manifest_files(fm.MANIFEST)
    assert len(files) == 6 and all(len(h) == 64 for h in files)
