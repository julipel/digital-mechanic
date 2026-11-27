"""Загрузка материалов задачи при развёртывании: PDF руководств и данные заглушек — по ссылке
из переменной окружения MATERIALS_URL (публичная папка Яндекс Диска). Ссылки в коде нет.

Руководства и данные заглушек в репозиторий не коммитятся (права правообладателей, п. 9.1).
Скрипт:
- получает список файлов папки через открытое API Яндекс Диска;
- каждый PDF скачивает и кладёт в KB_DIR под именем из манифеста config/kb.yaml — сопоставление
  по sha256 (имена на Диске другие); PDF с неизвестным хэшем пропускается; файл с верным хэшем
  повторно не скачивается;
- из архива заглушек берёт data/vehicles.json → VEHICLES_PATH и сверяет data/fault_types.json
  с копией в репозитории;
- копирует манифест config/kb.yaml в KB_DIR/kb.yaml;
- права: каталог KB_DIR — 700, файлы — 600 (на Windows chmod ограничен — не ошибка).
Только стандартная библиотека: запускается на хосте сервера без установки пакетов. Прокси — из
HTTPS_PROXY, как у urllib.

    MATERIALS_URL=... KB_DIR=/srv/digital-mechanic/kb python3 scripts/fetch_materials.py
"""
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://cloud-api.yandex.net/v1/disk/public/resources"
MANIFEST = ROOT / "config" / "kb.yaml"
CHUNK = 1 << 20


def read_env(path: Path) -> dict[str, str]:
    """KEY=VALUE из .env (без интерпретации): скрипт работает и без python-dotenv."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                out[m.group(1)] = m.group(2).strip("'\"")
    return out


def manifest_files(manifest: Path) -> dict[str, str]:
    """sha256 → имя файла из манифеста. Разбор строк file:/sha256: — без PyYAML."""
    files, current = {}, None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-?\s*file:\s*(\S+)\s*$", line)
        if m:
            current = m.group(1).strip("'\"")
        m = re.match(r"^\s*sha256:\s*([0-9a-f]{64})\s*$", line)
        if m and current:
            files[m.group(1)] = current
            current = None
    return files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def list_items(public_url: str) -> list[dict]:
    query = urllib.parse.urlencode({"public_key": public_url, "limit": 200})
    with urllib.request.urlopen(f"{API}?{query}", timeout=60) as r:
        data = json.load(r)
    return [i for i in data.get("_embedded", {}).get("items", []) if i.get("type") == "file"]


def download(public_url: str, path: str, dest: Path) -> None:
    query = urllib.parse.urlencode({"public_key": public_url, "path": path})
    with urllib.request.urlopen(f"{API}/download?{query}", timeout=60) as r:
        href = json.load(r)["href"]
    with urllib.request.urlopen(href, timeout=600) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f, CHUNK)


def private(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def fetch(public_url: str, kb_dir: Path, vehicles_path: Path, *, manifest: Path = MANIFEST,
          fault_types_repo: Path = ROOT / "mocks" / "data" / "fault_types.json",
          list_fn=list_items, download_fn=download, log=print) -> int:
    """Возвращает 0 — всё на месте, 1 — чего-то из манифеста или данных заглушек не хватает."""
    known = manifest_files(manifest)
    kb_dir.mkdir(parents=True, exist_ok=True)
    private(kb_dir, 0o700)
    have = {sha256_file(p): p.name for p in kb_dir.glob("*.pdf")}
    got_vehicles = False
    for item in list_fn(public_url):
        name, sha = item["name"], item.get("sha256")
        if name.lower().endswith(".pdf"):
            target = known.get(sha) if sha else None
            if sha and target and have.get(sha) == target:
                log(f"  {target}: уже на месте")
                continue
            with tempfile.TemporaryDirectory(dir=kb_dir) as tmp:
                part = Path(tmp) / "download.part"
                download_fn(public_url, item["path"], part)
                real = sha256_file(part)
                target = known.get(real)
                if target is None:
                    log(f"  {name}: sha256 {real[:12]}… нет в манифесте — пропущен")
                    continue
                dest = kb_dir / target
                os.replace(part, dest)
                private(dest, 0o600)
                have[real] = target
                log(f"  {name} → {target}")
        elif name.lower().endswith(".zip"):
            buf = io.BytesIO()
            with tempfile.TemporaryDirectory() as tmp:
                part = Path(tmp) / "mock.zip"
                download_fn(public_url, item["path"], part)
                buf.write(part.read_bytes())
            with zipfile.ZipFile(buf) as z:
                names = z.namelist()
                veh = next((n for n in names if n.endswith("data/vehicles.json")), None)
                ft = next((n for n in names if n.endswith("data/fault_types.json")), None)
                if veh is None:
                    log(f"  {name}: нет data/vehicles.json — пропущен")
                    continue
                vehicles_path.parent.mkdir(parents=True, exist_ok=True)
                vehicles_path.write_bytes(z.read(veh))
                private(vehicles_path, 0o600)
                got_vehicles = True
                log(f"  {name} → {vehicles_path}")
                if ft and fault_types_repo.exists() and z.read(ft) != fault_types_repo.read_bytes():
                    log(f"  ВНИМАНИЕ: fault_types.json в архиве отличается от {fault_types_repo}")
    kb_manifest = kb_dir / "kb.yaml"
    shutil.copyfile(manifest, kb_manifest)
    private(kb_manifest, 0o600)
    missing = sorted(set(known.values()) - set(have.values()))
    for m in missing:
        log(f"  НЕТ: {m} — в материалах не найден файл с sha256 из манифеста")
    if not got_vehicles and not vehicles_path.exists():
        log("  НЕТ: vehicles.json — в материалах не найден архив заглушек")
    return 1 if missing or not vehicles_path.exists() else 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    env = {**read_env(ROOT / ".env"), **os.environ}
    url = env.get("MATERIALS_URL")
    kb = env.get("KB_DIR")
    if not url or not kb:
        print("Задайте MATERIALS_URL и KB_DIR в .env или окружении")
        return 2
    vehicles = Path(env.get("VEHICLES_PATH") or "mocks/data/vehicles.json")
    if not vehicles.is_absolute():
        vehicles = ROOT / vehicles
    print(f"Материалы → KB_DIR={kb}, vehicles={vehicles}")
    return fetch(url, Path(kb), vehicles)


if __name__ == "__main__":
    sys.exit(main())
