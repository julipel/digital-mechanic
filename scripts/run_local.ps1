# Запуск без Docker под Windows: заглушки в фоне, приложение — в этом окне.
# Перед первым запуском: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$ml = Start-Process -FilePath $py -ArgumentList "-m uvicorn mock_ml.app:app --app-dir mocks --port 8001" -PassThru -NoNewWindow
$toir = Start-Process -FilePath $py -ArgumentList "-m uvicorn mock_toir.app:app --app-dir mocks --port 8002" -PassThru -NoNewWindow
try {
    & $py -m uvicorn app.main:app --port 8000 --reload
} finally {
    Stop-Process -Id $ml.Id, $toir.Id -ErrorAction SilentlyContinue
}
