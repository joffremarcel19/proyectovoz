# Inicia VozApp en http://localhost:8100
# Requiere haber aceptado la Coqui Public Model License (ver README).
$env:COQUI_TOS_AGREED = "1"
Set-Location "$PSScriptRoot\backend"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8100
