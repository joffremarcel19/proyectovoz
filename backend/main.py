"""API de clonación de voz: registrar voces a partir de audio y sintetizar texto."""
import io
import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from motor import IDIOMAS, SAMPLE_RATE, MotorVoz, convertir_a_wav

BASE = Path(__file__).resolve().parent.parent
DIR_VOCES = BASE / "datos" / "voces"
DIR_VOCES.mkdir(parents=True, exist_ok=True)

EXTENSIONES = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".webm", ".aac", ".mp4"}
MAX_BYTES = 25 * 1024 * 1024
DURACION_MIN = 5.0

app = FastAPI(title="VozApp - Clonación de voz")
motor = MotorVoz()


def _leer_meta(voz_id: str) -> dict:
    ruta = DIR_VOCES / voz_id / "meta.json"
    if not ruta.exists():
        raise HTTPException(404, "Voz no encontrada")
    return json.loads(ruta.read_text(encoding="utf-8"))


@app.get("/api/estado")
def estado():
    return {"modelo_cargado": motor.cargado, "dispositivo": motor.dispositivo, "idiomas": IDIOMAS}


@app.get("/api/voces")
def listar_voces():
    voces = []
    for carpeta in sorted(DIR_VOCES.iterdir()):
        meta = carpeta / "meta.json"
        if meta.exists():
            voces.append(json.loads(meta.read_text(encoding="utf-8")))
    return sorted(voces, key=lambda v: v["creado"], reverse=True)


@app.post("/api/voces")
def crear_voz(
    nombre: str = Form(..., min_length=1, max_length=80),
    consentimiento: bool = Form(...),
    archivos: list[UploadFile] = File(...),
):
    if not consentimiento:
        raise HTTPException(400, "Debe confirmar que tiene autorización de la persona dueña de la voz.")

    voz_id = uuid.uuid4().hex[:12]
    carpeta = DIR_VOCES / voz_id
    carpeta.mkdir()
    try:
        muestras, duracion = [], 0.0
        with tempfile.TemporaryDirectory() as tmp:
            for i, archivo in enumerate(archivos):
                ext = Path(archivo.filename or "").suffix.lower()
                if ext not in EXTENSIONES:
                    raise HTTPException(400, f"Formato no soportado: {archivo.filename}")
                datos = archivo.file.read(MAX_BYTES + 1)
                if len(datos) > MAX_BYTES:
                    raise HTTPException(400, f"{archivo.filename} supera 25 MB")
                origen = Path(tmp) / f"orig{i}{ext}"
                origen.write_bytes(datos)
                destino = carpeta / f"muestra{i}.wav"
                try:
                    duracion += convertir_a_wav(origen, destino)
                except Exception:
                    raise HTTPException(400, f"No se pudo leer el audio {archivo.filename}")
                muestras.append(destino)

        if duracion < DURACION_MIN:
            raise HTTPException(400, f"El audio es muy corto ({duracion:.1f} s). Se necesitan al menos {DURACION_MIN:.0f} s.")

        try:
            huella = motor.extraer_huella(muestras)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        torch.save(huella, carpeta / "huella.pt")

        meta = {
            "id": voz_id,
            "nombre": nombre.strip(),
            "duracion_muestra": round(duracion, 1),
            "creado": datetime.now().isoformat(timespec="seconds"),
        }
        (carpeta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta
    except Exception:
        shutil.rmtree(carpeta, ignore_errors=True)
        raise


@app.get("/api/voces/{voz_id}/muestra")
def escuchar_muestra(voz_id: str):
    _leer_meta(voz_id)
    return FileResponse(DIR_VOCES / voz_id / "muestra0.wav", media_type="audio/wav")


@app.delete("/api/voces/{voz_id}")
def eliminar_voz(voz_id: str):
    _leer_meta(voz_id)
    shutil.rmtree(DIR_VOCES / voz_id)
    return {"ok": True}


class PeticionSintesis(BaseModel):
    voz_id: str
    texto: str = Field(..., min_length=1, max_length=3000)
    idioma: str = "es"
    velocidad: float = Field(1.0, ge=0.5, le=2.0)
    temperatura: float = Field(0.7, ge=0.1, le=1.0)


@app.post("/api/sintetizar")
def sintetizar(p: PeticionSintesis):
    _leer_meta(p.voz_id)
    if p.idioma not in IDIOMAS:
        raise HTTPException(400, f"Idioma no soportado: {p.idioma}")
    huella = torch.load(DIR_VOCES / p.voz_id / "huella.pt", weights_only=True)
    try:
        wav = motor.sintetizar(p.texto, huella, p.idioma, p.velocidad, p.temperatura)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav",
                             headers={"Content-Disposition": 'inline; filename="voz.wav"'})


app.mount("/", StaticFiles(directory=BASE / "frontend", html=True), name="frontend")
