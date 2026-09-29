"""Motor de clonación de voz basado en Coqui XTTS-v2."""
import os
import re
import subprocess
import threading
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch

MODELO = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000
# XTTS tiene un límite de ~250 caracteres por fragmento en español.
MAX_CHARS = 220

IDIOMAS = ["es", "en", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
           "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]


def convertir_a_wav(origen: Path, destino: Path) -> float:
    """Convierte cualquier audio a WAV mono 22.05 kHz y devuelve su duración en segundos."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(origen), "-ac", "1", "-ar", "22050",
         "-af", "silenceremove=start_periods=1:start_threshold=-50dB", str(destino)],
        check=True, capture_output=True,
    )
    info = sf.info(str(destino))
    return info.frames / info.samplerate


def dividir_texto(texto: str) -> list[str]:
    """Divide el texto en fragmentos cortos respetando oraciones y comas."""
    texto = re.sub(r"\s+", " ", texto).strip()
    oraciones = re.split(r"(?<=[.!?¿¡;:])\s+", texto)
    fragmentos: list[str] = []
    for oracion in oraciones:
        while len(oracion) > MAX_CHARS:
            corte = oracion.rfind(",", 0, MAX_CHARS)
            if corte < MAX_CHARS // 3:
                corte = oracion.rfind(" ", 0, MAX_CHARS)
            if corte <= 0:
                corte = MAX_CHARS
            fragmentos.append(oracion[:corte + 1].strip())
            oracion = oracion[corte + 1:].strip()
        if oracion:
            fragmentos.append(oracion)
    return fragmentos


class MotorVoz:
    def __init__(self):
        self._modelo = None
        self._lock = threading.Lock()
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def cargado(self) -> bool:
        return self._modelo is not None

    def cargar(self):
        if self._modelo is not None:
            return
        from TTS.api import TTS

        if os.environ.get("COQUI_TOS_AGREED") != "1":
            raise RuntimeError(
                "Debe aceptar la licencia Coqui Public Model License (uso no comercial) "
                "definiendo la variable de entorno COQUI_TOS_AGREED=1."
            )
        if self.dispositivo == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        tts = TTS(MODELO).to(self.dispositivo)
        self._modelo = tts.synthesizer.tts_model

    def extraer_huella(self, muestras: list[Path]) -> dict:
        """Calcula los latentes de la voz a partir de las muestras de audio."""
        self.cargar()
        with self._lock, torch.inference_mode():
            gpt_latent, speaker_emb = self._modelo.get_conditioning_latents(
                audio_path=[str(m) for m in muestras],
                gpt_cond_len=30, max_ref_length=60,
            )
        return {"gpt_cond_latent": gpt_latent.cpu(), "speaker_embedding": speaker_emb.cpu()}

    def sintetizar(self, texto: str, huella: dict, idioma: str = "es",
                   velocidad: float = 1.0, temperatura: float = 0.7) -> np.ndarray:
        self.cargar()
        pausa = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
        partes: list[np.ndarray] = []
        gpt_latent = huella["gpt_cond_latent"].to(self.dispositivo)
        speaker_emb = huella["speaker_embedding"].to(self.dispositivo)
        with self._lock, torch.inference_mode():
            for fragmento in dividir_texto(texto):
                salida = self._modelo.inference(
                    fragmento, idioma, gpt_latent, speaker_emb,
                    temperature=temperatura, speed=velocidad,
                    repetition_penalty=5.0, enable_text_splitting=False,
                )
                wav = salida["wav"]
                if torch.is_tensor(wav):
                    wav = wav.cpu().numpy()
                partes.extend([np.asarray(wav, dtype=np.float32), pausa])
        return np.concatenate(partes[:-1]) if partes else np.zeros(0, dtype=np.float32)
