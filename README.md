# VozApp: clonación de voz con XTTS-v2

Subes o grabas una muestra de voz (entre 10 y 30 s) y generas audio de cualquier texto con esa voz. Todo corre localmente; el audio no sale del equipo.

## Instalación (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uso

```powershell
.\iniciar.ps1
```

Abre http://localhost:8100. La primera vez se descarga el modelo XTTS-v2 (unos 1,8 GB).

## Licencia del modelo: importante

XTTS-v2 se distribuye bajo la **Coqui Public Model License (CPML)**, que **solo permite uso no comercial**. `iniciar.ps1` define `COQUI_TOS_AGREED=1`, lo que equivale a aceptar esa licencia. Si se necesita uso comercial o institucional con fines de lucro, se debe usar otro motor (por ejemplo, F5-TTS con licencia compatible o una API de pago).

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/estado` | Dispositivo (CPU o GPU) e idiomas |
| GET | `/api/voces` | Lista las voces registradas |
| POST | `/api/voces` | `multipart`: `nombre`, `consentimiento=true`, `archivos` (uno o varios) |
| GET | `/api/voces/{id}/muestra` | Reproduce la muestra original |
| DELETE | `/api/voces/{id}` | Elimina una voz |
| POST | `/api/sintetizar` | JSON `{voz_id, texto, idioma, velocidad, temperatura}` → WAV |

## Consejos de calidad

- Usa audio limpio, con una sola persona, sin música ni eco. Entre 15 y 30 s es lo ideal.
- Puedes subir varios archivos de la misma persona para mejorar la similitud.
- En CPU, cada oración tarda aproximadamente entre 20 y 60 s. Con una GPU NVIDIA es casi en tiempo real (instala la versión CUDA de torch).

## Uso responsable

Clona únicamente tu propia voz o la de personas que te lo autorizaron expresamente. No uses el audio generado para suplantar a alguien, cometer fraude ni desinformar.
