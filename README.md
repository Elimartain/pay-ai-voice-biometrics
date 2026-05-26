# Pay AI — Voice biometric ML

Standalone voice biometric system that enrolls users, learns voice embeddings over time, and verifies speakers through a REST API you can plug into any app.

![Pay AI voice biometrics](docs/images/app-screenshot.png)

Two-layer verification: **ECAPA-TDNN** speaker matching plus optional **Whisper** passphrase challenges, with encrypted templates, rate limits, liveness checks, and adaptive ML scoring.

---

## Features

- Voice enrollment and speaker verification via REST
- Adaptive ML that learns per-user genuine vs impostor behavior
- Optional passphrase challenge (digits + words) for layer-2 verification
- AES-GCM encrypted voice templates
- Reference web UI included (`static/`)
- OpenAPI docs at `/docs`

---

## Tech stack

| Layer | Technology |
|--------|------------|
| Runtime | Python 3.11+ |
| API | FastAPI, Uvicorn |
| Speaker ID | SpeechBrain — ECAPA-TDNN |
| ASR | OpenAI Whisper |
| Audio | librosa, pydub, NumPy |
| ML | PyTorch, torchaudio |
| Security | cryptography (AES-GCM), PyJWT |

---

## Quick start

```powershell
git clone https://github.com/Elimartain/pay-ai-voice-biometrics.git
cd pay-ai-voice-biometrics

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the UI, or [http://localhost:8000/docs](http://localhost:8000/docs) for the API.

### Environment

Copy `.env.example` to `.env` and set your secrets before production:

```env
AES_KEY=your_secure_key
ASR_MODEL_SIZE=tiny
DEV_MODE=true
THRESH_LOW=0.5
THRESH_MED=0.6
THRESH_HIGH=0.7
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Reference UI |
| POST | `/enroll/{user_id}` | Enroll with ≥2 audio files |
| POST | `/verify/{user_id}` | Verify speaker (optional `X-Amount` header) |
| POST | `/verify_passphrase/{user_id}` | Layer-2 verify (`X-Nonce` header required) |
| GET | `/status/{user_id}` | Enrollment status |
| GET | `/health` | Service health |

**Enroll**

```bash
curl -X POST "http://localhost:8000/enroll/user123" \
  -F "files=@voice1.wav" \
  -F "files=@voice2.wav"
```

**Verify**

```bash
curl -X POST "http://localhost:8000/verify/user123" \
  -H "X-Amount: 5000" \
  -F "file=@test.wav"
```

---

## Project structure

```
main.py                  # Primary API server
main_enterprise.py       # Hardened deployment variant
static/                  # Reference web UI
requirements.txt
.env.example
```

---

## License

See [LICENSE](LICENSE) if present, or contact the maintainer for usage terms.
