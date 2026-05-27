# Pay AI — Voice biometric ML

Standalone voice biometric system that enrolls users, learns voice embeddings over time, and verifies speakers through a REST API you can plug into any app.



<img width="785" height="613" alt="image" src="https://github.com/user-attachments/assets/b9cdf21f-caf6-472e-b959-23f0c4a4a659" />
<img width="960" height="322" alt="image" src="https://github.com/user-attachments/assets/f5372b93-ed3c-43d5-be4d-191f607a5561" />
<img width="960" height="347" alt="image" src="https://github.com/user-attachments/assets/51604005-ed7e-423c-9df4-14d2fd73425b" />
<img width="844" height="302" alt="image" src="https://github.com/user-attachments/assets/c64c8916-92d2-4f09-902a-9fd8fb0cba8e" />





_Two-layer verification: **ECAPA-TDNN** speaker matching plus optional **Whisper** passphrase challenges, with encrypted templates, rate limits, liveness checks, and adaptive ML scoring._

---

## How it works

```
User records 2+ enrollment samples
        ↓
Audio pipeline: normalize → VAD → liveness (spectral flatness) → 16 kHz mono
        ↓
ECAPA-TDNN extracts 192-dimensional voice embedding per sample
        ↓
Embeddings averaged, L2-normalized, stored as AES-GCM encrypted template
        ↓
On verification → new embedding vs stored template
        ↓
Cosine-style similarity score + adaptive ML threshold (per-user history)
        ↓
Optional risk tier via X-Amount header (LOW / MED / HIGH thresholds)
        ↓
If score passes → Layer 1 success
        ↓
If challenged → Whisper transcribes spoken passphrase (digits + words) + speaker re-check
        ↓
Both layers pass → Biometric verification approved
```

The **adaptive ML** layer tracks genuine vs impostor score patterns per user (sliding windows), nudging decision boundaries over time so the system improves with real traffic—not a fixed single threshold forever.

---

## Features

- Voice enrollment and speaker verification over **REST**
- **Adaptive scoring** — per-user genuine / impostor history informs decisions
- **Two-layer security** — voice match + optional Whisper passphrase challenge (nonce-bound, time-limited)
- **AES-GCM** encrypted embeddings — no raw audio persisted as the long-term template
- **Configurable tiers** — `THRESH_LOW` / `THRESH_MED` / `THRESH_HIGH` (e.g. with `X-Amount` on verify)
- **Rate limiting**, **input sanitization**, and **liveness** checks in the default server
- Reference **web UI** (`static/`) — optional; integrations can use HTTP only
- **OpenAPI** — interactive docs at `/docs`, schema at `/openapi.json`
- **Enterprise blueprint** — `main_enterprise.py` (JWT, HMAC, stricter limits — configure before production)

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Runtime | Python 3.11+ |
| API | FastAPI, Uvicorn |
| Speaker ID | SpeechBrain — ECAPA-TDNN (`spkrec-ecapa-voxceleb`) |
| ASR | OpenAI Whisper |
| Audio | librosa, pydub, NumPy; optional webrtcvad |
| ML | PyTorch, torchaudio |
| Security | cryptography (AES-GCM), PyJWT (enterprise variant) |

---

## ML architecture

### Speaker verification — ECAPA-TDNN

- Produces **192-dimensional** embeddings (asserted in the pipeline for consistency).
- Strong on **short utterances** compared to classic x-vector style pipelines.
- Verification compares enrollment template vs probe embedding; similarity feeds the adaptive layer.

### Adaptive scoring

- Maintains per-user statistics from past accepts / rejects.
- Uses embedding-derived confidence signals alongside raw similarity.
- Reduces “one global threshold fits nobody” failure modes as usage grows.

### Layer 2 — Whisper ASR

- Server issues a **nonce-bound** challenge (digits + curated words), short expiry window.
- Client records the user speaking the challenge; **Whisper** transcribes and normalizes text.
- Pass requires **correct transcript + nonce** and continued speaker consistency checks.

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

- **UI:** [http://localhost:8000](http://localhost:8000)
- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### GPU (optional)

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment (`.env`)

```env
AES_KEY=your_secure_key
HMAC_SECRET=your_hmac_secret
JWT_SECRET=your_jwt_secret
ASR_MODEL_SIZE=tiny
DEV_MODE=true
THRESH_LOW=0.5
THRESH_MED=0.6
THRESH_HIGH=0.7
```

Never commit `.env`. Use strong random values in production.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Reference web UI |
| POST | `/enroll/{user_id}` | Multipart: ≥2 `files` (WAV recommended) |
| POST | `/verify/{user_id}` | Multipart: `file`; optional `X-Amount` for tiered threshold |
| POST | `/verify_passphrase/{user_id}` | Multipart: `file`; header `X-Nonce` from challenge |
| GET | `/status/{user_id}` | Enrollment / challenge-related status |
| GET | `/health` | Models loaded, VAD mode, config snapshot |
| GET | `/admin/ml_stats/{user_id}` | Adaptive ML telemetry (protect in production) |
| POST | `/admin/reset_cohort` | Reset synthetic cohort (admin) |

### Enroll

```bash
curl -X POST "http://localhost:8000/enroll/user123" \
  -F "files=@voice1.wav" \
  -F "files=@voice2.wav"
```

### Verify

```bash
curl -X POST "http://localhost:8000/verify/user123" \
  -H "X-Amount: 5000" \
  -F "file=@test.wav"
```

### Example response shapes

Responses are JSON; exact fields depend on pass/fail and whether a **layer-2 challenge** is issued. Typical patterns:

**Layer 1 success (illustrative)**

```json
{
  "verified": true,
  "score": 0.6945,
  "message": "Voice verification successful",
  "confidence": "HIGH"
}
```

**Challenge issued (illustrative)**

```json
{
  "verified": false,
  "requires_passphrase": true,
  "challenge": "123456 word1 word2",
  "nonce": "abc123...",
  "expires_in": 30
}
```

_Use `/docs` on a running instance for the authoritative schema per endpoint._

---

## Project structure

```
main.py                  # Primary FastAPI server (adaptive ML + two-layer flow)
main_enterprise.py       # Hardened deployment sketch (JWT, HMAC, rate buckets)
main_backup_working.py   # Earlier reference implementation
static/                  # Reference UI (HTML + voice-auth.js)
debug_embeddings.py      # ECAPA embedding inspection utility
test_embedding_dims.py   # Embedding dimension guardrail
test_direct.py           # Smoke tests against running server
requirements.txt
.env.example
docs/images/             # Screenshots for README
```

---

## Results & design targets

| Item | Value / note |
|------|----------------|
| Embedding size | **192** (ECAPA path) |
| Enrollment samples | **≥ 2** files |
| Template at rest | **AES-GCM** encrypted averaged embedding |
| ASR | **Whisper** (`ASR_MODEL_SIZE` env) |
| Demo similarity score | e.g. **~0.69** on a passing run (varies by mic, room, user) |

_Benchmarks are environment-dependent; treat the score above as an example order of magnitude, not a guarantee._

---

## Security notes

- Rotate `AES_KEY`, `HMAC_SECRET`, and `JWT_SECRET` for any shared deployment.
- Set `DEV_MODE=false` in production to avoid leaking stack traces.
- Lock down `/admin/*` behind your own auth / network policy.
- Tighten CORS in `main.py` or use `main_enterprise.py` patterns when exposing publicly.

---

## Author

**Anish Raj** — B.Tech AI, USAR Delhi  
[LinkedIn](https://in.linkedin.com/in/anish-raj-3976b029b) · [GitHub](https://github.com/Elimartain)

---

## License

Specify your license here (e.g. MIT) or add a `LICENSE` file.

