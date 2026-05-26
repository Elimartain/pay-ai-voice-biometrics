# Pay AI — Voice biometric ML

**Standalone** voice biometric system: enroll users, learn their voice embeddings over time (**adaptive ML**), verify who is speaking—then expose everything through a **clean REST API** so you can plug it into **any app** (fintech flows, SSO, workforce access, or custom tooling). The bundled web UI is a reference client, not required for integrations.

Under the hood: **two-layer verification** — ECAPA-TDNN speaker checks plus optional passphrase challenges (Whisper ASR)—with encrypted templates, rate limits, liveness, and risk-tier thresholds.

![App screenshot placeholder](docs/images/app-screenshot.png)

_Add a screenshot: save your UI capture as `docs/images/app-screenshot.png` (create the folder if needed), or update this line to point to your image._

---

## Tech stack

| Layer | Technology |
|--------|------------|
| Runtime | Python **3.11+** |
| API | **FastAPI**, **Uvicorn** |
| Speaker ID | **SpeechBrain** — ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) |
| ASR | **OpenAI Whisper** (`openai-whisper`, size via `ASR_MODEL_SIZE`) |
| Audio | **librosa**, **pydub**, **NumPy**; optional **webrtcvad** (falls back if missing) |
| ML | **PyTorch**, **torchaudio** |
| Security | **cryptography** (AES-GCM), **PyJWT** (enterprise variant), PBKDF2 key derivation |
| Frontend | Static **HTML/CSS/JS** (`static/`), **MediaRecorder** → 16 kHz WAV uploads |
| Config | **python-dotenv** (`.env`) |

---

## Plug this into your product (REST API)

Treat this repo as **drop-in biometric infrastructure**:

1. **Run the server** (`uvicorn main:app`).
2. **Call HTTP endpoints** from your backend or mobile/web client—JSON + multipart uploads, compatible with OpenAPI codegen and API gateways.
3. **Discover the contract interactively**: `GET /docs` (Swagger UI) and `GET /openapi.json` for codegen (TypeScript clients, Kotlin, etc.).

Your users can live entirely in **your UX** while enrollment, verification, challenge issuance, and learning happen through this API. Typical integration path: **`POST /enroll/{user_id}`** → store your own `user_id` mapping → **`POST /verify/{user_id}`** on each biometric check → handle optional **`verify_passphrase`** when the API returns a layer-2 challenge.

---

## What this project includes

Detailed design notes live in [`NOTES.md`](NOTES.md). Summary:

### Core API server (`main.py`)

- **`main.py`** — Primary standalone service: enterprise-style audio preprocessing (normalization, VAD, duration limits 0.5–7 s, liveness via spectral flatness), ECAPA embeddings with **192-dim assertion**, AES-GCM–encrypted averaged enrollment templates, **ML adaptive scoring** so the system keeps learning per-user genuine vs impostor behavior, risk-based thresholds from **`X-Amount`**, nonce-bound passphrase challenges (digits + words, 30 s expiry), Whisper transcription for layer 2, rate limiting, input sanitization, admin endpoints for ML stats / cohort reset, `/health` and `/status`.

- **`main_enterprise.py`** — Optional **hardened** deployment sketch for teams who need JWT, signed requests, KMS-style envelope patterns, and stricter ingress controls before putting the same biometric API behind production traffic.

- **`main_backup_working.py`** — Earlier reference server with simpler preprocessing and `/debug-upload`; handy if you bisect regressions—not the preferred integration surface.

### Frontend

- **`static/index.html`** — Glass-style UI for enroll, verify, and passphrase flows with challenge display.
- **`static/voice-auth.js`** — `VoiceAuthenticator` class: microphone, waveform UI, WAV conversion, `FormData` uploads, `X-Nonce` on passphrase verify.

### Tools & tests

- **`debug_embeddings.py`** — Inspect ECAPA embeddings and cosine behavior on synthetic audio.
- **`test_embedding_dims.py`** — Guardrail that ECAPA output dimension is expected (192).
- **`test_direct.py`** — Smoke test `/health`, `/enroll`, `/verify` with **requests**.

### Models & weights

- **ECAPA**: loaded via SpeechBrain (Hugging Face); you can place weights under `pretrained_models/spkrec-ecapa-voxceleb/` if you need offline drops.
- **Whisper**: downloaded according to `ASR_MODEL_SIZE` on first use.

---

## Local setup

### 1. Clone and enter the repo

```powershell
git clone <your-repo-url>
cd voice-ml
```

### 2. Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. PyTorch (optional GPU)

CPU-only is fine; for CUDA (example CUDA 11.8 wheel index):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Application dependencies

```powershell
pip install -r requirements.txt
```

Optional **webrtcvad** (needs a C++ toolchain on Windows; app works without it):

```powershell
pip install webrtcvad
```

### 5. Environment

```powershell
copy .env.example .env
```

Edit `.env`: set strong secrets (`AES_KEY`, `HMAC_SECRET`, `JWT_SECRET`, etc.) and thresholds. Never commit `.env`.

### 6. Run the server

```powershell
uvicorn main:app --reload
```

- UI: [http://localhost:8000](http://localhost:8000)
- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

### 7. Optional checks

```powershell
python test_embedding_dims.py
python debug_embeddings.py
python test_direct.py
```

_(Start the server first for `test_direct.py`.)_

---

## API reference (`main.py`)

These routes are stable integration points—you can gateway them, attach service auth, or run behind API management.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serves SPA (`static/index.html`) |
| POST | `/enroll/{user_id}` | Multipart `files` (≥2); preprocess, embed, encrypt template |
| POST | `/verify/{user_id}` | Layer 1 verify; optional `X-Amount` for threshold tier; may return passphrase challenge |
| POST | `/verify_passphrase/{user_id}` | Layer 2; `file` + `X-Nonce` |
| GET | `/status/{user_id}` | Enrollment / legacy status |
| GET | `/health` | Models, VAD mode, config snapshot |
| GET | `/admin/ml_stats/{user_id}` | Adaptive ML stats |
| POST | `/admin/reset_cohort` | Reset impostor cohort (admin) |

Example **curl** (see also docstring at top of `main.py`):

```bash
curl -X POST "http://localhost:8000/enroll/user123" -F "files=@voice1.wav" -F "files=@voice2.wav"
curl -X POST "http://localhost:8000/verify/user123" -H "X-Amount: 5000" -F "file=@test.wav"
```

---

## Screenshots folder

Suggested layout:

```text
docs/
  images/
    app-screenshot.png      # Main UI
    enrollment-flow.png      # Optional: extra captures
```

Reference them from this README with standard Markdown images.

---

## Publish to GitHub

### Recommended repo metadata

Use this when creating the repo (GitHub **About** box, `gh repo create`, or repo settings).

| Field | Suggested value |
|--------|------------------|
| **Repository name** | `pay-ai-voice-biometrics` |
| **Short description** | Standalone voice biometric ML with adaptive speaker learning — FastAPI REST API for enrollment, verification, and plug-in integrations. |
| **Website** (optional) | Link to your demo or docs once deployed |
| **Topics** | `voice-biometrics`, `speaker-verification`, `speaker-recognition`, `biometrics`, `machine-learning`, `fastapi`, `rest-api`, `speechbrain`, `whisper`, `pytorch`, `ecapa-tdnn`, `voice-authentication`, `adaptive-learning` |

**Alternative repo names** (if `pay-ai-voice-biometrics` is taken): `pay-ai-voice-ml`, `payai-voice-biometrics`, `pay-ai-speaker-biometrics`.

**About / tagline** (paste into GitHub description):

```text
Standalone voice biometric ML that learns users over time. Enroll, verify, and integrate via REST — ECAPA-TDNN + Whisper, adaptive scoring, encrypted templates.
```

---

## Git: first push

Repo is already initialized locally. Before committing, set **your** identity for this repo only:

```powershell
cd "g:\voice ml"
git config --local user.name "Your Name"
git config --local user.email your-github-email@example.com
```

Use the same email as your GitHub account (or your GitHub `noreply` address) so you—not Cursor—show as the contributor.

### Option A — Create repo with GitHub CLI

Replace `<your-github-username>` with your handle:

```powershell
cd "g:\voice ml"
git add .
git commit -m "Initial commit: Pay AI voice biometric ML"
gh repo create pay-ai-voice-biometrics --public --source=. --remote=origin --description "Standalone voice biometric ML with adaptive speaker learning — FastAPI REST API for enrollment, verification, and plug-in integrations." --push
```

Add topics after create:

```powershell
gh repo edit --add-topic voice-biometrics,speaker-verification,biometrics,machine-learning,fastapi,rest-api,speechbrain,whisper,pytorch,ecapa-tdnn,adaptive-learning
```

### Option B — Create repo on github.com, then push

1. On GitHub: **New repository** → name **`pay-ai-voice-biometrics`** → paste the short description above → do **not** add a README (this repo already has one).
2. Then:

```powershell
cd "g:\voice ml"
git add .
git commit -m "Initial commit: Pay AI voice biometric ML"
git remote add origin https://github.com/<your-github-username>/pay-ai-voice-biometrics.git
git branch -M main
git push -u origin main
```

This repository ships with `.gitignore` entries for `.venv`, `.env`, `__pycache__`, and `.cursor/` so local tooling is not tracked.

---

## Security notes

- Replace all default/demo secrets before production.
- `DEV_MODE=true` exposes more error detail—disable in production.
- Review CORS (`allow_origins`) for real deployments; **`main_enterprise.py`** sketches stricter controls.
- Stored voice templates are AES-GCM–encrypted at rest (in-memory store in dev; persistence is your deployment concern).

---

## License

Specify your license here (e.g. MIT, Apache-2.0, or proprietary).
