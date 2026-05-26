# Pay AI — Voice Biometric ML (standalone) — Comprehensive Notes

This document captures everything implemented in the `voice ml` workspace: standalone **voice biometric ML** that learns users’ voices via adaptive scoring and verification, exposed as a **pluggable REST API** plus optional reference UI. It covers components, two-layer flows, security, and tooling—not skipped; each major file is summarized.

---

## 1. Repository Layout

- `main.py` – Primary **standalone** FastAPI service: biometric audio preprocessing, ML-based scoring, nonce-bound passphrase challenges, adaptive learning.
- `main_enterprise.py` – Hardened production variant with JWT auth, HMAC signing, envelope encryption, token-bucket rate limiting, and extended audit/security scaffolding.
- `main_backup_working.py` – Earlier stable baseline with simpler preprocessing, AES-GCM storage, passphrase fallback, and debug endpoints.
- `static/index.html` + `static/voice-auth.js` – Glassmorphic single-page UI and rich MediaRecorder client for enrollment, verification, and passphrase capture.
- `debug_embeddings.py` – CLI tool to inspect real ECAPA embeddings and understand score normalization behavior.
- `test_direct.py` – Smoke-test client hitting `/health`, `/enroll`, `/verify` directly (bypasses frontend).
- `test_embedding_dims.py` – Verifies ECAPA output dimensionality (requirement #9 guardrail).
- `requirements.txt` – Python dependencies (FastAPI, SpeechBrain, Whisper, cryptography, etc.).
- `README.md` – Setup instructions, API overview, security notes, and model references.
- `pretrained_models/spkrec-ecapa-voxceleb/` – Local cache placeholder for ECAPA weights (used when avoiding symlink issues).

---

## 2. Core API service (`main.py`)

### 2.1 Configuration & Models
- Loads env vars via `python-dotenv`; defines risk-based thresholds (`THRESH_LOW/MED/HIGH`) and legacy `THRESHOLD`.
- AES key derived with PBKDF2-HMAC-SHA256; HMAC/JWT/KMS placeholders included for enterprise parity.
- Audio pipeline constants: 16 kHz mono, 0.5–7 s duration clamps, WebRTC VAD aggressiveness level 2, spectral-flatness liveness threshold.
- Global state caches SpeechBrain ECAPA-TDNN (`speaker_model`), Whisper ASR (`whisper_model`), encrypted embeddings, ML learning stats, impostor cohort, and challenge/ratelimit trackers.

### 2.2 Audio & Liveness Pipeline
- `load_audio()` enforces file size limits (≤100 MB), writes to temp WAV, loads via librosa with fallback manual WAV parser.
- Peak-normalizes, runs enterprise VAD (`enterprise_webrtc_vad`) that emulates WebRTC with RMS, spectral centroid, and ZCR heuristics plus morphological smoothing.
- Pads/truncates to `[0.5, 7]` seconds and validates final waveform before converting to tensors.
- `check_liveness()` computes spectral flatness (without `sr` param) and compares to 0.5 threshold; errors fail open.

### 2.3 Embedding Extraction & Scoring
- `extract_embedding()` loads ECAPA with CPU run opts, ensures ≥0.5 s audio, pads if needed, and **asserts 192-dim output** per requirement #9 before L2-normalizing.
- `initialize_impostor_cohort()` synthesizes 150 realistic male/female embeddings (vs earlier random noise) for future s-norm use.
- `ml_adaptive_scoring()` replaces fake s-norm with a learning system: tracks genuine/impostor scores per user, calculates confidence from embedding sparsity/norm, adapts thresholds using historical statistics, and penalizes deviations.
- `update_user_learning()` maintains sliding windows (20 genuine, 50 impostor) so the adaptive threshold keeps evolving.

### 2.4 Passphrase Challenges & ASR
- Challenges mix 6 random digits + 2 curated words, bind them to a nonce, and expire after 30 s (`Challenge` dataclass).
- `generate_passphrase_challenge()` stores active challenges; `verify_passphrase_challenge()` enforces nonce, expiry, and normalized text equality.
- `asr_check_passphrase_enterprise()` runs Whisper transcription (language forced to English), strips punctuation, normalizes whitespace, and feeds challenge verification.

### 2.5 Encryption & Storage
- Current production path encrypts averaged enrollment embeddings via AES-GCM (`encrypt_data`/`decrypt_data`). Envelope-encryption scaffolding (`EnvelopeKey`, `decrypt_embedding`) is stubbed for future KMS integration; `main.py` warns if invoked.
- Active data structures also maintain legacy `pending_passphrases` for backward compatibility.

### 2.6 Rate Limiting & Security
- Legacy `check_rate_limit()` enforces 3 attempts/minute per user with logging.
- Input sanitization rejects empty/long IDs and enforces `^[a-zA-Z0-9._-]+$`.
- Logs highlight challenge generation, verification stats, and failure traces (`traceback.format_exc()` in DEV_MODE).

### 2.7 API Endpoints
- `/` serves `static/index.html`.
- `/enroll/{user_id}`: expects ≥2 files, performs enterprise preprocessing + liveness, averages embeddings, encrypts template, stores in-memory.
- `/verify/{user_id}`: risk-tier thresholding via `X-Amount`, enterprise preprocessing, ML adaptive scoring, challenge issuance on fail (with nonce text for frontend). Learning system updates genuine/impostor histories accordingly.
- `/verify_passphrase/{user_id}`: requires `X-Nonce`, reuses enterprise preprocessing, Whisper transcription, nonce verification, adaptive speaker scoring, and reports success/failure with transcripts.
- `/admin/ml_stats/{userId}` & `/admin/reset_cohort`: admin utilities for inspecting ML stats and reinitializing the impostor cohort.
- `/status/{user_id}`: reports enrollment/migration status (legacy `pending_passphrases`).
- `/health`: exposes model load state, VAD mode, and config snapshot.
- `uvicorn` entrypoint guarded by `if __name__ == "__main__":`.

---

## 3. Alternate API entrypoints

### 3.1 `main_backup_working.py`
- Represents the pre-enterprise implementation with simpler preprocessing (librosa + optional manual parsing), AES-GCM encrypted embeddings, deterministic passphrase words, and `/debug-upload` endpoint to inspect uploads.
- Uses traditional cosine scoring against averaged embeddings, triggers word-only challenges (no nonce/digit requirement), lacks ML adaptation but already enforces VAD, liveness, and rate limiting.
- Serves as fallback reference if enterprise branch regresses.

### 3.2 `main_enterprise.py`
- Blueprint for production deployment:
  - JWT authentication with `HTTPBearer`, claim validation, and expiration checks.
  - Request HMAC signature validation (`X-Device-Id`, `X-Nonce`, `X-Timestamp`, `X-Signature`) that hashes `method + path + sha256(body) + timestamp + nonce`.
  - Envelope encryption with mock KMS/KEK flow (AES-GCM DEK wrapping, IV/tag storage).
  - Token-bucket rate limiting keyed by user, device, and IP; risk-based thresholds derived from amount headers.
  - Expanded passphrase dictionary (512+ curated words) and `Challenge` dataclass with digits+words and expiry.
  - Enterprise audio preprocessing identical to `main.py`, plus placeholder anti-spoof models.
  - `s_norm_score()` uses impostor cohort for normalized scoring; still seeds random embeddings but is intended to pair with `debug_embeddings.py` learnings.
  - HTTPS redirect middleware toggle for production environments and restricted CORS origins.
  - Admin endpoints for key rotation, crypto-erase, etc., scaffolded (some not yet implemented).

---

## 4. Frontend (static/)

### 4.1 `index.html`
- Modern glassmorphic layout with sections for enrollment and verification, status alerts, progress bars, animated wave bars, and passphrase challenge UI.
- Buttons: start/stop recording, submit enrollment/verification/passphrase.
- Displays challenge instructions (digit+word string) when the API requests layer-2 verification.

### 4.2 `voice-auth.js`
- `VoiceAuthenticator` class orchestrates microphone permissions, recordings, waveform visualization, and HTTP submission.
- Uses `MediaRecorder` with dynamic MIME negotiation (`audio/webm;codecs=opus` fallback to WAV/default).
- Maintains reusable `audioStream`, handles analyzer data for wave bars, converts recordings to 16 kHz mono WAV via `AudioContext`, and enforces 2+ recordings before allowing enrollment submission.
- Enrollment uses FormData array field `files`; verification and passphrase use `file`.
- Handles passphrase challenge display (stores `challengeNonce`, passes via `X-Nonce` header on submission).
- Shows success/error/info status cards, auto-hides successes, manages progress bar visibility, and exposes instance to global scope for inline button handlers.

---

## 5. Tooling & Tests

- `debug_embeddings.py`: Generates 10 synthetic audio clips (noise + sine waves), extracts ECAPA embeddings, prints shape/mean/std/stats, and cosine similarities to understand why synthetic impostor cohorts fail. Guides the realistic cohort changes in `main.py`.
- `test_embedding_dims.py`: Loads ECAPA, encodes random audio, prints dimension (expected 192) to guarantee requirement #9; warns if 512 or unexpected.
- `test_direct.py`: Quick `requests`-based ping to ensure `/health`, `/enroll`, `/verify` endpoints exist (expects 422 validation errors for missing files).

---

## 6. Dependencies & Models

- Core stack: FastAPI, Uvicorn, SpeechBrain (ECAPA), Torch/Torchaudio, Librosa, NumPy, Pydub, Cryptography, Whisper, python-multipart.
- Optional: `webrtcvad` (falls back gracefully on Windows/librosa path), PyJWT, SciPy (for enterprise VAD morphology).
- `requirements.txt` groups standard deps plus optional ones; we run on Python 3.11+ per README.
- Models: SpeechBrain `spkrec-ecapa-voxceleb` for embeddings, OpenAI Whisper (size configurable via `ASR_MODEL_SIZE` env). `pretrained_models/spkrec-ecapa-voxceleb` directory exists for manual model drops if HuggingFace cache is blocked.

---

## 7. How the System Works (End-to-End)

1. **Enrollment**
   - User records ≥2 samples via frontend; each sample converted to WAV, uploaded as multipart FormData.
   - The API validates ID, preprocesses audio (normalization, VAD, liveness), extracts ECAPA embeddings, averages and normalizes them, encrypts with AES-GCM, and stores in-memory map keyed by user ID.

2. **Primary Verification**
   - User records fresh audio; the service preprocesses identically, passes spectral-flatness liveness, extracts embedding.
   - Adaptive ML scoring compares with stored template; risk-based threshold selected from `X-Amount`.
   - If score ≥ threshold and confidence ≠ LOW, success; otherwise, the API issues nonce-bound challenge text + expiry and updates learning stats (genuine vs impostor).

3. **Passphrase Challenge**
   - Frontend shows digits + 2 words and records passphrase audio; sends along `X-Nonce`.
   - The service transcribes via Whisper, normalizes text, confirms nonce/time window, and re-runs adaptive speaker scoring.
   - Both challenge text and speaker score must pass; success updates ML genuine stats, failure updates impostor stats and reports reason.

4. **Admin & Monitoring**
   - `/health` reveals readiness; `/admin/ml_stats` surfaces learning telemetry.
   - `debug_embeddings.py` and `test_embedding_dims.py` supply offline verification of embedding pipeline consistency.

---

## 8. Quality & Security Considerations

- **Audio Quality**: Strict 0.5–7 s range, normalization, and VAD ensure embeddings are stable. Liveness check (spectral flatness) mitigates replay/synthetic inputs; enterprise plan references anti-spoof models (AASIST/RawNet2) for future.
- **Security**: AES-GCM encryption, nonce-bound passphrases, pending KMS envelope encryption, rate limiting, user input sanitization, risk-based thresholds, and logging provide layered defenses. Enterprise variant adds JWT, HMAC, HTTPS redirects, and token buckets per user/device/IP.
- **Reliability**: Preloading models on startup reduces first-call latency; fallback manual WAV parser keeps pipeline alive even if librosa fails.
- **Learning System**: Adaptive thresholds prevent rigid scores from locking out real users and stop “fake s-norm” issues. System continues to learn from both successful and failed attempts.

---

## 9. Usage Summary

- Install deps (`pip install -r requirements.txt`), set env (`THRESH_*`, `AES_KEY`, `ASR_MODEL_SIZE`, `DEV_MODE`, etc.).
- Run the service: `uvicorn main:app --reload`.
- Open `http://localhost:8000` to access frontend UI or hit endpoints directly (curl examples in `README.md`).
- Optional: use `main_enterprise.py` for production-grade deployments once JWT/HMAC/HTTPS creds configured.

---

This note consolidates every component in the repository—core and alternate APIs, frontend, scripts, and dependencies—and how they support a standalone **voice biometric ML** product with REST integration for external apps.

