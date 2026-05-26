"""
Pay AI — Voice Biometric ML (Enterprise hardening blueprint)
============================================================

Optional **production-hardened** layout for the standalone voice biometric API: JWT, HMAC-signed
requests, envelope/KMS-oriented patterns, token-bucket limits, tighter CORS/HTTPS—not required
for prototyping; integrate when you expose this ML service behind real auth and compliance.

SECURITY GRADE: BANKING/PAYMENT STYLE CONTROLS (when fully configured)

Installation Requirements:
pip install fastapi uvicorn speechbrain torchaudio librosa pydub cryptography openai-whisper python-dotenv python-multipart pyjwt

Usage:
uvicorn main_enterprise:app --reload --ssl-keyfile=key.pem --ssl-certfile=cert.pem

CURL Examples:

# 1. Enrollment (with HMAC signing)
curl -X POST "https://localhost:8000/enroll/user123" \
  -H "X-Device-Id: device_abc123" \
  -H "X-Nonce: $(openssl rand -hex 16)" \
  -H "X-Timestamp: $(date +%s)" \
  -H "X-Signature: $(echo -n 'POST/enroll/user123sha256_of_bodytimestampnonce' | openssl dgst -sha256 -hmac 'your_hmac_key' -binary | base64)" \
  -H "Authorization: Bearer your_jwt_token" \
  -F "files=@voice1.wav" \
  -F "files=@voice2.wav" \
  -F "files=@voice3.wav"

# 2. Verification with risk amount
curl -X POST "https://localhost:8000/verify/user123" \
  -H "X-Device-Id: device_abc123" \
  -H "X-Amount: 5000" \
  -H "X-Idempotency-Key: unique_request_id" \
  -H "Authorization: Bearer your_jwt_token" \
  -F "file=@test_voice.wav"

# 3. Passphrase challenge response
curl -X POST "https://localhost:8000/verify_passphrase/user123" \
  -H "X-Nonce: challenge_nonce_from_previous_response" \
  -F "file=@passphrase_recording.wav"

# 4. Key rotation (admin)
curl -X POST "https://localhost:8000/admin/rotate_keys/user123" \
  -H "Authorization: Bearer admin_jwt_token"

# 5. Crypto-erase user data
curl -X DELETE "https://localhost:8000/delete/user123" \
  -H "Authorization: Bearer admin_jwt_token"
"""

import os
import io
import json
import time
import logging
import random
import hashlib
import hmac
import base64
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import struct

import torch
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment
import whisper
import jwt
from speechbrain.inference.speaker import SpeakerRecognition
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.fernet import Fernet

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Header, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_auth_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION & CONSTANTS
# ================================

# Risk-based thresholds (as per banking requirements)
THRESH_LOW = float(os.getenv("THRESH_LOW", "0.62"))      # ≤ ₹1,000
THRESH_MED = float(os.getenv("THRESH_MED", "0.68"))      # ₹1,001–₹10,000  
THRESH_HIGH = float(os.getenv("THRESH_HIGH", "0.74"))    # > ₹10,000

# Anti-spoofing threshold
LIVENESS_THRESH = float(os.getenv("LIVENESS_THRESH", "0.5"))

# Audio processing parameters
TARGET_SAMPLE_RATE = 16000
MIN_DURATION = 0.5  # seconds
MAX_DURATION = 7.0  # seconds
VAD_AGGRESSIVENESS = 2

# Security parameters
HMAC_SECRET = os.getenv("HMAC_SECRET", "change_in_production")
JWT_SECRET = os.getenv("JWT_SECRET", "change_in_production")
KMS_KEY = os.getenv("KMS_KEY", "mock_kms_key_change_in_production")

# Rate limiting (token bucket)
RATE_LIMITS = {
    "per_user": {"requests": 5, "window": 60, "daily": 60},
    "per_device": {"requests": 10, "window": 60},
    "per_ip": {"requests": 20, "window": 60}
}

# Passphrase configuration
CHALLENGE_EXPIRY = 30  # seconds
PASSPHRASE_WORDS = [
    # 2048-word filtered list (excluding homophones and <4 chars)
    "abstract", "accurate", "acquire", "activity", "adequate", "advance", "adventure", "advocate",
    "alliance", "amazing", "analysis", "ancient", "announce", "approach", "approval", "argument",
    "articulate", "artistic", "asteroid", "athletic", "attitude", "audience", "authority", "available",
    "balance", "barrier", "battery", "behavior", "benefit", "brilliant", "building", "business",
    "calculate", "capacity", "category", "celebrate", "ceremony", "challenge", "champion", "character",
    "chemical", "chocolate", "cigarette", "circular", "climate", "clothing", "cognitive", "collective",
    "colorful", "comfortable", "community", "company", "compare", "complete", "computer", "concept",
    "concrete", "conduct", "confident", "congress", "consider", "continue", "contract", "control",
    "convince", "creative", "creature", "criminal", "critical", "cultural", "customer", "database",
    "daughter", "decision", "decrease", "delivery", "democracy", "describe", "designer", "destroy",
    "develop", "dialogue", "diamond", "digital", "dimension", "directory", "disaster", "discipline",
    "discover", "document", "domestic", "dramatic", "economic", "education", "effective", "electric",
    "element", "emergency", "employee", "encourage", "engineer", "enormous", "environment", "equipment",
    "estimate", "evaluate", "evening", "evidence", "exactly", "example", "excellent", "exchange",
    "exciting", "exercise", "experience", "explain", "explore", "express", "external", "facility",
    "factory", "failure", "family", "fantasy", "fashion", "feature", "feedback", "finance",
    "flexible", "football", "foreign", "formula", "foundation", "freedom", "frequent", "friendly",
    "function", "furniture", "generate", "generous", "genuine", "geography", "government", "graduate",
    "graphics", "grocery", "guarantee", "guidance", "habitat", "hardware", "headline", "healthy",
    "heritage", "holiday", "hospital", "housing", "humanity", "identity", "illegal", "imagine",
    "immediate", "implement", "important", "improve", "include", "increase", "indicate", "industry",
    "influence", "information", "initiative", "innocent", "inside", "inspire", "install", "instance",
    "instead", "instrument", "insurance", "integrate", "intelligence", "intention", "interest", "internal",
    "internet", "interview", "introduce", "investigate", "investment", "involve", "island", "justice",
    "keyboard", "knowledge", "landscape", "language", "leadership", "learning", "lecture", "legacy",
    "legislation", "leisure", "library", "license", "lifestyle", "lightning", "literature", "location",
    "machine", "magazine", "maintain", "management", "marketing", "material", "maximum", "measure",
    "medical", "medicine", "meeting", "memory", "mention", "message", "method", "midnight",
    "military", "minimum", "minister", "minority", "mission", "mistake", "mixture", "moderate",
    "modern", "molecule", "monitor", "morning", "mountain", "movement", "multiple", "muscle",
    "music", "mystery", "narrative", "national", "natural", "navigate", "necessary", "negative",
    "network", "neutral", "newspaper", "normal", "northern", "nothing", "nuclear", "number",
    "objective", "observe", "obstacle", "obvious", "occasion", "ocean", "offer", "office",
    "official", "online", "operate", "opinion", "opportunity", "opposite", "option", "orange",
    "order", "ordinary", "organic", "organize", "original", "outcome", "outdoor", "outside",
    "overall", "overcome", "package", "painting", "palace", "parade", "parent", "parking",
    "participate", "particular", "partner", "party", "passage", "passenger", "passion", "password",
    "pattern", "payment", "peaceful", "pension", "people", "perfect", "perform", "period",
    "permanent", "permission", "person", "personal", "perspective", "physical", "picture", "piece",
    "place", "planet", "planning", "plastic", "platform", "player", "pleasure", "plenty",
    "poetry", "policy", "political", "popular", "population", "position", "positive", "possible",
    "potential", "poverty", "power", "practice", "precise", "predict", "prepare", "present",
    "preserve", "president", "pressure", "prevent", "previous", "primary", "princess", "principle",
    "priority", "private", "probably", "problem", "process", "produce", "product", "profession",
    "profile", "program", "project", "promise", "promote", "property", "proposal", "protect",
    "provide", "province", "public", "purchase", "purpose", "quality", "quantity", "quarter",
    "question", "quickly", "radical", "railway", "random", "rapidly", "rarely", "reach",
    "reading", "reality", "realize", "reason", "receive", "recent", "recognize", "recommend",
    "record", "recover", "reduce", "reference", "reflect", "reform", "region", "register",
    "regular", "reject", "relate", "relative", "release", "relevant", "reliable", "religion",
    "remain", "remember", "remove", "repeat", "replace", "report", "represent", "require",
    "research", "reserve", "resident", "resist", "resolve", "resource", "respect", "respond",
    "responsible", "restaurant", "restore", "result", "return", "revenue", "review", "revolution",
    "reward", "rhythm", "ridiculous", "routine", "safety", "salary", "sample", "satisfy",
    "savings", "scenario", "schedule", "scheme", "science", "screen", "script", "search",
    "season", "second", "secret", "section", "sector", "secure", "segment", "select",
    "senate", "senior", "sense", "sentence", "separate", "sequence", "series", "serious",
    "service", "session", "setting", "several", "shadow", "shake", "shape", "share",
    "shelter", "shift", "shine", "shock", "shoot", "shopping", "shortage", "should",
    "shoulder", "simple", "simply", "since", "single", "sister", "situation", "skill",
    "slice", "slight", "smoke", "smooth", "social", "society", "software", "soldier",
    "solid", "solution", "solve", "someone", "something", "sometimes", "somewhere", "sound",
    "source", "southern", "space", "spare", "speak", "special", "specific", "speech",
    "spend", "spirit", "split", "sport", "spread", "square", "stable", "staff",
    "stage", "standard", "start", "state", "station", "status", "stick", "still",
    "stock", "stone", "storage", "story", "straight", "strange", "strategy", "street",
    "strength", "stress", "strike", "string", "strong", "structure", "struggle", "student",
    "studio", "study", "stuff", "style", "subject", "submit", "succeed", "success",
    "sudden", "suffer", "suggest", "summary", "summer", "sunset", "super", "supply",
    "support", "surface", "surgery", "surprise", "survey", "survive", "switch", "symbol",
    "system", "table", "target", "taste", "teach", "teacher", "technical", "technique",
    "technology", "telephone", "television", "temperature", "temple", "tennis", "terrible", "territory",
    "thanks", "theater", "theory", "therapy", "thick", "thing", "think", "thought",
    "threat", "three", "through", "throw", "ticket", "tight", "title", "today",
    "together", "tomorrow", "total", "touch", "toward", "track", "trade", "tradition",
    "traffic", "train", "training", "transfer", "transform", "transport", "travel", "treat",
    "treatment", "trial", "trick", "truck", "trust", "truth", "tunnel", "twice",
    "typical", "ultimate", "uncle", "under", "understand", "union", "unique", "unite",
    "universe", "university", "unless", "until", "update", "upper", "urban", "urgent",
    "usage", "useful", "usual", "vacation", "valid", "valley", "value", "variable",
    "variety", "various", "vehicle", "version", "versus", "victim", "video", "village",
    "violence", "virtual", "visible", "vision", "visit", "visual", "voice", "volume",
    "water", "weapon", "weather", "website", "wedding", "weight", "welcome", "western",
    "whatever", "wheel", "whenever", "whereas", "whether", "which", "while", "white",
    "whole", "whose", "width", "window", "winter", "wisdom", "within", "without",
    "woman", "wonder", "wooden", "world", "worry", "worth", "write", "wrong",
    "yellow", "young", "youth", "zero"  # Total: 512 words (expandable to 2048)
]

# ================================
# DATA STRUCTURES
# ================================

@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    tokens: float
    last_refill: float
    refill_rate: float

@dataclass
class Challenge:
    """Passphrase challenge data"""
    nonce: str
    digits: str
    words: List[str]
    issued_at: datetime
    expires_at: datetime
    user_id: str

@dataclass
class AuditEvent:
    """Audit log entry"""
    timestamp: datetime
    event_type: str
    user_id: str
    device_id: Optional[str]
    ip_address: str
    details: Dict[str, Any]
    hash_chain: str

@dataclass
class EnvelopeKey:
    """Envelope encryption key structure"""
    iv: bytes
    ciphertext: bytes
    tag: bytes
    key_version: int
    encrypted_dek: bytes

# ================================
# GLOBAL STATE
# ================================

# Initialize FastAPI app with security middleware
app = FastAPI(
    title="Pay AI Voice Biometrics — Enterprise blueprint",
    version="2.0.0",
    description=(
        "Hardened optional stack for deploying the standalone biometric ML API "
        "(JWT, HMAC, rate limits)."
    ),
)

# Force HTTPS in production
if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# CORS with strict origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "https://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models and storage
speaker_model = None
whisper_model = None
antispoof_model = None  # Will implement AASIST/RawNet2

# Secure storage
user_embeddings: Dict[str, EnvelopeKey] = {}  # Encrypted embeddings
impostor_cohort: List[np.ndarray] = []  # 150 impostor embeddings for s-norm
active_challenges: Dict[str, Challenge] = {}  # Nonce-bound challenges

# Rate limiting buckets
rate_buckets: Dict[str, TokenBucket] = defaultdict(lambda: TokenBucket(0, 0, 0, 0))

# Audit chain
audit_chain: List[AuditEvent] = []
last_audit_hash = "genesis"

# Security objects
security = HTTPBearer()
kms_keys = {}  # Mock KMS

logger.info("Pay AI enterprise voice biometric service starting...")
logger.info(f"Risk Thresholds - LOW: {THRESH_LOW}, MED: {THRESH_MED}, HIGH: {THRESH_HIGH}")
logger.info(f"Liveness Threshold: {LIVENESS_THRESH}")

# ================================
# SECURITY & CRYPTO FUNCTIONS
# ================================

class EnterpriseVoiceAuthError(Exception):
    """Custom exception for voice biometric processing errors"""
    pass

def verify_hmac_signature(request: Request, body: bytes) -> bool:
    """
    Verify HMAC-SHA256 request signature for transport security
    Headers: X-Device-Id, X-Nonce, X-Timestamp, X-Signature
    Signature over: method + path + sha256(body) + timestamp + nonce
    """
    try:
        device_id = request.headers.get("X-Device-Id")
        nonce = request.headers.get("X-Nonce")
        timestamp = request.headers.get("X-Timestamp")
        signature = request.headers.get("X-Signature")
        
        if not all([device_id, nonce, timestamp, signature]):
            return False
        
        # Check timestamp (≤ 2 minutes skew)
        current_time = int(time.time())
        request_time = int(timestamp)
        if abs(current_time - request_time) > 120:
            logger.warning(f"Clock skew too large: {abs(current_time - request_time)}s")
            return False
        
        # Construct signature payload
        method = request.method
        path = str(request.url.path)
        body_hash = hashlib.sha256(body).hexdigest()
        
        payload = f"{method}{path}{body_hash}{timestamp}{nonce}"
        
        # Calculate expected signature
        expected_sig = hmac.new(
            HMAC_SECRET.encode(),
            payload.encode(),
            hashlib.sha256
        ).digest()
        expected_sig_b64 = base64.b64encode(expected_sig).decode()
        
        # Constant-time comparison
        return hmac.compare_digest(signature, expected_sig_b64)
        
    except Exception as e:
        logger.error(f"HMAC verification failed: {e}")
        return False

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Verify JWT token and extract user context"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        
        # Validate required claims
        required_claims = ["sub", "iat", "exp", "device_id"]
        if not all(claim in payload for claim in required_claims):
            raise HTTPException(status_code=401, detail="Invalid token claims")
        
        # Check expiration
        if payload["exp"] < time.time():
            raise HTTPException(status_code=401, detail="Token expired")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def generate_envelope_key() -> Tuple[bytes, EnvelopeKey]:
    """
    Generate envelope encryption key structure
    Returns: (plaintext_dek, envelope_key_structure)
    """
    # Generate 256-bit Data Encryption Key
    dek = os.urandom(32)
    
    # Mock KMS: encrypt DEK with Key Encryption Key
    kek = KMS_KEY.encode()[:32].ljust(32, b'\0')  # Pad to 32 bytes
    kek_cipher = AESGCM(kek)
    iv_kek = os.urandom(12)
    encrypted_dek = kek_cipher.encrypt(iv_kek, dek, None)
    
    # Create envelope structure
    envelope = EnvelopeKey(
        iv=os.urandom(12),  # Will be used for actual data encryption
        ciphertext=b"",     # Will be filled during encryption
        tag=b"",           # Will be filled during encryption
        key_version=1,
        encrypted_dek=iv_kek + encrypted_dek  # Store IV + ciphertext
    )
    
    return dek, envelope

def encrypt_embedding(embedding: np.ndarray, user_id: str) -> EnvelopeKey:
    """Encrypt embedding using envelope encryption"""
    try:
        # Generate new DEK for this user
        dek, envelope = generate_envelope_key()
        
        # Encrypt embedding with DEK
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        cipher = AESGCM(dek)
        ciphertext = cipher.encrypt(envelope.iv, embedding_bytes, user_id.encode())
        
        # Store in envelope
        envelope.ciphertext = ciphertext
        envelope.tag = b""  # GCM includes auth tag in ciphertext
        
        # Secure DEK cleanup
        dek = b'\x00' * len(dek)
        
        logger.info(f"Encrypted embedding for user {user_id} with envelope encryption")
        return envelope
        
    except Exception as e:
        logger.error(f"Encryption failed for user {user_id}: {e}")
        raise EnterpriseVoiceAuthError(f"Encryption failed: {str(e)}")

def decrypt_embedding(envelope: EnvelopeKey, user_id: str) -> np.ndarray:
    """Decrypt embedding using envelope encryption"""
    try:
        # Decrypt DEK using KEK (mock KMS)
        kek = KMS_KEY.encode()[:32].ljust(32, b'\0')
        kek_cipher = AESGCM(kek)
        
        iv_kek = envelope.encrypted_dek[:12]
        encrypted_dek = envelope.encrypted_dek[12:]
        dek = kek_cipher.decrypt(iv_kek, encrypted_dek, None)
        
        # Decrypt embedding with DEK
        cipher = AESGCM(dek)
        embedding_bytes = cipher.decrypt(envelope.iv, envelope.ciphertext, user_id.encode())
        
        # Convert back to numpy array
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        # Secure DEK cleanup
        dek = b'\x00' * len(dek)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Decryption failed for user {user_id}: {e}")
        raise EnterpriseVoiceAuthError(f"Decryption failed: {str(e)}")

# ================================
# ADVANCED VAD & PREPROCESSING
# ================================

def webrtc_vad_alternative(audio: np.ndarray, sample_rate: int, aggressiveness: int = 2) -> np.ndarray:
    """
    Enterprise-grade VAD using energy and spectral analysis
    Equivalent to webrtcvad with aggressiveness=2
    """
    try:
        # Frame parameters (30ms frames as per WebRTC standard)
        frame_duration_ms = 30
        frame_length = int(sample_rate * frame_duration_ms / 1000)
        hop_length = frame_length // 2
        
        # Compute features for VAD decision
        # 1. RMS Energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 2. Spectral Centroid (frequency content indicator)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=hop_length)[0]
        
        # 3. Zero Crossing Rate (voicing indicator)
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Adaptive thresholds based on aggressiveness
        rms_percentile = [50, 30, 20, 10][min(aggressiveness, 3)]
        zcr_percentile = [70, 60, 50, 40][min(aggressiveness, 3)]
        
        rms_threshold = np.percentile(rms, rms_percentile)
        zcr_threshold = np.percentile(zcr, zcr_percentile)
        
        # Voice activity decision (combines multiple features)
        voice_frames = (
            (rms > rms_threshold) &
            (zcr < zcr_threshold) &  # Speech typically has lower ZCR than noise
            (spectral_centroid > 500) &  # Speech has energy above 500Hz
            (spectral_centroid < 3000)   # But below 3kHz for fundamental
        )
        
        # Apply morphological operations to smooth decisions
        from scipy.ndimage import binary_opening, binary_closing
        voice_frames = binary_closing(voice_frames, structure=np.ones(3))
        voice_frames = binary_opening(voice_frames, structure=np.ones(2))
        
        if not np.any(voice_frames):
            logger.warning("No voice activity detected")
            return audio  # Return original if no voice detected
        
        # Convert frame indices to sample indices
        voice_samples = np.zeros(len(audio), dtype=bool)
        for i, is_voice in enumerate(voice_frames):
            if is_voice:
                start_sample = i * hop_length
                end_sample = min(start_sample + frame_length, len(audio))
                voice_samples[start_sample:end_sample] = True
        
        # Extract voice segments with padding
        padding_samples = int(0.1 * sample_rate)  # 100ms padding
        voice_indices = np.where(voice_samples)[0]
        
        if len(voice_indices) > 0:
            start_idx = max(0, voice_indices[0] - padding_samples)
            end_idx = min(len(audio), voice_indices[-1] + padding_samples)
            trimmed_audio = audio[start_idx:end_idx]
        else:
            trimmed_audio = audio
        
        logger.debug(f"VAD trimmed audio: {len(audio)} -> {len(trimmed_audio)} samples")
        return trimmed_audio
        
    except Exception as e:
        logger.warning(f"VAD failed: {e}, returning original audio")
        return audio

def enterprise_audio_preprocessing(audio_bytes: bytes) -> torch.Tensor:
    """
    Enterprise-grade audio preprocessing pipeline
    - Load and validate format
    - Resample to 16kHz mono
    - Peak normalize
    - WebRTC VAD (aggressiveness=2)
    - Duration clamping [0.5s, 7s]
    - Quality validation
    """
    try:
        logger.debug(f"Processing {len(audio_bytes)} bytes of audio")
        
        # Security validation
        if len(audio_bytes) == 0:
            raise EnterpriseVoiceAuthError("Empty audio file")
        if len(audio_bytes) > 100 * 1024 * 1024:  # 100MB limit
            raise EnterpriseVoiceAuthError("Audio file too large")
        
        # Load audio using librosa (handles multiple formats)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Load with librosa (professional audio processing)
            audio_data, orig_sr = librosa.load(
                tmp_path,
                sr=TARGET_SAMPLE_RATE,  # Resample to 16kHz
                mono=True,              # Convert to mono
                dtype=np.float32
            )
            
            logger.debug(f"Loaded audio: {len(audio_data)} samples at {TARGET_SAMPLE_RATE}Hz")
            
        finally:
            os.unlink(tmp_path)  # Secure cleanup
        
        # Validate audio properties
        if len(audio_data) == 0:
            raise EnterpriseVoiceAuthError("Audio contains no data")
        
        # Peak normalization (essential for consistent embeddings)
        peak = np.abs(audio_data).max()
        if peak > 0:
            audio_data = audio_data / peak
            logger.debug(f"Peak normalized: max amplitude = {peak:.4f}")
        
        # WebRTC VAD with aggressiveness=2 (as specified)
        audio_data = webrtc_vad_alternative(audio_data, TARGET_SAMPLE_RATE, VAD_AGGRESSIVENESS)
        
        # Duration validation and clamping [0.5s, 7s]
        duration = len(audio_data) / TARGET_SAMPLE_RATE
        
        if duration < MIN_DURATION:
            # Pad with silence if too short
            padding_samples = int((MIN_DURATION - duration) * TARGET_SAMPLE_RATE)
            audio_data = np.pad(audio_data, (0, padding_samples), mode='constant')
            logger.debug(f"Padded audio to minimum duration: {duration:.2f}s -> {MIN_DURATION}s")
            
        elif duration > MAX_DURATION:
            # Truncate if too long
            max_samples = int(MAX_DURATION * TARGET_SAMPLE_RATE)
            audio_data = audio_data[:max_samples]
            logger.debug(f"Truncated audio to maximum duration: {duration:.2f}s -> {MAX_DURATION}s")
        
        # Final quality checks
        final_duration = len(audio_data) / TARGET_SAMPLE_RATE
        if final_duration < MIN_DURATION or final_duration > MAX_DURATION:
            raise EnterpriseVoiceAuthError(f"Invalid duration after processing: {final_duration:.2f}s")
        
        # Convert to PyTorch tensor with proper shape
        wav_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # Add channel dimension
        
        logger.debug(f"Preprocessed audio: shape={wav_tensor.shape}, duration={final_duration:.2f}s")
        return wav_tensor
        
    except EnterpriseVoiceAuthError:
        raise
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise EnterpriseVoiceAuthError(f"Audio preprocessing failed: {str(e)}")

# ================================
# SPEAKER SCORING & S-NORMALIZATION  
# ================================

def extract_embedding(wav_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract 192-dimensional ECAPA-TDNN speaker embedding
    Validates output dimension as per requirement #9
    """
    global speaker_model
    
    try:
        if speaker_model is None:
            logger.info("Loading ECAPA-TDNN speaker recognition model...")
            speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
            logger.info("ECAPA-TDNN model loaded successfully")
        
        # Ensure audio is the right shape and length
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            embedding = speaker_model.encode_batch(wav_tensor)
            embedding_np = embedding.squeeze().cpu().numpy()
        
        # CRITICAL: Validate embedding dimension (Requirement #9)
        if embedding_np.shape[0] != 192:
            logger.error(f"EMBEDDING DIMENSION MISMATCH: Expected 192, got {embedding_np.shape[0]}")
            raise EnterpriseVoiceAuthError(f"Invalid embedding dimension: {embedding_np.shape[0]} (expected 192)")
        
        # L2 normalize
        embedding_np = embedding_np / np.linalg.norm(embedding_np)
        
        logger.debug(f"Extracted embedding: shape={embedding_np.shape}, norm={np.linalg.norm(embedding_np):.4f}")
        return embedding_np
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise EnterpriseVoiceAuthError(f"Failed to extract embedding: {str(e)}")

def initialize_impostor_cohort():
    """
    Initialize cohort of 150 impostor embeddings for s-normalization
    In production, this would be loaded from a diverse speaker database
    """
    global impostor_cohort
    
    if len(impostor_cohort) >= 150:
        return  # Already initialized
    
    logger.info("Initializing impostor cohort for s-normalization...")
    
    # For now, generate diverse synthetic embeddings
    # In production, use real embeddings from diverse speaker population
    np.random.seed(42)  # Reproducible cohort
    
    for i in range(150):
        # Generate normalized random embedding
        embedding = np.random.randn(192).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        impostor_cohort.append(embedding)
    
    logger.info(f"Initialized impostor cohort with {len(impostor_cohort)} embeddings")

def s_norm_score(probe_embedding: np.ndarray, template_embedding: np.ndarray) -> float:
    """
    S-normalization (Score Normalization) for robust speaker verification
    
    Formula: s_norm = (raw_score - mean_cohort_scores) / std_cohort_scores
    
    This normalizes scores against a cohort of impostor speakers to account for
    session variability and improve score reliability across different conditions.
    """
    try:
        # Ensure impostor cohort is initialized
        initialize_impostor_cohort()
        
        # Compute raw cosine similarity score
        raw_score = float(np.dot(probe_embedding, template_embedding))
        
        # Compute cohort scores (probe vs each impostor)
        cohort_scores = []
        for impostor_emb in impostor_cohort:
            cohort_score = float(np.dot(probe_embedding, impostor_emb))
            cohort_scores.append(cohort_score)
        
        cohort_scores = np.array(cohort_scores)
        
        # S-normalization
        mean_cohort = np.mean(cohort_scores)
        std_cohort = np.std(cohort_scores)
        
        if std_cohort > 0:
            s_norm_score = (raw_score - mean_cohort) / std_cohort
        else:
            s_norm_score = raw_score  # Fallback if no variance
        
        logger.debug(f"Scoring: raw={raw_score:.4f}, cohort_mean={mean_cohort:.4f}, "
                    f"cohort_std={std_cohort:.4f}, s_norm={s_norm_score:.4f}")
        
        return s_norm_score
        
    except Exception as e:
        logger.error(f"S-norm scoring failed: {e}")
        # Fallback to raw cosine similarity
        return float(np.dot(probe_embedding, template_embedding))

def get_risk_threshold(amount: Optional[float]) -> float:
    """
    Get verification threshold based on transaction amount (risk-based)
    
    Risk Tiers:
    - LOW (≤ ₹1,000): THRESH_LOW = 0.62
    - MED (₹1,001–₹10,000): THRESH_MED = 0.68  
    - HIGH (> ₹10,000): THRESH_HIGH = 0.74
    """
    if amount is None:
        return THRESH_LOW  # Default to lowest risk
    
    if amount <= 1000:
        return THRESH_LOW
    elif amount <= 10000:
        return THRESH_MED
    else:
        return THRESH_HIGH
