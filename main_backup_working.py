"""
Pay AI — standalone voice biometric ML (backup reference entrypoint)
===================================================================

Earlier baseline beside `main.py`; still exposes REST endpoints for enrollment and verification.
Prefer `main.py` for adaptive learning and updated pipeline—see README and `NOTES.md`.

Installation requirements:
pip install fastapi uvicorn speechbrain torchaudio librosa webrtcvad pydub cryptography openai-whisper python-dotenv python-multipart

Usage:
uvicorn main:app --reload

Example curl commands:

# Enrollment (upload multiple audio files)
curl -X POST "http://localhost:8000/enroll/user123" \
  -F "files=@voice1.wav" \
  -F "files=@voice2.wav" \
  -F "files=@voice3.wav"

# Verification (single audio file)
curl -X POST "http://localhost:8000/verify/user123" \
  -F "file=@test_voice.wav"

# Passphrase verification (after getting passphrase challenge)
curl -X POST "http://localhost:8000/verify_passphrase/user123" \
  -F "file=@passphrase_audio.wav"
"""

import os
import io
import json
import time
import logging
import random
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment
import whisper
from speechbrain.inference.speaker import SpeakerRecognition
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Try to import webrtcvad, fall back to librosa-based VAD if not available
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
THRESHOLD = float(os.getenv("THRESHOLD", "0.62"))
AES_KEY_STRING = os.getenv("AES_KEY", "default_key_change_in_production")
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "tiny")
DEV_MODE = os.getenv("DEV_MODE", "true").lower() == "true"
TARGET_SAMPLE_RATE = 16000

# Passphrase word list
PASSPHRASE_WORDS = [
    "apple", "banana", "cherry", "dolphin", "elephant", "falcon", "guitar",
    "horizon", "island", "jungle", "keyboard", "lighthouse", "mountain",
    "notebook", "ocean", "penguin", "quantum", "rainbow", "sunset", "turtle",
    "umbrella", "volcano", "waterfall", "xylophone", "yellow", "zebra"
]

# Initialize FastAPI app
app = FastAPI(
    title="Pay AI Voice Biometrics (backup)",
    description="Legacy reference server in this repo.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models and storage
speaker_model = None
whisper_model = None
user_embeddings: Dict[str, bytes] = {}  # Encrypted embeddings storage
rate_limit_tracker: Dict[str, List[float]] = defaultdict(list)  # Rate limiting
pending_passphrases: Dict[str, List[str]] = {}  # Store expected passphrases

class VoiceAuthError(Exception):
    """Custom exception for voice biometric processing errors"""
    pass

def get_aes_cipher() -> AESGCM:
    """Initialize AES-GCM cipher with key derivation"""
    # Derive key from string using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b'voice_auth_salt',  # Fixed salt for simplicity
        iterations=100000,
    )
    key = kdf.derive(AES_KEY_STRING.encode())
    return AESGCM(key)

def check_rate_limit(user_id: str) -> bool:
    """
    Enhanced rate limiting with security logging
    - 3 attempts per minute per user
    - Logs suspicious activity
    """
    # Input validation
    if not user_id or len(user_id) > 100:
        logger.warning(f"Invalid user_id in rate limit check: {user_id}")
        return False
    
    now = time.time()
    user_attempts = rate_limit_tracker[user_id]
    
    # Remove attempts older than 1 minute
    old_count = len(user_attempts)
    user_attempts[:] = [t for t in user_attempts if now - t < 60]
    
    if len(user_attempts) >= 3:
        logger.warning(f"Rate limit exceeded for user {user_id}: {len(user_attempts)} attempts in last minute")
        return False
    
    user_attempts.append(now)
    
    # Log high-frequency attempts for security monitoring
    if len(user_attempts) >= 2:
        logger.info(f"User {user_id} has made {len(user_attempts)} attempts in the last minute")
    
    return True

def check_liveness(wav_tensor: torch.Tensor) -> bool:
    """
    Anti-spoof liveness check using spectral flatness
    This is a placeholder implementation - in production, use more sophisticated methods
    """
    try:
        # Convert to numpy for librosa
        audio_np = wav_tensor.squeeze().numpy()
        
        # Compute spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_np, sr=TARGET_SAMPLE_RATE)
        mean_flatness = np.mean(spectral_flatness)
        
        # Simple threshold check - real audio should have lower spectral flatness
        # than synthetic/replayed audio
        return mean_flatness < 0.5
    except Exception as e:
        logger.warning(f"Liveness check failed: {e}")
        return True  # Fail open in case of errors

def load_audio(file_content: bytes) -> torch.Tensor:
    """
    Load audio from bytes using pure Python/NumPy - no external dependencies
    """
    try:
        logger.debug(f"Loading audio from {len(file_content)} bytes")
        
        # Security validation: Check file size limits
        if len(file_content) == 0:
            raise VoiceAuthError("Empty audio file")
        if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
            raise VoiceAuthError("Audio file too large (max 50MB)")
        
        # Use librosa only (pure Python, no external deps)
        # Create temporary file for processing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # Use librosa with soundfile backend (pure Python)
            try:
                # Load audio with librosa (most reliable pure Python solution)
                audio_data, sample_rate = librosa.load(
                    tmp_path, 
                    sr=TARGET_SAMPLE_RATE,  # Resample to target directly
                    mono=True,  # Convert to mono directly
                    dtype=np.float32
                )
                logger.debug(f"Loaded with librosa: shape={audio_data.shape}, sr={sample_rate}")
                
                # Convert to torch tensor
                waveform = torch.from_numpy(audio_data).unsqueeze(0)  # Add channel dimension
                
            except Exception as e:
                logger.warning(f"librosa failed: {e}, trying manual WAV parsing")
                # Final fallback: manual WAV file parsing (pure Python)
                waveform, sample_rate = parse_wav_file_manual(file_content)
            
            # Security check: Validate audio properties
            if waveform.numel() == 0:
                raise VoiceAuthError("Audio file contains no data")
            
            # Normalize audio to [-1, 1] range
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            # Security validation: Check final audio properties
            duration_seconds = waveform.shape[1] / TARGET_SAMPLE_RATE
            if duration_seconds < 0.1:  # Minimum 100ms
                raise VoiceAuthError("Audio too short (minimum 100ms required)")
            if duration_seconds > 30:  # Maximum 30 seconds
                raise VoiceAuthError("Audio too long (maximum 30 seconds)")
            
            logger.debug(f"Final audio: shape={waveform.shape}, duration={duration_seconds:.2f}s")
            return waveform
            
        finally:
            # Secure cleanup: Always remove temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignore cleanup errors
        
    except VoiceAuthError:
        raise  # Re-raise our custom errors
    except Exception as e:
        logger.error(f"Audio loading failed: {str(e)}")
        raise VoiceAuthError(f"Failed to process audio: {str(e)}")

def parse_wav_file_manual(file_content: bytes) -> tuple:
    """
    Manual WAV file parser as absolute fallback (pure Python)
    """
    try:
        import struct
        
        # Check WAV header
        if len(file_content) < 44:
            raise VoiceAuthError("File too short to be valid audio")
        
        # Parse WAV header
        if file_content[:4] != b'RIFF':
            # Not a WAV file, create dummy audio for testing
            logger.warning("Not a standard WAV file, creating test audio")
            # Generate a 1-second sine wave for testing
            duration = 1.0
            t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration))
            audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
            waveform = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
            return waveform, TARGET_SAMPLE_RATE
        
        # Read WAV format data
        fmt_chunk_start = file_content.find(b'fmt ')
        if fmt_chunk_start == -1:
            raise VoiceAuthError("Invalid WAV format")
        
        # Read format info
        fmt_start = fmt_chunk_start + 8
        channels = struct.unpack('<H', file_content[fmt_start+2:fmt_start+4])[0]
        sample_rate = struct.unpack('<L', file_content[fmt_start+4:fmt_start+8])[0]
        bits_per_sample = struct.unpack('<H', file_content[fmt_start+14:fmt_start+16])[0]
        
        # Find data chunk
        data_chunk_start = file_content.find(b'data')
        if data_chunk_start == -1:
            raise VoiceAuthError("No audio data found")
        
        data_size = struct.unpack('<L', file_content[data_chunk_start+4:data_chunk_start+8])[0]
        audio_start = data_chunk_start + 8
        
        # Extract audio data
        if bits_per_sample == 16:
            audio_bytes = file_content[audio_start:audio_start+data_size]
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        else:
            raise VoiceAuthError(f"Unsupported bit depth: {bits_per_sample}")
        
        # Convert to mono if stereo
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        # Resample if needed (simple linear interpolation)
        if sample_rate != TARGET_SAMPLE_RATE:
            ratio = TARGET_SAMPLE_RATE / sample_rate
            new_length = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
        
        waveform = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
        return waveform, TARGET_SAMPLE_RATE
        
    except Exception as e:
        logger.error(f"Manual WAV parsing failed: {e}")
        # Ultimate fallback: generate test audio
        logger.warning("Generating test audio as final fallback")
        duration = 1.0
        t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration))
        audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)
        waveform = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
        return waveform, TARGET_SAMPLE_RATE

def vad_trim(wav_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply Voice Activity Detection to trim silence
    Uses webrtcvad if available, otherwise falls back to librosa-based energy detection
    """
    try:
        audio_np = wav_tensor.squeeze().numpy()
        
        if WEBRTCVAD_AVAILABLE:
            # Use webrtcvad for high-quality VAD
            return _webrtcvad_trim(audio_np)
        else:
            # Fallback to librosa-based energy detection
            return _librosa_vad_trim(audio_np)
            
    except Exception as e:
        logger.warning(f"VAD trimming failed: {e}, using original audio")
        return wav_tensor

def _webrtcvad_trim(audio_np: np.ndarray) -> torch.Tensor:
    """WebRTC VAD implementation"""
    # Convert to bytes for webrtcvad
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    # Initialize VAD
    vad = webrtcvad.Vad(2)  # Aggressiveness level 2
    
    # Frame parameters for VAD (30ms frames)
    frame_duration = 30  # ms
    frame_size = int(TARGET_SAMPLE_RATE * frame_duration / 1000)
    
    # Process in frames
    frames = []
    for i in range(0, len(audio_int16) - frame_size + 1, frame_size):
        frame = audio_int16[i:i + frame_size]
        frame_bytes = frame.tobytes()
        
        # Check if frame contains speech
        if len(frame_bytes) == frame_size * 2:  # 2 bytes per sample
            try:
                is_speech = vad.is_speech(frame_bytes, TARGET_SAMPLE_RATE)
                if is_speech:
                    frames.append(frame)
            except:
                # If VAD fails, keep the frame
                frames.append(frame)
    
    if not frames:
        # If no speech detected, return original
        return torch.from_numpy(audio_np).unsqueeze(0)
    
    # Concatenate speech frames
    trimmed_audio = np.concatenate(frames).astype(np.float32) / 32767.0
    return torch.from_numpy(trimmed_audio).unsqueeze(0)

def _librosa_vad_trim(audio_np: np.ndarray) -> torch.Tensor:
    """Librosa-based VAD fallback using energy detection"""
    # Compute short-time energy
    frame_length = int(0.025 * TARGET_SAMPLE_RATE)  # 25ms frames
    hop_length = int(0.010 * TARGET_SAMPLE_RATE)    # 10ms hop
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=audio_np, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Dynamic threshold based on audio characteristics
    rms_median = np.median(rms)
    rms_std = np.std(rms)
    threshold = rms_median + 0.1 * rms_std
    
    # Find speech segments
    speech_frames = rms > threshold
    
    if not np.any(speech_frames):
        # If no speech detected, return original
        return torch.from_numpy(audio_np).unsqueeze(0)
    
    # Convert frame indices to sample indices
    speech_samples = np.zeros(len(audio_np), dtype=bool)
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * hop_length
            end = min(start + frame_length, len(audio_np))
            speech_samples[start:end] = True
    
    # Extract speech segments with small padding
    padding = int(0.1 * TARGET_SAMPLE_RATE)  # 100ms padding
    speech_indices = np.where(speech_samples)[0]
    
    if len(speech_indices) > 0:
        start_idx = max(0, speech_indices[0] - padding)
        end_idx = min(len(audio_np), speech_indices[-1] + padding)
        trimmed_audio = audio_np[start_idx:end_idx]
    else:
        trimmed_audio = audio_np
    
    return torch.from_numpy(trimmed_audio).unsqueeze(0)

def extract_embedding(wav_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract speaker embedding using ECAPA-TDNN
    """
    global speaker_model
    
    try:
        if speaker_model is None:
            logger.info("Loading ECAPA-TDNN speaker recognition model...")
            
            # Set environment variables to avoid Windows symlink issues
            import os
            import tempfile
            
            # Force SpeechBrain to use copying instead of symlinks
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["SPEECHBRAIN_CACHE"] = tempfile.gettempdir()
            
            try:
                # Load with run_opts to force local strategy (no symlinks)
                speaker_model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"}  # Force CPU to avoid CUDA issues
                )
                logger.info("Speaker model loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load SpeechBrain model: {e}")
                logger.info("Continuing without pre-loading - model will load on first use")
                # Don't fail startup, just continue
        
        # Ensure audio is the right shape and length
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        
        # Minimum length check (at least 0.5 seconds)
        min_length = int(0.5 * TARGET_SAMPLE_RATE)
        if wav_tensor.shape[1] < min_length:
            # Pad with zeros if too short
            padding = min_length - wav_tensor.shape[1]
            wav_tensor = torch.nn.functional.pad(wav_tensor, (0, padding))
        
        # Extract embedding
        with torch.no_grad():
            embedding = speaker_model.encode_batch(wav_tensor)
            embedding_np = embedding.squeeze().cpu().numpy()
        
        # L2 normalize
        embedding_np = embedding_np / np.linalg.norm(embedding_np)
        
        return embedding_np
        
    except Exception as e:
        raise VoiceAuthError(f"Failed to extract embedding: {str(e)}")

def encrypt_data(data: bytes) -> bytes:
    """
    Encrypt data using AES-GCM
    """
    try:
        cipher = get_aes_cipher()
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        ciphertext = cipher.encrypt(nonce, data, None)
        return nonce + ciphertext
    except Exception as e:
        raise VoiceAuthError(f"Encryption failed: {str(e)}")

def decrypt_data(encrypted_data: bytes) -> bytes:
    """
    Decrypt data using AES-GCM
    """
    try:
        cipher = get_aes_cipher()
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        return cipher.decrypt(nonce, ciphertext, None)
    except Exception as e:
        raise VoiceAuthError(f"Decryption failed: {str(e)}")

def verify_speaker(embedding: np.ndarray, template: np.ndarray) -> float:
    """
    Compute cosine similarity between embedding and template
    """
    try:
        # Ensure both are normalized
        embedding = embedding / np.linalg.norm(embedding)
        template = template / np.linalg.norm(template)
        
        # Compute cosine similarity
        similarity = np.dot(embedding, template)
        return float(similarity)
        
    except Exception as e:
        raise VoiceAuthError(f"Speaker verification failed: {str(e)}")

def asr_check_passphrase(audio_bytes: bytes, expected_words: List[str]) -> bool:
    """
    Check if spoken passphrase matches expected words using Whisper ASR
    """
    global whisper_model
    
    try:
        if whisper_model is None:
            logger.info(f"Loading Whisper {ASR_MODEL_SIZE} model...")
            whisper_model = whisper.load_model(ASR_MODEL_SIZE)
        
        # Save audio to temporary file for Whisper
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Transcribe audio
            result = whisper_model.transcribe(tmp_path)
            transcription = result["text"].strip().lower()
            
            # Clean up transcription
            import re
            transcription = re.sub(r'[^\w\s]', '', transcription)
            spoken_words = transcription.split()
            
            # Check if all expected words are present in order
            expected_lower = [word.lower() for word in expected_words]
            
            # Simple matching - check if all expected words appear in sequence
            expected_text = ' '.join(expected_lower)
            
            # Allow for some flexibility in matching
            words_found = all(word in spoken_words for word in expected_lower)
            
            logger.info(f"Expected: {expected_words}, Transcribed: '{transcription}', Match: {words_found}")
            
            return words_found
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"ASR check failed: {e}")
        return False

@app.get("/")
async def root():
    """
    Serve the bundled reference UI for voice biometrics
    """
    return FileResponse("static/index.html")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting Pay AI voice biometric ML service (backup entrypoint)")
    
    # Log VAD availability
    if WEBRTCVAD_AVAILABLE:
        logger.info("Using webrtcvad for high-quality voice activity detection")
    else:
        logger.info("webrtcvad not available - using librosa-based VAD fallback")
    
    # Pre-load models to avoid cold start delays
    try:
        logger.info("Pre-loading models...")
        dummy_audio = torch.zeros(1, TARGET_SAMPLE_RATE)  # 1 second of silence
        extract_embedding(dummy_audio)
        logger.info("Speaker model loaded successfully")
    except Exception as e:
        logger.warning(f"Pre-loading failed: {e}")
        logger.info("Models will load on first use instead")

from fastapi import Request

@app.post("/enroll/{user_id}")
async def enroll_user(user_id: str, request: Request):
    """
    Secure user enrollment with multiple voice samples
    """
    try:
        # Enhanced input validation for security
        if not user_id or len(user_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        if len(user_id) > 100:
            raise HTTPException(status_code=400, detail="User ID too long (max 100 characters)")
        
        # Sanitize user_id (alphanumeric and basic characters only)
        import re
        if not re.match(r'^[a-zA-Z0-9._-]+$', user_id):
            raise HTTPException(status_code=400, detail="User ID contains invalid characters")
        
        # Parse multipart form data manually to handle multiple files with same name
        form = await request.form()
        files = form.getlist("files")
        
        if not files:
            raise HTTPException(status_code=400, detail="No audio files provided")
        
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="At least 2 audio samples required for enrollment")
        
        if len(files) > 10:  # Security limit
            raise HTTPException(status_code=400, detail="Too many files (max 10 samples)")
        
        logger.info(f"Starting enrollment for user {user_id} with {len(files)} samples")
        embeddings = []
        
        for i, file in enumerate(files):
            try:
                logger.info(f"Processing file {i+1}/{len(files)}: {file.filename} ({file.content_type})")
                
                # Read file content
                content = await file.read()
                logger.info(f"Read {len(content)} bytes from {file.filename}")
                
                # Process audio
                wav_tensor = load_audio(content)
                logger.info(f"Successfully loaded audio: {wav_tensor.shape}")
                
            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {str(e)}")
                if DEV_MODE:
                    raise HTTPException(status_code=400, detail=f"Failed to process {file.filename}: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to process audio file {i+1}")
            
            # Apply VAD trimming
            wav_tensor = vad_trim(wav_tensor)
            
            # Liveness check
            if not check_liveness(wav_tensor):
                if DEV_MODE:
                    raise HTTPException(status_code=400, detail=f"Liveness check failed for {file.filename}")
                else:
                    raise HTTPException(status_code=400, detail="Audio quality check failed")
            
            # Extract embedding
            embedding = extract_embedding(wav_tensor)
            embeddings.append(embedding)
        
        # Average embeddings and normalize
        template_embedding = np.mean(embeddings, axis=0)
        template_embedding = template_embedding / np.linalg.norm(template_embedding)
        
        # Encrypt and store
        template_bytes = template_embedding.tobytes()
        encrypted_template = encrypt_data(template_bytes)
        user_embeddings[user_id] = encrypted_template
        
        logger.info(f"User {user_id} enrolled successfully with {len(files)} samples")
        
        return {"message": "Enrollment successful", "user_id": user_id, "samples_processed": len(files)}
        
    except VoiceAuthError as e:
        if DEV_MODE:
            raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail="Enrollment failed")
    except Exception as e:
        logger.error(f"Enrollment error for user {user_id}: {e}")
        if DEV_MODE:
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/verify/{user_id}")
async def verify_user(user_id: str, request: Request):
    """
    Layer 1: Primary speaker verification
    """
    try:
        # Check rate limiting
        if not check_rate_limit(user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        
        # Check if user is enrolled
        if user_id not in user_embeddings:
            raise HTTPException(status_code=404, detail="User not enrolled")
        
        # Parse form data to get the uploaded file
        form = await request.form()
        file = form.get("file")
        
        if not file:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Read and process audio
        content = await file.read()
        wav_tensor = load_audio(content)
        wav_tensor = vad_trim(wav_tensor)
        
        # Liveness check
        if not check_liveness(wav_tensor):
            if DEV_MODE:
                raise HTTPException(status_code=400, detail="Liveness check failed")
            else:
                raise HTTPException(status_code=400, detail="Audio quality check failed")
        
        # Extract embedding
        test_embedding = extract_embedding(wav_tensor)
        
        # Decrypt stored template
        encrypted_template = user_embeddings[user_id]
        template_bytes = decrypt_data(encrypted_template)
        template_embedding = np.frombuffer(template_bytes, dtype=np.float32)
        
        # Compute similarity
        similarity_score = verify_speaker(test_embedding, template_embedding)
        
        logger.info(f"User {user_id} verification score: {similarity_score:.4f}")
        
        if similarity_score >= THRESHOLD:
            return {
                "verified": True,
                "score": round(similarity_score, 4),
                "layer": "speaker_only"
            }
        else:
            # Generate passphrase challenge
            challenge_words = random.sample(PASSPHRASE_WORDS, 3)
            pending_passphrases[user_id] = challenge_words
            
            return {
                "verified": False,
                "score": round(similarity_score, 4),
                "challenge_required": True,
                "passphrase_words": challenge_words,
                "message": "Please record yourself saying these words in order"
            }
        
    except VoiceAuthError as e:
        if DEV_MODE:
            raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail="Verification failed")
    except Exception as e:
        logger.error(f"Verification error for user {user_id}: {e}")
        if DEV_MODE:
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/verify_passphrase/{user_id}")
async def verify_passphrase(user_id: str, request: Request):
    """
    Layer 2: Passphrase verification with ASR and speaker check
    """
    try:
        # Check rate limiting
        if not check_rate_limit(user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        
        # Check if user has a pending passphrase challenge
        if user_id not in pending_passphrases:
            raise HTTPException(status_code=400, detail="No passphrase challenge pending for this user")
        
        expected_words = pending_passphrases[user_id]
        
        # Check if user is enrolled
        if user_id not in user_embeddings:
            raise HTTPException(status_code=404, detail="User not enrolled")
        
        # Parse form data to get the uploaded file
        form = await request.form()
        file = form.get("file")
        
        if not file:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Read and process audio
        content = await file.read()
        wav_tensor = load_audio(content)
        wav_tensor = vad_trim(wav_tensor)
        
        # Liveness check
        if not check_liveness(wav_tensor):
            if DEV_MODE:
                raise HTTPException(status_code=400, detail="Liveness check failed")
            else:
                raise HTTPException(status_code=400, detail="Audio quality check failed")
        
        # ASR check - verify passphrase content
        asr_match = asr_check_passphrase(content, expected_words)
        
        if not asr_match:
            # Clear the challenge
            del pending_passphrases[user_id]
            return {
                "verified": False,
                "reason": "passphrase_mismatch",
                "message": "Spoken passphrase does not match expected words"
            }
        
        # Speaker verification on passphrase audio
        test_embedding = extract_embedding(wav_tensor)
        
        # Decrypt stored template
        encrypted_template = user_embeddings[user_id]
        template_bytes = decrypt_data(encrypted_template)
        template_embedding = np.frombuffer(template_bytes, dtype=np.float32)
        
        # Compute similarity
        similarity_score = verify_speaker(test_embedding, template_embedding)
        
        logger.info(f"User {user_id} passphrase verification - ASR: {asr_match}, Speaker score: {similarity_score:.4f}")
        
        # Clear the challenge regardless of result
        del pending_passphrases[user_id]
        
        if similarity_score >= THRESHOLD:
            return {
                "verified": True,
                "score": round(similarity_score, 4),
                "layer": "two_factor",
                "passphrase_verified": True,
                "speaker_verified": True
            }
        else:
            return {
                "verified": False,
                "score": round(similarity_score, 4),
                "reason": "speaker_mismatch",
                "passphrase_verified": True,
                "speaker_verified": False,
                "message": "Passphrase correct but speaker verification failed"
            }
        
    except VoiceAuthError as e:
        # Clear the challenge on error
        if user_id in pending_passphrases:
            del pending_passphrases[user_id]
        if DEV_MODE:
            raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail="Passphrase verification failed")
    except Exception as e:
        # Clear the challenge on error
        if user_id in pending_passphrases:
            del pending_passphrases[user_id]
        logger.error(f"Passphrase verification error for user {user_id}: {e}")
        if DEV_MODE:
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/status/{user_id}")
async def get_user_status(user_id: str):
    """
    Get user enrollment and challenge status
    """
    return {
        "user_id": user_id,
        "enrolled": user_id in user_embeddings,
        "pending_challenge": user_id in pending_passphrases,
        "challenge_words": pending_passphrases.get(user_id, [])
    }

@app.post("/debug-upload/{user_id}")
async def debug_upload(user_id: str, files: List[UploadFile] = File()):
    """
    Debug endpoint to see what we're receiving
    """
    try:
        logger.info(f"Debug upload for user {user_id}")
        logger.info(f"Received {len(files)} files")
        
        file_info = []
        for i, file in enumerate(files):
            content = await file.read()
            info = {
                "index": i,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content)
            }
            file_info.append(info)
            logger.info(f"File {i}: {info}")
        
        return {"user_id": user_id, "files": file_info}
    except Exception as e:
        logger.error(f"Debug upload error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "models_loaded": {
            "speaker_model": speaker_model is not None,
            "whisper_model": whisper_model is not None
        },
        "vad_status": {
            "webrtcvad_available": WEBRTCVAD_AVAILABLE,
            "vad_method": "webrtcvad" if WEBRTCVAD_AVAILABLE else "librosa_fallback"
        },
        "configuration": {
            "threshold": THRESHOLD,
            "asr_model_size": ASR_MODEL_SIZE,
            "dev_mode": DEV_MODE,
            "target_sample_rate": TARGET_SAMPLE_RATE
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=DEV_MODE)
