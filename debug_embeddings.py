#!/usr/bin/env python3
"""
Debug script to analyze real ECAPA-TDNN embeddings and understand why s-norm isn't working
"""

import numpy as np
import torch
import librosa
from speechbrain.inference.speaker import SpeakerRecognition
import tempfile
import os

def analyze_real_embeddings():
    """Analyze properties of real ECAPA-TDNN embeddings"""
    print("🔍 ANALYZING REAL ECAPA-TDNN EMBEDDINGS...")
    
    # Load the same model as main.py
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )
    
    # Generate multiple test embeddings with different synthetic audio
    embeddings = []
    
    for i in range(10):
        # Generate different types of synthetic audio
        if i < 5:
            # White noise (simulating different speakers)
            audio = np.random.randn(16000).astype(np.float32) * 0.1
        else:
            # Sine waves at different frequencies (simulating voice-like signals)
            freq = 100 + i * 50  # Different fundamental frequencies
            t = np.linspace(0, 1, 16000)
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.1
        
        # Normalize
        audio = audio / np.abs(audio).max() if np.abs(audio).max() > 0 else audio
        
        # Convert to tensor
        wav_tensor = torch.from_numpy(audio).unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            embedding = model.encode_batch(wav_tensor)
            embedding_np = embedding.squeeze().cpu().numpy()
            embedding_np = embedding_np / np.linalg.norm(embedding_np)  # L2 normalize
            embeddings.append(embedding_np)
    
    embeddings = np.array(embeddings)
    
    print(f"📊 REAL EMBEDDING ANALYSIS:")
    print(f"Shape: {embeddings.shape}")
    print(f"Mean values across dimensions: {np.mean(embeddings, axis=0)[:10]}...")  # First 10 dims
    print(f"Std values across dimensions: {np.std(embeddings, axis=0)[:10]}...")   # First 10 dims
    print(f"Overall mean: {np.mean(embeddings):.6f}")
    print(f"Overall std: {np.std(embeddings):.6f}")
    print(f"Min value: {np.min(embeddings):.6f}")
    print(f"Max value: {np.max(embeddings):.6f}")
    
    # Check similarity between different embeddings
    print(f"\n🔄 SIMILARITY ANALYSIS:")
    for i in range(min(5, len(embeddings))):
        for j in range(i+1, min(5, len(embeddings))):
            similarity = np.dot(embeddings[i], embeddings[j])
            print(f"Embedding {i} vs {j}: cosine similarity = {similarity:.4f}")
    
    # This will help us understand what realistic embeddings actually look like
    return embeddings

if __name__ == "__main__":
    embeddings = analyze_real_embeddings()
    print(f"\n✅ Analysis complete. Use this data to fix the impostor cohort!")
