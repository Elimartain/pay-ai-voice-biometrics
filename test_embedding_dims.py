#!/usr/bin/env python3
"""
Test script to check ECAPA-TDNN embedding dimensions
This is CRITICAL for requirement #9 validation
"""

import torch
import numpy as np
from speechbrain.inference.speaker import SpeakerRecognition

def test_ecapa_dimensions():
    """Test ECAPA-TDNN embedding output dimensions"""
    try:
        print("Loading ECAPA-TDNN model...")
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        print("Model loaded successfully!")
        
        # Generate test audio (1 second at 16kHz)
        test_audio = torch.randn(1, 16000)
        print(f"Test audio shape: {test_audio.shape}")
        
        # Extract embedding
        with torch.no_grad():
            embedding = model.encode_batch(test_audio)
            embedding_np = embedding.squeeze().cpu().numpy()
        
        print(f"Embedding shape: {embedding_np.shape}")
        print(f"Embedding dimensions: {embedding_np.shape[0]}")
        
        # Check if it's 192 or 512
        if embedding_np.shape[0] == 192:
            print("✅ ECAPA outputs 192 dimensions - matches requirement")
            return 192
        elif embedding_np.shape[0] == 512:
            print("⚠️  ECAPA outputs 512 dimensions - requirement needs update")
            return 512
        else:
            print(f"❌ Unexpected dimension: {embedding_np.shape[0]}")
            return embedding_np.shape[0]
            
    except Exception as e:
        print(f"❌ Error testing embedding dimensions: {e}")
        return None

if __name__ == "__main__":
    dims = test_ecapa_dimensions()
    print(f"\nFinal result: ECAPA-TDNN outputs {dims} dimensions")
