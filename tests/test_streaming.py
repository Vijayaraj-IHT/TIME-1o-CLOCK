"""
Unit Tests for Streaming Pipeline & State Machine
SIH Problem Statement 26172 - Milestone 10 Verification
"""

import sys
import unittest
import numpy as np

sys.path.insert(0, r"D:\SIH_Model")
from src.streaming.ring_buffer import AudioRingBuffer
from src.streaming.vad import EnergyVAD
from src.streaming.state_machine import DetectionStateMachine, DetectionState
from src.streaming.detector import StreamingVoiceActivator
from src.models.tiny_cnn import build_tiny_cnn_encoder

class TestStreamingPipeline(unittest.TestCase):

    def test_ring_buffer_chronological_order(self):
        capacity = 100
        rb = AudioRingBuffer(capacity_samples=capacity)

        # Write sequential data 0 to 149 (150 samples total)
        data = np.arange(150, dtype=np.float32)
        chunk_size = 25
        for i in range(0, 150, chunk_size):
            rb.append(data[i:i + chunk_size])

        self.assertTrue(rb.is_full())
        snapshot = rb.get_snapshot()
        self.assertEqual(len(snapshot), capacity)
        # Oldest should be 50, newest should be 149
        expected = np.arange(50, 150, dtype=np.float32)
        np.testing.assert_array_equal(snapshot, expected)

    def test_vad_silence_vs_speech(self):
        vad = EnergyVAD(sample_rate=16000, min_energy_threshold=0.01)

        # 1. Pure silence
        silence = np.zeros(800, dtype=np.float32)
        self.assertFalse(vad.is_speech(silence))

        # 2. Quiet noise (below threshold)
        quiet_noise = np.random.normal(0, 0.001, 800).astype(np.float32)
        self.assertFalse(vad.is_speech(quiet_noise))

        # 3. Active speech signal
        speech_chunk = np.sin(2 * np.pi * 440 * np.linspace(0, 0.05, 800)).astype(np.float32) * 0.3
        self.assertTrue(vad.is_speech(speech_chunk))

    def test_state_machine_hysteresis_and_cooldown(self):
        sm = DetectionStateMachine(
            threshold=0.78,
            hysteresis=0.05,        # tau_low = 0.73
            consecutive_windows=3,
            smoothing_window=3,
            cooldown_ms=1000
        )

        # Step 1: Low similarity (silence) -> LISTENING
        r = sm.process_similarity(0.20, timestamp_ms=0)
        self.assertEqual(r["state"], DetectionState.LISTENING)
        self.assertFalse(r["is_activated"])

        # Step 2: Transient spike that immediately drops -> VERIFYING then resets to LISTENING
        sm.reset()
        r = sm.process_similarity(0.85, timestamp_ms=50)   # spike >= 0.78
        self.assertEqual(r["state"], DetectionState.VERIFYING)
        r = sm.process_similarity(0.50, timestamp_ms=100)  # smoothed drops below 0.73
        self.assertEqual(r["state"], DetectionState.LISTENING)
        self.assertFalse(r["is_activated"])

        # Step 3: Sustained match above threshold for 3 consecutive windows -> ACTIVATED
        sm.reset()
        _ = sm.process_similarity(0.82, timestamp_ms=200)
        _ = sm.process_similarity(0.82, timestamp_ms=250)
        r3 = sm.process_similarity(0.82, timestamp_ms=300)
        self.assertTrue(r3["is_activated"], "Must trigger activation on 3 consecutive confirmed frames")
        self.assertEqual(r3["state"], DetectionState.COOLDOWN)

        # Step 4: Cooldown lockout: high similarity during cooldown must NOT re-trigger
        r_cool = sm.process_similarity(0.95, timestamp_ms=500)
        self.assertFalse(r_cool["is_activated"])
        self.assertEqual(r_cool["state"], DetectionState.COOLDOWN)

        # Step 5: After cooldown period (timestamp > 300 + 1000 = 1300 ms) -> transitions back to LISTENING
        r_after = sm.process_similarity(0.10, timestamp_ms=1350)
        self.assertEqual(r_after["state"], DetectionState.LISTENING)

    def test_streaming_voice_activator_integration(self):
        encoder = build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32)
        proto = np.random.randn(32).astype(np.float32)
        proto /= np.linalg.norm(proto)

        activator = StreamingVoiceActivator(encoder=encoder, target_prototype=proto)

        # Feed 1.5 seconds of synthetic audio in 50ms chunks (800 samples)
        audio = np.random.normal(0, 0.05, 24000).astype(np.float32)
        sim_res = activator.simulate_stream(audio, chunk_size_ms=50)

        self.assertEqual(sim_res["audio_duration_seconds"], 1.5)
        self.assertEqual(sim_res["total_chunks_processed"], 30)
        self.assertIn("real_time_factor", sim_res)
        self.assertIn("vad_skip_percentage", sim_res)
        self.assertLess(sim_res["real_time_factor"], 1.0, "Must be faster than real-time (RTF < 1.0)")

if __name__ == "__main__":
    unittest.main()
