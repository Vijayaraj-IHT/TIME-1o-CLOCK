"""
Robustness and False Activation Rate (FAR) Evaluation Suite
SIH Problem Statement 26172 - Milestone 11

Evaluates:
1. False Activation Rate per Hour (FAR/h) over continuous negative speech and noise
2. Signal-to-Noise Ratio (SNR) degradation curve (Clean, 20 dB, 10 dB, 0 dB SNR)
3. Rejection of phonetically similar confuser words (e.g. 'zero', 'four', 'no', 'go')
4. Optimal operating threshold calibration
"""

import numpy as np
import soundfile as sf
from src.streaming.detector import StreamingVoiceActivator

class RobustnessEvaluator:
    def __init__(self, activator: StreamingVoiceActivator):
        self.activator = activator

    @staticmethod
    def mix_noise_at_snr(clean_audio: np.ndarray, noise_audio: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Mixes noise into clean audio at an exact Signal-to-Noise Ratio (in dB).
        SNR = 10 * log10(P_signal / P_noise)
        """
        clean = np.asarray(clean_audio, dtype=np.float32)
        noise = np.asarray(noise_audio, dtype=np.float32)

        # Loop or crop noise to match clean length
        if len(noise) < len(clean):
            repeats = int(np.ceil(len(clean) / len(noise)))
            noise = np.tile(noise, repeats)[:len(clean)]
        else:
            noise = noise[:len(clean)]

        p_signal = np.mean(clean ** 2)
        p_noise = np.mean(noise ** 2)

        if p_signal == 0 or p_noise == 0:
            return clean

        target_p_noise = p_signal / (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(target_p_noise / p_noise)
        mixed = clean + scale * noise
        return np.clip(mixed, -1.0, 1.0)

    def evaluate_snr_robustness(self, target_files: list, noise_files: list,
                                snr_levels=(None, 20, 10, 0), chunk_size_ms=50) -> dict:
        """
        Tests target keyword detection across degraded SNR conditions.
        snr_levels: None = Clean, 20 = 20 dB (mild), 10 = 10 dB (moderate), 0 = 0 dB (severe).
        """
        # Pre-load noise clips
        noise_pool = []
        for nf in noise_files:
            try:
                na, _ = sf.read(nf, dtype='float32')
                if na.ndim > 1:
                    na = na[:, 0]
                noise_pool.append(na)
            except Exception:
                pass
        if not noise_pool:
            noise_pool = [np.random.normal(0, 0.01, 16000).astype(np.float32)]

        results = {}
        total_targets = len(target_files)

        for snr in snr_levels:
            tag = "clean" if snr is None else f"{snr}dB_snr"
            detections = 0
            similarities = []

            for idx, tf in enumerate(target_files):
                clean_audio, _ = sf.read(tf, dtype='float32')
                if clean_audio.ndim > 1:
                    clean_audio = clean_audio.mean(axis=1)  # Fix: was channel-0-only downmix, silently dropped other channels
                if len(clean_audio) < 16000:
                    clean_audio = np.pad(clean_audio, (0, 16000 - len(clean_audio)))
                elif len(clean_audio) > 16000:
                    clean_audio = clean_audio[:16000]

                if snr is not None:
                    noise_sample = noise_pool[idx % len(noise_pool)]
                    test_audio = self.mix_noise_at_snr(clean_audio, noise_sample, snr)
                else:
                    test_audio = clean_audio

                # Wrap in 0.5s silence before and after to simulate isolated utterance
                padded_audio = np.concatenate([
                    np.zeros(8000, dtype=np.float32),
                    test_audio,
                    np.zeros(8000, dtype=np.float32)
                ])

                sim_out = self.activator.simulate_stream(padded_audio, chunk_size_ms=chunk_size_ms)
                if sim_out["total_activations"] >= 1:
                    detections += 1
                    similarities.append(sim_out["activation_events"][0]["smoothed_similarity"])

            tpr = (detections / total_targets * 100.0) if total_targets > 0 else 0.0
            mean_sim = float(np.mean(similarities)) if similarities else 0.0

            results[tag] = {
                "snr_db": snr if snr is not None else "Clean",
                "total_samples": total_targets,
                "successful_detections": detections,
                "tpr_percentage": round(tpr, 2),
                "mean_detection_similarity": round(mean_sim, 4)
            }

        return results

    def evaluate_false_activation_rate(self, negative_audio_files: list, chunk_size_ms=50) -> dict:
        """
        Streams continuous negative speech and ambient noise to measure False Activations per Hour.
        """
        audio_segments = []
        sr = 16000

        for fpath in negative_audio_files:
            try:
                a, _ = sf.read(fpath, dtype='float32')
                if a.ndim > 1:
                    a = a[:, 0]
                if len(a) < 16000:
                    a = np.pad(a, (0, 16000 - len(a)))
                elif len(a) > 16000:
                    a = a[:16000]
                audio_segments.append(a)
            except Exception:
                pass

        if not audio_segments:
            return {"error": "No valid negative audio clips provided"}

        continuous_audio = np.concatenate(audio_segments)
        total_duration_s = len(continuous_audio) / sr
        total_duration_hours = total_duration_s / 3600.0

        sim_out = self.activator.simulate_stream(continuous_audio, chunk_size_ms=chunk_size_ms)
        false_activations = sim_out["total_activations"]
        far_per_hour = (false_activations / total_duration_hours) if total_duration_hours > 0 else 0.0

        return {
            "total_negative_files": len(audio_segments),
            "total_stream_seconds": round(total_duration_s, 2),
            "total_stream_hours": round(total_duration_hours, 4),
            "false_activations_count": false_activations,
            "false_activation_rate_per_hour": round(far_per_hour, 2),
            "vad_inference_skip_percentage": sim_out["vad_skip_percentage"],
            "real_time_factor": sim_out["real_time_factor"]
        }

    def evaluate_confuser_words(self, confuser_dict: dict, chunk_size_ms=50) -> dict:
        """
        Evaluates rejection of phonetically similar confuser words (e.g., 'zero', 'four', 'no', 'go').
        confuser_dict: {word_name: [list of audio filepaths]}
        """
        results = {}
        for word, files in confuser_dict.items():
            triggers = 0
            peak_sims = []

            for fpath in files:
                try:
                    a, _ = sf.read(fpath, dtype='float32')
                    if a.ndim > 1:
                        a = a[:, 0]
                    padded = np.concatenate([np.zeros(4000, dtype=np.float32), a, np.zeros(4000, dtype=np.float32)])
                    out = self.activator.simulate_stream(padded, chunk_size_ms=chunk_size_ms)
                    if out["total_activations"] > 0:
                        triggers += 1
                    max_sim = max([step["smoothed_similarity"] for step in out["telemetry"]]) if out["telemetry"] else 0.0
                    peak_sims.append(max_sim)
                except Exception:
                    pass

            results[word] = {
                "total_tested": len(files),
                "false_triggers": triggers,
                "rejection_rate_pct": round((1.0 - triggers / max(len(files), 1)) * 100.0, 2),
                "peak_similarity_mean": round(float(np.mean(peak_sims)), 4) if peak_sims else 0.0,
                "peak_similarity_max": round(float(np.max(peak_sims)), 4) if peak_sims else 0.0
            }
        return results
