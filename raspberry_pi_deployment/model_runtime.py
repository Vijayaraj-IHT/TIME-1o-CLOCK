"""Low-memory TFLite runtime for Raspberry Pi deployment."""

import json
from pathlib import Path

import librosa
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

SAMPLE_RATE = 16000
AUDIO_SAMPLES = 16000
FRAME_SAMPLES = 800
PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_DIR / "voice_activator_int8.tflite"
KEYWORD_DIR = PACKAGE_DIR / "custom_keywords"
CURRENT_KEYWORD_PATH = KEYWORD_DIR / "current_keyword.json"


def load_encoder(num_threads=2):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"TFLite model not found: {MODEL_PATH}")
    interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def load_audio(path):
    audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return standardize_audio(audio, sample_rate)


def standardize_audio(audio, sample_rate=SAMPLE_RATE):
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(audio) < AUDIO_SAMPLES:
        audio = np.pad(audio, (0, AUDIO_SAMPLES - len(audio)))
    return audio[:AUDIO_SAMPLES]


def extract_features(audio):
    mel = librosa.feature.melspectrogram(
        y=standardize_audio(audio), sr=SAMPLE_RATE, n_fft=512,
        hop_length=160, win_length=480, n_mels=40, fmin=20, fmax=4000,
        htk=True, norm="slaney", power=2.0, center=False,
    )
    log_mel = np.log(np.maximum(mel.T, 1e-6))
    n = np.arange(40)
    k = np.arange(13)[:, None]
    dct_basis = np.cos(np.pi * k * (2 * n + 1) / (2 * 40)) * np.sqrt(2.0 / 40)
    dct_basis[0] *= 1.0 / np.sqrt(2.0)
    features = np.dot(log_mel, dct_basis.T).astype(np.float32)
    if features.shape[0] < 98:
        features = np.pad(features, ((0, 98 - features.shape[0]), (0, 0)))
    return features[:98]


def embed_audio(interpreter, audio):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    features = extract_features(audio)
    tensor = np.expand_dims(features, (0, -1)).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details["index"])[0].astype(np.float32)
    return embedding / (np.linalg.norm(embedding) + 1e-9)


def load_current_keyword():
    if not CURRENT_KEYWORD_PATH.exists():
        raise FileNotFoundError(f"No active keyword: {CURRENT_KEYWORD_PATH}")
    with CURRENT_KEYWORD_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    prototype = np.asarray(data["prototype"], dtype=np.float32)
    prototype /= np.linalg.norm(prototype) + 1e-9
    return data["keyword"], prototype


def save_current_keyword(keyword, prototype, sample_paths):
    KEYWORD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "keyword": keyword.upper(),
        "shots": len(sample_paths),
        "prototype": prototype.astype(float).tolist(),
        "samples": [str(path) for path in sample_paths],
    }
    with CURRENT_KEYWORD_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
