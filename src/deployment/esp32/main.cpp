/*
 * ESP32-S3 Firmware Entry Point: Low Latency Voice Activator
 * SIH Problem Statement 26172 (ISRO)
 *
 * Features:
 * - INMP441 I2S Digital Microphone Sampling (16 kHz, 16-bit Mono)
 * - Static Audio Ring Buffer (16,000 samples / 32 KB)
 * - Short-Time Energy VAD (bypasses CNN on silence to keep idle CPU < 10%)
 * - Edge MFCC Feature Extractor (98 x 13)
 * - TensorFlow Lite Micro INT8 Inference Engine with ESP-NN Optimizations
 * - Few-Shot Target Prototype Matching ("ZORA")
 * - Dual-Threshold Hysteresis State Machine
 * - Wake-Up GPIO Trigger & Remote ASR Handover Notification
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Local Component Headers
#include "tflite_micro_model.h"
#include "keyword_prototype.h"
#include "audio_ring_buffer.h"
#include "feature_extractor.h"
#include "activator_state_machine.h"

// Configuration Constants
constexpr size_t CHUNK_SIZE = 800; // 50 ms @ 16 kHz
constexpr float VAD_ENERGY_THRESHOLD = 0.015f; // RMS Energy Gate
// VAD-miss fix (Sep-14, measured on host in experiments/vad_gate/): keep
// inferring for 10 chunks past the last speech chunk. Short-keyword
// completion windows fall in post-word silence; without a hangover tail
// the gate skips them and smoothing decays before persistence completes.
// 10 covers clean + noisy streams with zero added FA on negatives.
constexpr int VAD_HANGOVER_CHUNKS = 10; // 10 x 50ms tail after speech
// Fix: models_manifest.json reports tensor_arena_kb=63.0 for voice_activator_int8 -
// the previous 64KB allocation left <2% headroom (framework/graph overhead can push
// this over on a bad day). Bumped with real safety margin.
constexpr size_t TENSOR_ARENA_SIZE = 96 * 1024; // 96 KB Static Arena in SRAM (was 64 KB)

// Fix #1: thresholds synced to configs/config.yaml -> detection block
// (threshold=0.89, hysteresis=0.05 -> tau_low=0.84, consecutive_windows=4,
// smoothing_window=8, cooldown_ms=1500). Previously hardcoded to 0.88/0.84/4,
// which did NOT match what experiments/threshold/*.json calibrated against.
constexpr float TAU_HIGH = 0.89f;
constexpr float TAU_LOW = 0.84f;
constexpr int PERSISTENCE_COUNT = 4;
constexpr int SMOOTHING_WINDOW = 8;
constexpr uint32_t COOLDOWN_MS = 1500;

// Static Memory Allocations (Zero heap allocations during streaming loop)
static alignas(16) uint8_t g_tensor_arena[TENSOR_ARENA_SIZE];
static AudioRingBuffer g_ring_buffer;
static EdgeMFCCExtractor g_mfcc_extractor;
static ActivatorStateMachine g_state_machine(TAU_HIGH, TAU_LOW, PERSISTENCE_COUNT, SMOOTHING_WINDOW, COOLDOWN_MS);

// Fix: setup() previously just printf'd on allocation failure and returned, but
// app_main()'s loop kept calling process_audio_chunk() against a half-initialized
// interpreter afterwards (undefined behavior on a live device). This flag hard-gates
// every audio-processing call until initialization has verifiably succeeded.
static bool g_system_ready = false;

// Global TFLite Micro Pointers
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input_tensor = nullptr;
static TfLiteTensor* g_output_tensor = nullptr;

// Compute RMS Energy for VAD Gating
static float compute_rms_energy(const float* samples, size_t count) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum_sq += samples[i] * samples[i];
    }
    return std::sqrt(sum_sq / static_cast<float>(count));
}

// Compute Cosine Similarity against Enrolled Prototype
static float compute_cosine_similarity(const float* embedding, const float* prototype, int dim) {
    float dot = 0.0f;
    float norm_e = 0.0f;
    float norm_p = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += embedding[i] * prototype[i];
        norm_e += embedding[i] * embedding[i];
        norm_p += prototype[i] * prototype[i];
    }
    float denom = std::sqrt(norm_e) * std::sqrt(norm_p);
    return (denom > 1e-9f) ? (dot / denom) : 0.0f;
}

// System Setup
void setup_voice_activator() {
    printf("================================================================\n");
    printf("SIH 26172: ESP32-S3 VOICE ACTIVATOR FIRMWARE INITIALIZATION\n");
    printf("Target Keyword: %s\n", ENROLLED_KEYWORD_NAME);
    printf("Model Size:     %u bytes (%.2f KB)\n", g_voice_activator_model_data_len, g_voice_activator_model_data_len / 1024.0f);
    printf("Tensor Arena:   %u bytes (%.2f KB)\n", TENSOR_ARENA_SIZE, TENSOR_ARENA_SIZE / 1024.0f);
    printf("================================================================\n");

    // 1. Load TFLite Model
    g_model = tflite::GetModel(g_voice_activator_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[ERROR] Model schema mismatch! Expected %d, got %ld\n", TFLITE_SCHEMA_VERSION, g_model->version());
        return;
    }

    // 2. Register Required Quantized Kernels (ESP-NN Optimized)
    static tflite::MicroMutableOpResolver<7> op_resolver;
    op_resolver.AddConv2D();
    op_resolver.AddRelu();
    op_resolver.AddAveragePool2D();
    op_resolver.AddFullyConnected();
    op_resolver.AddReshape();
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();

    // 3. Instantiate MicroInterpreter
    static tflite::MicroInterpreter static_interpreter(
        g_model,
        op_resolver,
        g_tensor_arena,
        TENSOR_ARENA_SIZE
    );
    g_interpreter = &static_interpreter;

    // 4. Allocate Tensors in SRAM Arena
    TfLiteStatus allocate_status = g_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("[FATAL] AllocateTensors() failed! Check TENSOR_ARENA_SIZE (currently %zu bytes).\n", TENSOR_ARENA_SIZE);
        printf("[FATAL] Halting - refusing to run inference against an uninitialized interpreter.\n");
        g_system_ready = false;
        return; // g_system_ready stays false; process_audio_chunk() will no-op below.
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);
    // Cold-start fix (Sep-14, mirrors host demo_pipeline): pre-fill the ring
    // with silence so is_full() holds from the first chunk -- a keyword in
    // the first second previously never reached inference. Zero windows sit
    // below the VAD gate, so boot stays silent.
    {
        int16_t silence[800] = {};
        for (int i = 0; i < 20; ++i) g_ring_buffer.push(silence, 800);
    }
    g_system_ready = true;

    printf("[INFO] System ready. Listening for keyword '%s'...\n", ENROLLED_KEYWORD_NAME);
}

// Audio Chunk Processing Loop (Called every 50ms from I2S DMA ISR/task)
void process_audio_chunk(const int16_t* pcm_chunk, size_t chunk_len, uint32_t current_time_ms) {
    if (!g_system_ready) {
        // Fix: previously this function ran unconditionally even if setup() failed to
        // allocate tensors, invoking on a null/partial interpreter. Hard no-op instead.
        return;
    }

    // 1. Push new 50ms audio chunk into 1.0s Ring Buffer
    g_ring_buffer.push(pcm_chunk, chunk_len);

    if (!g_ring_buffer.is_full()) {
        return; // Fill initial 1.0s window
    }

    // 2. Read full 1.0s window
    static float s_audio_window[16000];
    g_ring_buffer.read_window(s_audio_window, 16000);

    // 3. Short-Time RMS Energy VAD Gating (+ hangover tail -- see constexpr).
    // Host-mirror semantics: speech reloads the counter; each silent chunk
    // with counter > 0 still runs inference, then the counter decays.
    static int s_vad_hangover = 0;
    float rms = compute_rms_energy(s_audio_window + (16000 - chunk_len), chunk_len);
    bool vad_speech = (rms >= VAD_ENERGY_THRESHOLD);
    if (vad_speech) {
        s_vad_hangover = VAD_HANGOVER_CHUNKS;
    } else if (s_vad_hangover > 0) {
        --s_vad_hangover;
        vad_speech = true;  // tail of a recent speech burst: still infer
    }
    if (!vad_speech) {
        // Ambient silence/noise: skip CNN inference to keep idle CPU < 10%,
        // but still advance the state machine as UNOBSERVED (Change 3a):
        // decay smoothing, no background learning. Same as Python.
        // Change 3d: persistence can complete ON this silence frame (decay
        // edge still above tau_low). Check the return: a missed wake here
        // would be a SILENT miss (keyword-END trigger with ideal latency).
        bool activated_on_silence = g_state_machine.update(0.0f, current_time_ms, false);
        if (activated_on_silence) {
            printf(">>> [ACTIVATION EVENT] Target Keyword '%s' Detected on silence frame! (Time: %lu ms)\n",
                   ENROLLED_KEYWORD_NAME, current_time_ms);
            // Trigger GPIO Interrupt / Handover subsequent audio to Remote ASR
        }
        return;
    }

    // 4. MFCC Feature Extraction (98 x 13)
    static float s_mfcc_matrix[98 * 13];
    g_mfcc_extractor.extract_features(s_audio_window, s_mfcc_matrix);

    // 5. Quantize Input into Model Input Tensor (INT8)
    if (g_input_tensor->type == kTfLiteInt8) {
        float scale = g_input_tensor->params.scale;
        int zero_point = g_input_tensor->params.zero_point;
        int8_t* in_data = g_input_tensor->data.int8;
        for (int i = 0; i < 98 * 13; ++i) {
            int q = static_cast<int>(std::round(s_mfcc_matrix[i] / scale)) + zero_point;
            in_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
        }
    } else {
        memcpy(g_input_tensor->data.f, s_mfcc_matrix, sizeof(s_mfcc_matrix));
    }

    // 6. Invoke TFLite Micro Model
    if (g_interpreter->Invoke() != kTfLiteOk) {
        printf("[WARN] Model invocation error\n");
        return;
    }

    // 7. Extract Output Embedding (Dequantize to float32 if needed)
    float current_embedding[EMBEDDING_DIMENSION];
    if (g_output_tensor->type == kTfLiteInt8) {
        float scale = g_output_tensor->params.scale;
        int zero_point = g_output_tensor->params.zero_point;
        int8_t* out_data = g_output_tensor->data.int8;
        for (int i = 0; i < EMBEDDING_DIMENSION; ++i) {
            current_embedding[i] = (out_data[i] - zero_point) * scale;
        }
    } else {
        memcpy(current_embedding, g_output_tensor->data.f, sizeof(current_embedding));
    }

    // 8. Cosine similarity vs keyword + garbage veto (Phase-1 Change 2)
    float kw_similarity = compute_cosine_similarity(
        current_embedding,
        ENROLLED_KEYWORD_PROTOTYPE,
        EMBEDDING_DIMENSION
    );
    float gb_similarity = compute_cosine_similarity(
        current_embedding,
        GARBAGE_PROTOTYPE,
        EMBEDDING_DIMENSION
    );
    float raw_similarity = ((kw_similarity - gb_similarity) < GARBAGE_MARGIN) ? -1.0f : kw_similarity;

    // 9. Update Hysteresis State Machine
    bool activated = g_state_machine.update(raw_similarity, current_time_ms);

    if (activated) {
        printf(">>> [ACTIVATION EVENT] Target Keyword '%s' Detected! (Sim: %.4f, Time: %lu ms)\n",
               ENROLLED_KEYWORD_NAME, raw_similarity, current_time_ms);
        // Trigger GPIO Interrupt / Handover subsequent audio to Remote ASR
    }
}

// Firmware Entry Point (e.g. app_main in ESP-IDF)
extern "C" void app_main(void) {
    setup_voice_activator();

    // Simulated DMA audio streaming loop
    int16_t dummy_chunk[CHUNK_SIZE] = {0};
    uint32_t simulated_time = 0;

    for (int i = 0; i < 20; ++i) {
        process_audio_chunk(dummy_chunk, CHUNK_SIZE, simulated_time);
        simulated_time += 50;
    }

    printf("[INFO] ESP32-S3 Firmware loop initialized successfully.\n");
}
