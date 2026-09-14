/*
 * ESP32-WROOM Firmware: Low-Latency Voice Activator with On-Device Enrollment
 * Target: ESP32-WROOM-32 / ESP32 DevKit v1 (Xtensa LX6 dual-core)
 * SIH Problem Statement 26172
 *
 * Features:
 * - Real-time Keyword Spotting using INT8 Universal Metric CNN
 * - ON-DEVICE ENROLLMENT: Change custom keyword using ONLY the ESP32 & Mic!
 *   (Press onboard BOOT button -> Speak new keyword 3 times -> Saved to Flash NVS)
 * - Persistent Storage: Enrolled keywords survive reboots & power cycles via NVS
 * - Zero PC / Re-flashing needed to change keywords!
 */

#include <Arduino.h>
#include <Preferences.h>
#include "driver/i2s.h"

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

// ==============================================================================
// Hardware Pin Definitions (Configured for ESP32-WROOM / ESP32-DevKit v1)
// ==============================================================================
#define I2S_MIC_PORT        I2S_NUM_0
#define I2S_SCK_PIN         14    // BCLK (Serial Clock) -> INMP441 SCK
#define I2S_WS_PIN          15    // WS / LRCL (Word Select) -> INMP441 WS
#define I2S_SD_PIN          32    // SD / DOUT (Serial Data) -> INMP441 SD
#define WAKE_LED_PIN         2    // On-board Blue LED (GPIO 2 on DevKit v1)
#define WAKE_TRIGGER_PIN     4    // External GPIO pulse to wake host MCU / remote ASR
#define ENROLL_BUTTON_PIN    0    // On-board BOOT button (GPIO 0, active LOW)

// ==============================================================================
// Audio & ML Configuration
// ==============================================================================
constexpr size_t SAMPLE_RATE = 16000;
constexpr size_t CHUNK_SIZE = 800;              // 50 ms @ 16 kHz (800 samples)
constexpr float VAD_ENERGY_THRESHOLD = 0.012f;  // RMS Gate to skip CNN on silence
constexpr size_t TENSOR_ARENA_SIZE = 64 * 1024; // 64 KB Static Arena in SRAM

// Static Memory Allocations in SRAM (Zero dynamic heap allocations during loop)
static alignas(16) uint8_t g_tensor_arena[TENSOR_ARENA_SIZE];
static AudioRingBuffer g_ring_buffer;
static EdgeMFCCExtractor g_mfcc_extractor;
static ActivatorStateMachine g_state_machine(0.82f, 0.78f, 3, 4, 1500);

// Active Keyword Prototype Vector (32-D)
static float g_active_prototype[KEYWORD_PROTOTYPE_DIM];
static char g_active_keyword_name[32] = ENROLLED_KEYWORD_NAME;

// Global TFLite Micro Pointers
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input_tensor = nullptr;
static TfLiteTensor* g_output_tensor = nullptr;

// Activation & Button State Tracking
static bool g_led_active = false;
static uint32_t g_activation_time_ms = 0;
static uint32_t g_button_press_start_ms = 0;
static bool g_is_enrolling = false;

// Preferences for Flash NVS Storage
static Preferences g_prefs;

// ==============================================================================
// Helper Functions
// ==============================================================================

// Compute RMS Energy for Voice Activity Detection (VAD) gating
static float compute_rms_energy(const float* samples, size_t count) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum_sq += samples[i] * samples[i];
    }
    return std::sqrt(sum_sq / static_cast<float>(count));
}

// Compute Cosine Similarity between 32-D Embedding and Active Prototype
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

// Extract 32-D Embedding vector from 1.0s audio window using TFLite Micro
static void extract_embedding_from_audio(const float* audio_1sec, float* out_embedding) {
    static float s_mfcc_matrix[98 * 13];
    g_mfcc_extractor.extract_features(audio_1sec, s_mfcc_matrix);

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

    g_interpreter->Invoke();

    if (g_output_tensor->type == kTfLiteInt8) {
        float scale = g_output_tensor->params.scale;
        int zero_point = g_output_tensor->params.zero_point;
        int8_t* out_data = g_output_tensor->data.int8;
        for (int i = 0; i < KEYWORD_PROTOTYPE_DIM; ++i) {
            out_embedding[i] = (out_data[i] - zero_point) * scale;
        }
    } else {
        memcpy(out_embedding, g_output_tensor->data.f, KEYWORD_PROTOTYPE_DIM * sizeof(float));
    }

    // Ensure L2 normalization
    float norm = 0.0f;
    for (int i = 0; i < KEYWORD_PROTOTYPE_DIM; ++i) {
        norm += out_embedding[i] * out_embedding[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (int i = 0; i < KEYWORD_PROTOTYPE_DIM; ++i) {
            out_embedding[i] /= norm;
        }
    }
}

// LED Blink Helper
static void blink_led(int times, int delay_ms) {
    for (int i = 0; i < times; ++i) {
        digitalWrite(WAKE_LED_PIN, HIGH);
        delay(delay_ms);
        digitalWrite(WAKE_LED_PIN, LOW);
        delay(delay_ms);
    }
}

// ==============================================================================
// Persistent Keyword Storage (ESP32 Non-Volatile Flash NVS)
// ==============================================================================
void load_enrolled_keyword_from_nvs() {
    g_prefs.begin("voice_act", true);
    if (g_prefs.isKey("prototype")) {
        size_t len = g_prefs.getBytes("prototype", g_active_prototype, sizeof(g_active_prototype));
        if (len == sizeof(g_active_prototype)) {
            g_prefs.getString("kw_name", g_active_keyword_name, sizeof(g_active_keyword_name));
            Serial.printf("[NVS] Loaded on-device enrolled keyword '%s' from Flash storage!\n", g_active_keyword_name);
            g_prefs.end();
            return;
        }
    }
    g_prefs.end();

    // Default fallback to keyword_prototype.h
    memcpy(g_active_prototype, KEYWORD_PROTOTYPE, sizeof(KEYWORD_PROTOTYPE));
    strncpy(g_active_keyword_name, ENROLLED_KEYWORD_NAME, sizeof(g_active_keyword_name));
    Serial.printf("[INFO] Using firmware default keyword '%s'\n", g_active_keyword_name);
}

void save_enrolled_keyword_to_nvs(const float* prototype, const char* name) {
    g_prefs.begin("voice_act", false);
    g_prefs.putBytes("prototype", prototype, KEYWORD_PROTOTYPE_DIM * sizeof(float));
    g_prefs.putString("kw_name", name);
    g_prefs.end();
    Serial.println("[NVS] Successfully saved new keyword prototype vector to Flash NVS!");
}

void reset_keyword_to_default() {
    g_prefs.begin("voice_act", false);
    g_prefs.clear();
    g_prefs.end();
    memcpy(g_active_prototype, KEYWORD_PROTOTYPE, sizeof(KEYWORD_PROTOTYPE));
    strncpy(g_active_keyword_name, ENROLLED_KEYWORD_NAME, sizeof(g_active_keyword_name));
    Serial.println("\n[NVS] Custom enrollment cleared. Restored firmware default keyword!");
    blink_led(4, 100);
}

// ==============================================================================
// ON-DEVICE KEYWORD ENROLLMENT (Directly via Mic & ESP32)
// ==============================================================================
void perform_on_device_enrollment() {
    g_is_enrolling = true;
    Serial.println("\n================================================================");
    Serial.println(">>> ENTERING ON-DEVICE KEYWORD ENROLLMENT MODE <<<");
    Serial.println("You will be prompted to speak your new keyword 3 TIMES.");
    Serial.println("Wait for LED to light ON, then speak your keyword clearly!");
    Serial.println("================================================================\n");

    blink_led(3, 200);

    constexpr int NUM_SHOTS = 3;
    float shot_embeddings[NUM_SHOTS][KEYWORD_PROTOTYPE_DIM];

    for (int shot = 0; shot < NUM_SHOTS; ++shot) {
        Serial.printf(">>> [SHOT %d of %d] GET READY...\n", shot + 1, NUM_SHOTS);
        delay(1200);

        // Turn on LED to signal "SPEAK NOW"
        digitalWrite(WAKE_LED_PIN, HIGH);
        Serial.printf(">>> [SHOT %d of %d] SPEAK YOUR KEYWORD NOW! (Recording 1.0s)...\n", shot + 1, NUM_SHOTS);

        // Capture 1.0 second (16,000 samples = 20 chunks of 800)
        static float audio_buffer[16000];
        size_t samples_captured = 0;

        while (samples_captured < 16000) {
            int16_t raw_chunk[CHUNK_SIZE];
            size_t bytes_read = 0;
            i2s_read(I2S_MIC_PORT, raw_chunk, sizeof(raw_chunk), &bytes_read, portMAX_DELAY);
            size_t count = bytes_read / sizeof(int16_t);
            for (size_t i = 0; i < count && samples_captured < 16000; ++i) {
                audio_buffer[samples_captured++] = static_cast<float>(raw_chunk[i]) / 32768.0f;
            }
        }

        digitalWrite(WAKE_LED_PIN, LOW);
        Serial.printf(">>> [SHOT %d of %d] Captured! Computing embedding...\n", shot + 1, NUM_SHOTS);

        // Compute 32-D metric embedding
        extract_embedding_from_audio(audio_buffer, shot_embeddings[shot]);

        // Feedback blink
        blink_led(2, 100);
    }

    // Compute Centroid Prototype: c = (e1 + e2 + e3) / 3
    Serial.println("\n>>> Computing optimal prototype centroid on 32-D unit hypersphere...");
    float centroid[KEYWORD_PROTOTYPE_DIM] = {0};
    for (int k = 0; k < KEYWORD_PROTOTYPE_DIM; ++k) {
        float sum = 0.0f;
        for (int s = 0; s < NUM_SHOTS; ++s) {
            sum += shot_embeddings[s][k];
        }
        centroid[k] = sum / static_cast<float>(NUM_SHOTS);
    }

    // L2-Normalize centroid
    float norm = 0.0f;
    for (int k = 0; k < KEYWORD_PROTOTYPE_DIM; ++k) {
        norm += centroid[k] * centroid[k];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (int k = 0; k < KEYWORD_PROTOTYPE_DIM; ++k) {
            centroid[k] /= norm;
        }
    }

    // Compute intra-enrollment consistency
    float dot12 = 0.0f, dot23 = 0.0f;
    for (int k = 0; k < KEYWORD_PROTOTYPE_DIM; ++k) {
        dot12 += shot_embeddings[0][k] * shot_embeddings[1][k];
        dot23 += shot_embeddings[1][k] * shot_embeddings[2][k];
    }
    float avg_sim = (dot12 + dot23) / 2.0f;
    Serial.printf(">>> Enrollment Consistency: %.4f (Higher is better, > 0.70 recommended)\n", avg_sim);

    // Save and activate new prototype
    memcpy(g_active_prototype, centroid, sizeof(g_active_prototype));
    snprintf(g_active_keyword_name, sizeof(g_active_keyword_name), "CUSTOM_%lu", millis() / 1000);
    save_enrolled_keyword_to_nvs(g_active_prototype, g_active_keyword_name);

    Serial.println("\n================================================================");
    Serial.printf(">>> SUCCESS! NEW KEYWORD ENROLLED DIRECTLY ON ESP32!\n");
    Serial.printf(">>> Active Keyword: '%s' (Persistent in Flash NVS)\n", g_active_keyword_name);
    Serial.println(">>> Now returning to live listening mode...");
    Serial.println("================================================================\n");

    g_ring_buffer.reset();
    g_state_machine.reset();

    // Victory celebration blink (5 quick pulses)
    blink_led(6, 80);
    g_is_enrolling = false;
}

// ==============================================================================
// I2S Microphone Initialization
// ==============================================================================
void setup_i2s_microphone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = CHUNK_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN
    };

    esp_err_t err = i2s_driver_install(I2S_MIC_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Failed to install I2S driver: %d\n", err);
        return;
    }

    err = i2s_set_pin(I2S_MIC_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Failed to set I2S pins: %d\n", err);
        return;
    }

    Serial.println("[INFO] I2S INMP441 Microphone Initialized (16 kHz, 16-bit Mono)");
}

// ==============================================================================
// TensorFlow Lite Micro Initialization
// ==============================================================================
void setup_tflite_micro() {
    Serial.println("================================================================");
    Serial.println("SIH 26172: ESP32-WROOM VOICE ACTIVATOR INITIALIZATION");
    Serial.printf("Active Keyword: %s\n", g_active_keyword_name);
    Serial.printf("Model Size:     %u bytes (%.2f KB)\n", g_voice_activator_model_data_len, g_voice_activator_model_data_len / 1024.0f);
    Serial.printf("Tensor Arena:   %u bytes (%.2f KB)\n", TENSOR_ARENA_SIZE, TENSOR_ARENA_SIZE / 1024.0f);
    Serial.println("Tip: Hold BOOT button for 2 seconds to enroll a NEW keyword!");
    Serial.println("================================================================");

    g_model = tflite::GetModel(g_voice_activator_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[ERROR] Model schema mismatch! Expected %d, got %ld\n", TFLITE_SCHEMA_VERSION, g_model->version());
        return;
    }

    static tflite::MicroMutableOpResolver<7> op_resolver;
    op_resolver.AddConv2D();
    op_resolver.AddRelu();
    op_resolver.AddAveragePool2D();
    op_resolver.AddFullyConnected();
    op_resolver.AddReshape();
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        g_model,
        op_resolver,
        g_tensor_arena,
        TENSOR_ARENA_SIZE
    );
    g_interpreter = &static_interpreter;

    TfLiteStatus allocate_status = g_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("[ERROR] AllocateTensors() failed! Increase TENSOR_ARENA_SIZE");
        return;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);

    Serial.printf("[INFO] System ready. Listening for keyword '%s'...\n", g_active_keyword_name);
}

// ==============================================================================
// Audio Chunk Processing Loop (50 ms Step)
// ==============================================================================
void process_audio_chunk(const int16_t* pcm_chunk, size_t chunk_len, uint32_t current_time_ms) {
    if (g_is_enrolling) return;

    g_ring_buffer.push(pcm_chunk, chunk_len);
    if (!g_ring_buffer.is_full()) return;

    static float s_audio_window[16000];
    g_ring_buffer.read_window(s_audio_window, 16000);

    // VAD Gate
    float rms = compute_rms_energy(s_audio_window + (16000 - chunk_len), chunk_len);
    if (rms < VAD_ENERGY_THRESHOLD) return;

    uint32_t t_start = millis();

    // Extract embedding from live audio
    float current_embedding[KEYWORD_PROTOTYPE_DIM];
    extract_embedding_from_audio(s_audio_window, current_embedding);

    // Cosine similarity against ACTIVE prototype (either from NVS or default header)
    float raw_similarity = compute_cosine_similarity(
        current_embedding,
        g_active_prototype,
        KEYWORD_PROTOTYPE_DIM
    );

    uint32_t latency_ms = millis() - t_start;

    // State machine update
    bool activated = g_state_machine.update(raw_similarity, current_time_ms);

    if (activated) {
        Serial.printf("\n>>> ========================================================\n");
        Serial.printf(">>> [ACTIVATION EVENT] Target Keyword '%s' Detected!\n", g_active_keyword_name);
        Serial.printf(">>> Cosine Similarity: %.4f | Inference Latency: %u ms\n", raw_similarity, latency_ms);
        Serial.printf(">>> Triggering Wake GPIO & Dispatching ASR Handover...\n");
        Serial.printf(">>> ========================================================\n\n");

        digitalWrite(WAKE_LED_PIN, HIGH);
        digitalWrite(WAKE_TRIGGER_PIN, HIGH);
        g_led_active = true;
        g_activation_time_ms = current_time_ms;
    }
}

// ==============================================================================
// Button Handling for On-Device Enrollment
// ==============================================================================
void handle_enroll_button() {
    // Check Serial input for "enroll" or "reset"
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.equalsIgnoreCase("enroll")) {
            perform_on_device_enrollment();
            return;
        } else if (cmd.equalsIgnoreCase("reset")) {
            reset_keyword_to_default();
            return;
        }
    }

    // BOOT Button on GPIO 0 (Active LOW with internal pull-up)
    if (digitalRead(ENROLL_BUTTON_PIN) == LOW) {
        if (g_button_press_start_ms == 0) {
            g_button_press_start_ms = millis();
        } else {
            uint32_t press_duration = millis() - g_button_press_start_ms;
            if (press_duration >= 5000) {
                // Held for 5 seconds -> Reset to firmware default keyword
                reset_keyword_to_default();
                g_button_press_start_ms = 0;
                while (digitalRead(ENROLL_BUTTON_PIN) == LOW) delay(10);
            } else if (press_duration >= 1800) {
                // Held for 2 seconds -> Enter On-Device Enrollment Mode
                digitalWrite(WAKE_LED_PIN, HIGH);
                while (digitalRead(ENROLL_BUTTON_PIN) == LOW) delay(10);
                digitalWrite(WAKE_LED_PIN, LOW);
                g_button_press_start_ms = 0;
                perform_on_device_enrollment();
            }
        }
    } else {
        g_button_press_start_ms = 0;
    }
}

// ==============================================================================
// Arduino Setup & Loop
// ==============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    pinMode(WAKE_LED_PIN, OUTPUT);
    pinMode(WAKE_TRIGGER_PIN, OUTPUT);
    pinMode(ENROLL_BUTTON_PIN, INPUT_PULLUP);
    digitalWrite(WAKE_LED_PIN, LOW);
    digitalWrite(WAKE_TRIGGER_PIN, LOW);

    // 1. Load active prototype from NVS Flash (or fallback to header)
    load_enrolled_keyword_from_nvs();

    // 2. Setup Mic and Model
    setup_i2s_microphone();
    setup_tflite_micro();
}

void loop() {
    // 1. Check for button press / Serial command to trigger On-Device Enrollment
    handle_enroll_button();

    // 2. Read 50 ms audio chunk (800 samples) from I2S DMA
    int16_t pcm_chunk[CHUNK_SIZE];
    size_t bytes_read = 0;

    esp_err_t err = i2s_read(I2S_MIC_PORT, pcm_chunk, sizeof(pcm_chunk), &bytes_read, portMAX_DELAY);
    if (err == ESP_OK && bytes_read == sizeof(pcm_chunk)) {
        uint32_t current_time_ms = millis();
        process_audio_chunk(pcm_chunk, CHUNK_SIZE, current_time_ms);
    }

    // 3. LED Cooldown
    if (g_led_active && (millis() - g_activation_time_ms >= 1500)) {
        digitalWrite(WAKE_LED_PIN, LOW);
        digitalWrite(WAKE_TRIGGER_PIN, LOW);
        g_led_active = false;
    }
}
