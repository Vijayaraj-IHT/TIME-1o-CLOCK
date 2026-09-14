/*
 * Audio Ring Buffer for ESP32-S3 Voice Activator
 * SIH Problem Statement 26172
 *
 * Implements a static, zero-allocation circular buffer maintaining a continuous
 * 1.0-second audio window (16,000 samples @ 16 kHz, 16-bit mono) in internal SRAM (32 KB).
 */

#ifndef AUDIO_RING_BUFFER_H_
#define AUDIO_RING_BUFFER_H_

#include <cstdint>
#include <cstddef>
#include <cstring>

class AudioRingBuffer {
public:
    static constexpr size_t CAPACITY = 16000; // 1.0 second @ 16 kHz

    AudioRingBuffer() : write_idx_(0), count_(0) {
        memset(buffer_, 0, sizeof(buffer_));
    }

    void push(const int16_t* data, size_t length) {
        for (size_t i = 0; i < length; ++i) {
            buffer_[write_idx_] = data[i];
            write_idx_ = (write_idx_ + 1) % CAPACITY;
            if (count_ < CAPACITY) {
                count_++;
            }
        }
    }

    void read_window(float* out_buffer, size_t length) const {
        if (length > CAPACITY) {
            length = CAPACITY;
        }
        size_t start_idx = (write_idx_ + CAPACITY - length) % CAPACITY;
        for (size_t i = 0; i < length; ++i) {
            size_t idx = (start_idx + i) % CAPACITY;
            // Normalize 16-bit PCM integer to float [-1.0, 1.0]
            out_buffer[i] = static_cast<float>(buffer_[idx]) / 32768.0f;
        }
    }

    bool is_full() const {
        return count_ >= CAPACITY;
    }

    size_t count() const {
        return count_;
    }

    void reset() {
        write_idx_ = 0;
        count_ = 0;
        memset(buffer_, 0, sizeof(buffer_));
    }

private:
    int16_t buffer_[CAPACITY]; // Exactly 32,000 bytes in static SRAM
    size_t write_idx_;
    size_t count_;
};

#endif // AUDIO_RING_BUFFER_H_
