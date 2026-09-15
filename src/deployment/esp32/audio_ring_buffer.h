/*
 * Audio Ring Buffer for ESP32-S3 Voice Activator (SPSC seqlock version)
 * SIH Problem Statement 26172
 *
 * Static, zero-allocation circular buffer holding a continuous 1.0-second
 * audio window (16,000 samples @ 16 kHz, 16-bit mono) in internal SRAM (32 KB).
 *
 * CONCURRENCY CONTRACT (fix for the FIXES_APPLIED.md "ring buffer race"):
 * Single producer, single consumer, no mutex, no FreeRTOS dependency:
 *   - WRITE side -- push() / reset(): call from ONE task only (the I2S
 *     producer). A seqlock odd/even sequence guards each write batch.
 *   - READ side -- read_window() / is_full() / count(): call from ONE task
 *     only (the inference consumer). read_window() snapshots the sequence
 *     before/after the copy and retries on mismatch, so a torn read is
 *     impossible: every returned window is a consistent snapshot.
 * All shared state is std::atomic, so there are no data races even across
 * cores (verified: tools/verify_spsc/ring_stress.cpp, incl. ThreadSanitizer).
 * On Xtensa these compile to plain L16I/L32I/S32I + barriers -- no CAS loops,
 * no extra RAM (atomics are the same size as the raw types).
 *
 * reset() is a WRITE-side op: in a split-task deployment, call it from the
 * producer side (or while the producer is stopped). The state machine and the
 * TFLite interpreter stay single-task (inference only) by construction and
 * need no locking -- this ring is the only cross-task handoff.
 */

#ifndef AUDIO_RING_BUFFER_H_
#define AUDIO_RING_BUFFER_H_

#include <atomic>
#include <cstdint>
#include <cstddef>

class AudioRingBuffer {
public:
    static constexpr size_t CAPACITY = 16000; // 1.0 second @ 16 kHz

    AudioRingBuffer() : seq_(0), write_idx_(0), count_(0) {
        for (size_t i = 0; i < CAPACITY; ++i) {
            buffer_[i].store(0, std::memory_order_relaxed);
        }
    }

    // WRITE side (producer task only). Batched under one seqlock critical
    // section: concurrent readers retry instead of seeing a torn window.
    void push(const int16_t* data, size_t length) {
        seq_.fetch_add(1, std::memory_order_relaxed); // -> odd: writing
        size_t w = write_idx_.load(std::memory_order_relaxed);
        size_t c = count_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < length; ++i) {
            buffer_[w].store(data[i], std::memory_order_relaxed);
            w = (w + 1) % CAPACITY;
            if (c < CAPACITY) {
                ++c;
            }
        }
        write_idx_.store(w, std::memory_order_relaxed);
        count_.store(c, std::memory_order_relaxed);
        seq_.fetch_add(1, std::memory_order_release); // -> even: stable
    }

    // READ side (consumer task only). Returns a consistent snapshot: the
    // window is re-read until the sequence is unchanged across the copy.
    // Single-threaded behavior is bit-identical to the pre-seqlock version.
    void read_window(float* out_buffer, size_t length) const {
        if (length > CAPACITY) {
            length = CAPACITY;
        }
        for (;;) {
            uint32_t s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1u) {
                continue; // writer inside; spin (a push is ~800 stores)
            }
            size_t w = write_idx_.load(std::memory_order_relaxed);
            size_t start_idx = (w + CAPACITY - length) % CAPACITY;
            for (size_t i = 0; i < length; ++i) {
                size_t idx = start_idx + i;
                if (idx >= CAPACITY) {
                    idx -= CAPACITY;
                }
                // Normalize 16-bit PCM integer to float [-1.0, 1.0]
                out_buffer[i] =
                    static_cast<float>(buffer_[idx].load(std::memory_order_relaxed))
                    / 32768.0f;
            }
            uint32_t s2 = seq_.load(std::memory_order_acquire);
            if (s1 == s2) {
                return;
            }
            // Writer interleaved: discard the torn copy and retry.
        }
    }

    bool is_full() const {
        return count_.load(std::memory_order_acquire) >= CAPACITY;
    }

    size_t count() const {
        return count_.load(std::memory_order_acquire);
    }

    // WRITE side (producer side only -- see contract above).
    void reset() {
        seq_.fetch_add(1, std::memory_order_relaxed); // -> odd: writing
        for (size_t i = 0; i < CAPACITY; ++i) {
            buffer_[i].store(0, std::memory_order_relaxed);
        }
        write_idx_.store(0, std::memory_order_relaxed);
        count_.store(0, std::memory_order_relaxed);
        seq_.fetch_add(1, std::memory_order_release); // -> even: stable
    }

private:
    // Same 32,000 bytes as before; atomic<> adds no size, only ordering.
    std::atomic<int16_t> buffer_[CAPACITY];
    std::atomic<uint32_t> seq_;
    std::atomic<size_t> write_idx_;
    std::atomic<size_t> count_;
};

#endif // AUDIO_RING_BUFFER_H_
