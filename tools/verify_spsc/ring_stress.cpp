// SPSC seqlock ring buffer stress test (see audio_ring_buffer.h contract).
//
// Producer thread pushes 800-sample chunks tagged (epoch_bit << 15 | seq)
// (epoch flips at the mid-run reset); the consumer read_window()s back-to-back
// and verifies EVERY window: leading exact-zeros (a stable refill transient),
// then ONE epoch with consecutive 15-bit seqs. Any torn read -- mixed epochs
// or a broken chain -- FAILs. Zeros are distinguishable because epoch 1 sets
// the high bit and epoch-0 seqs only touch it via exact sample value 0x8000,
// which as a refill prefix still validates (see consumer).
//
//   g++ -std=c++17 -O2 -pthread -Isrc/deployment/esp32 \
//       tools/verify_spsc/ring_stress.cpp -o /tmp/ring_stress && /tmp/ring_stress
//   g++ -std=c++17 -O2 -pthread -fsanitize=thread -Isrc/deployment/esp32 \
//       tools/verify_spsc/ring_stress.cpp -o /tmp/ring_stress_tsan && /tmp/ring_stress_tsan
//
// Both must print PASS. The TSan run proves zero data races (all shared
// state is std::atomic; the seqlock retry makes torn reads impossible).
// NOTE: this tests the HOST build. On Xtensa the same code compiles to
// plain L16I/L32I/S32I + barriers (no CAS loops, no extra RAM).
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "audio_ring_buffer.h"

static constexpr int CHUNK = 800;
static constexpr int N_CHUNKS = 3000; // 150 s of audio @ 16 kHz
static constexpr int RESET_AT = 1500; // producer-side reset() mid-run

static AudioRingBuffer g_ring;
static std::atomic<bool> g_producer_done{false};
static std::atomic<long> g_windows_checked{0};
static std::atomic<long> g_tears{0};
static std::atomic<long> g_resets_seen{0};

static void producer() {
    std::vector<int16_t> chunk(CHUNK);
    uint32_t seq = 0;
    uint32_t epoch = 0;
    for (int n = 0; n < N_CHUNKS; ++n) {
        if (n == RESET_AT) {
            g_ring.reset(); // write-side op from the producer side: legal
            epoch = 1;
        }
        for (int i = 0; i < CHUNK; ++i) {
            uint32_t v = (epoch << 15) | (seq & 0x7FFFu);
            chunk[i] = static_cast<int16_t>(v);
            ++seq;
        }
        g_ring.push(chunk.data(), CHUNK);
        // Paced like deployment (1ms vs real 50ms): without pacing the
        // producer outruns the consumer and no overlap is ever tested.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    g_producer_done.store(true);
}

// A valid full window is always 16000 CONSECUTIVE sequence numbers: is_full()
// flips true exactly when 16000 same-epoch samples are in (initial fill and
// post-reset refill alike), and the seqlock guarantees the copy is atomic.
// Windows read while !full (zero-fill mix) are never validated.
static void consumer() {
    std::vector<float> window(AudioRingBuffer::CAPACITY);
    bool was_full = false;
    while (!g_producer_done.load()) {
        if (!g_ring.is_full()) {
            was_full = false;
            continue;
        }
        if (!was_full) {
            was_full = true;
            g_resets_seen.fetch_add(1); // counts initial fill + refill
        }
        g_ring.read_window(window.data(), window.size());
        g_windows_checked.fetch_add(1);
        // Skip the leading exact-zero run (stable refill transient only).
        size_t i = 0;
        while (i < window.size() && window[i] == 0.0f) {
            ++i;
        }
        // Remainder: single epoch + consecutive 15-bit seqs (16000 < 32768,
        // so no wrap ambiguity inside one window).
        bool torn = false;
        if (i < window.size()) {
            int first = static_cast<int>(std::lround(window[i] * 32768.0f));
            int want_hi = (first >> 15) & 1;
            int prev = first;
            for (++i; i < window.size(); ++i) {
                int cur = static_cast<int>(std::lround(window[i] * 32768.0f));
                int delta = (cur - prev) & 0x7FFF;
                if (((cur >> 15) & 1) != want_hi || delta != 1) {
                    torn = true;
                    break;
                }
                prev = cur;
            }
        }
        if (torn) {
            g_tears.fetch_add(1);
        }
    }
}

int main() {
    std::thread prod(producer);
    std::thread cons(consumer);
    prod.join();
    cons.join();
    long checked = g_windows_checked.load();
    long tears = g_tears.load();
    std::printf("windows_checked=%ld tears=%ld refills=%ld\n",
                checked, tears, g_resets_seen.load());
    bool pass = (tears == 0) && (checked > 1000) && (g_resets_seen.load() >= 2);
    std::printf(pass ? "RING_STRESS: PASS\n" : "RING_STRESS: FAIL\n");
    return pass ? 0 : 1;
}
