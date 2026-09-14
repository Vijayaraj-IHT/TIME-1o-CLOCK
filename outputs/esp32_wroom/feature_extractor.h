/*
 * MFCC Feature Extractor for ESP32-S3
 * SIH Problem Statement 26172
 *
 * Computes 13-coefficient MFCCs across 98 temporal frames from 16 kHz audio.
 * Fixed memory layout matching Python MFCCFeatureExtractor (98 frames x 13 coeffs).
 *
 * REWRITTEN (Track A fix). Two independent, previously-verified defects are fixed here:
 *
 * 1. compute_mel_filterbank() never computed a spectrogram at all - it summed
 *    abs(windowed_frame[b]) over raw TIME-DOMAIN samples indexed as if they were
 *    frequency bins, with no FFT anywhere in the class. Replaced with a real
 *    iterative radix-2 FFT + the exact librosa-derived Mel filterbank (see
 *    mel_filterbank.h, exported directly from the same librosa.filters.mel() call
 *    mfcc.py/logmel.py use for training) so this firmware actually computes the same
 *    feature the model was trained on, rather than an unrelated approximation.
 *
 * 2. Pre-emphasis and Hamming window here did not match Python (no pre-emphasis, Hann
 *    window). Since the model was trained on the Python spec, Python is the canonical
 *    reference - firmware is changed to match it, not the reverse. This also removes
 *    the `float preemph[16000]` 64 KB stack-local array entirely (it existed only to
 *    support the pre-emphasis step that Python's pipeline never applied), which was
 *    independently identified as a likely direct cause of the "DRAM exceeded" failures:
 *    a 64 KB automatic array on every ~50ms call will overflow almost any realistic
 *    FreeRTOS task stack.
 *
 * IMPORTANT: this file's output has NOT yet been numerically verified against the
 * Python mfcc.py output on real hardware. See tools/verify_feature_parity/ for the
 * offline (desktop-compiled) verification harness that checks this before trusting
 * it on-device.
 */

#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "mel_filterbank.h"

class EdgeMFCCExtractor {
public:
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int FRAME_LENGTH = 480;   // 30 ms
    static constexpr int HOP_LENGTH = 160;     // 10 ms
    static constexpr int NUM_FRAMES = 98;      // 1.0 sec -> 98 frames
    static constexpr int NUM_MELS = 40;
    static constexpr int NUM_MFCC = 13;
    static constexpr int FFT_SIZE = 512;       // power of 2, required by the radix-2 FFT below
    static constexpr int NUM_FFT_BINS = FFT_SIZE / 2 + 1; // 257, matches np.fft.rfft output length

    static_assert(FFT_SIZE == 512 && (FFT_SIZE & (FFT_SIZE - 1)) == 0,
                  "fft_radix2() requires FFT_SIZE to be a power of two.");
    static_assert(NUM_FFT_BINS == MEL_FILTERBANK_N_BINS, "Mel filterbank bin count mismatch.");
    static_assert(NUM_MELS == MEL_FILTERBANK_N_MELS, "Mel filterbank mel-count mismatch.");

    EdgeMFCCExtractor() {
        init_hann_window();
        init_dct_basis();
        init_fft_twiddles();
        init_bit_reversal();
    }

    // audio_1sec: 16000 raw float samples in [-1, 1] (already normalized by the ring buffer).
    // out_mfcc_matrix: NUM_FRAMES * NUM_MFCC floats, row-major (frame-major), matching the
    // Python (num_frames, n_mfcc) layout used everywhere else in this project.
    void extract_features(const float* audio_1sec, float* out_mfcc_matrix) const {
        // Fix: no pre-emphasis, matching Python's pipeline exactly (mfcc.py/logmel.py never
        // apply one). This also removes the need for any full-length (16000-sample) scratch
        // buffer - each frame is windowed directly out of audio_1sec below.
        for (int f = 0; f < NUM_FRAMES; ++f) {
            int frame_start = f * HOP_LENGTH;

            // Real + imaginary FFT working buffers: member storage (re_scratch_/im_scratch_),
            // reused every frame - zero-padded from FRAME_LENGTH=480 up to FFT_SIZE=512,
            // exactly matching np.fft.rfft(frames, n=self.fft_length).
            std::memset(re_scratch_, 0, sizeof(re_scratch_));
            std::memset(im_scratch_, 0, sizeof(im_scratch_));
            for (int i = 0; i < FRAME_LENGTH; ++i) {
                re_scratch_[i] = audio_1sec[frame_start + i] * hann_window_[i];
            }

            fft_radix2(re_scratch_, im_scratch_);

            // Power spectrum: (re^2 + im^2) / FFT_SIZE, matching
            // power_spectrum = (np.abs(fft_complex) ** 2) / self.fft_length exactly.
            for (int k = 0; k < NUM_FFT_BINS; ++k) {
                power_spectrum_scratch_[k] = (re_scratch_[k] * re_scratch_[k] + im_scratch_[k] * im_scratch_[k]) / static_cast<float>(FFT_SIZE);
            }

            // Real Mel filterbank (exported directly from the same librosa.filters.mel() call
            // used for training) - replaces the old fake time-domain index-sum entirely.
            float mel_energies[NUM_MELS];
            for (int m = 0; m < NUM_MELS; ++m) {
                float energy = 0.0f;
                for (int k = 0; k < NUM_FFT_BINS; ++k) {
                    energy += power_spectrum_scratch_[k] * MEL_FILTERBANK[m][k];
                }
                mel_energies[m] = std::log(std::max(energy, 1e-6f)); // matches np.log(np.maximum(x, 1e-6))
            }

            // DCT-II projection to 13 coefficients (unchanged - this part already matched
            // Python's orthonormal DCT-II construction in mfcc.py).
            for (int c = 0; c < NUM_MFCC; ++c) {
                float sum = 0.0f;
                for (int m = 0; m < NUM_MELS; ++m) {
                    sum += mel_energies[m] * dct_basis_[c][m];
                }
                out_mfcc_matrix[f * NUM_MFCC + c] = sum;
            }
        }
    }

private:
    float hann_window_[FRAME_LENGTH];
    float dct_basis_[NUM_MFCC][NUM_MELS];
    float twiddle_re_[FFT_SIZE / 2];
    float twiddle_im_[FFT_SIZE / 2];
    uint16_t bit_reversal_[FFT_SIZE];
    // Fix: FFT working buffers moved from function-local (stack) to member (static-lifetime)
    // storage. At 512 floats each (2 KB) these were far less dangerous than the old 64 KB
    // preemph[] array, but "smaller stack allocation" isn't "no stack allocation" - this
    // extractor should have zero large automatic arrays, not just no catastrophic ones.
    mutable float re_scratch_[FFT_SIZE];
    mutable float im_scratch_[FFT_SIZE];
    mutable float power_spectrum_scratch_[FFT_SIZE / 2 + 1];

    void init_hann_window() {
        // Fix: matches Python's np.hanning(win_length) exactly (was Hamming).
        const float pi = 3.14159265358979323846f;
        for (int i = 0; i < FRAME_LENGTH; ++i) {
            hann_window_[i] = 0.5f - 0.5f * std::cos((2.0f * pi * i) / (FRAME_LENGTH - 1));
        }
    }

    void init_dct_basis() {
        // Unchanged: already matches mfcc.py's orthonormal DCT-II construction.
        const float pi = 3.14159265358979323846f;
        float factor = std::sqrt(2.0f / NUM_MELS);
        for (int k = 0; k < NUM_MFCC; ++k) {
            for (int n = 0; n < NUM_MELS; ++n) {
                float val = std::cos(pi * k * (2.0f * n + 1.0f) / (2.0f * NUM_MELS)) * factor;
                if (k == 0) {
                    val *= 1.0f / std::sqrt(2.0f);
                }
                dct_basis_[k][n] = val;
            }
        }
    }

    void init_fft_twiddles() {
        const float pi = 3.14159265358979323846f;
        for (int k = 0; k < FFT_SIZE / 2; ++k) {
            float angle = -2.0f * pi * k / FFT_SIZE;
            twiddle_re_[k] = std::cos(angle);
            twiddle_im_[k] = std::sin(angle);
        }
    }

    void init_bit_reversal() {
        int bits = 0;
        for (int n = FFT_SIZE; n > 1; n >>= 1) ++bits;
        for (int i = 0; i < FFT_SIZE; ++i) {
            uint16_t rev = 0;
            int x = i;
            for (int b = 0; b < bits; ++b) {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
            }
            bit_reversal_[i] = rev;
        }
    }

    // In-place iterative radix-2 Cooley-Tukey FFT. Requires FFT_SIZE to be a power of two
    // (enforced by static_assert above). No heap allocation, no recursion, fixed-size
    // twiddle/bit-reversal tables precomputed once in the constructor.
    void fft_radix2(float* re, float* im) const {
        for (int i = 0; i < FFT_SIZE; ++i) {
            int j = bit_reversal_[i];
            if (j > i) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
        for (int size = 2; size <= FFT_SIZE; size <<= 1) {
            int half = size / 2;
            int step = FFT_SIZE / size;
            for (int start = 0; start < FFT_SIZE; start += size) {
                for (int k = 0; k < half; ++k) {
                    float tw_re = twiddle_re_[k * step];
                    float tw_im = twiddle_im_[k * step];
                    int i0 = start + k;
                    int i1 = start + k + half;
                    float t_re = re[i1] * tw_re - im[i1] * tw_im;
                    float t_im = re[i1] * tw_im + im[i1] * tw_re;
                    re[i1] = re[i0] - t_re;
                    im[i1] = im[i0] - t_im;
                    re[i0] += t_re;
                    im[i0] += t_im;
                }
            }
        }
    }
};

#endif // FEATURE_EXTRACTOR_H_
