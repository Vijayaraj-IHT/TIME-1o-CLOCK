// Offline (desktop) verification harness for the Track A feature-extractor fix.
// Compiles the ACTUAL ESP32 firmware header (feature_extractor.h) unmodified, feeds it
// raw float audio from a binary file, and writes the resulting (98,13) MFCC matrix to
// stdout as raw float32 - to be diffed against Python's mfcc.py output on the same audio.
//
// Usage: ./verify_parity <input_audio_f32.bin> <output_mfcc_f32.bin>
// input_audio_f32.bin: exactly 16000 float32 samples in [-1, 1], little-endian.
// output_mfcc_f32.bin: 98*13 = 1274 float32 values, row-major (frame-major).

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../src/deployment/esp32/feature_extractor.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <input_audio_f32.bin> <output_mfcc_f32.bin>\n", argv[0]);
        return 1;
    }
    std::vector<float> audio(16000);
    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) { std::fprintf(stderr, "Cannot open input %s\n", argv[1]); return 1; }
    size_t n = std::fread(audio.data(), sizeof(float), 16000, fin);
    std::fclose(fin);
    if (n != 16000) { std::fprintf(stderr, "Expected 16000 samples, got %zu\n", n); return 1; }

    static EdgeMFCCExtractor extractor;
    std::vector<float> mfcc(98 * 13);
    extractor.extract_features(audio.data(), mfcc.data());

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::fprintf(stderr, "Cannot open output %s\n", argv[2]); return 1; }
    std::fwrite(mfcc.data(), sizeof(float), mfcc.size(), fout);
    std::fclose(fout);

    std::fprintf(stderr, "Wrote %zu MFCC values to %s\n", mfcc.size(), argv[2]);
    return 0;
}
