// Host parity harness for the firmware DTW spotter.
// Reads float32 MFCC frames (T*dim row-major per frame, dim from header),
// runs dtw_spotter.h, prints one line per trigger: <frame_idx> <cost> <consumed>.
// Usage: spotter_harness frames.bin dim
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "dtw_spotter.h"
#include "dtw_template_zora.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s frames.bin dim\n", argv[0]);
        return 2;
    }
    const int dim = std::atoi(argv[2]);
    if (dim != DTW_ZORA_DIM) {
        std::fprintf(stderr, "dim mismatch: file=%d header=%d\n", dim, DTW_ZORA_DIM);
        return 2;
    }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) {
        std::perror("open frames");
        return 2;
    }
    DTWSpotter spot(DTW_ZORA_CENTROID, DTW_ZORA_FRAMES, DTW_ZORA_DIM,
                    DTW_ZORA_THRESHOLD, DTW_ZORA_COOLDOWN_FRAMES,
                    DTW_ZORA_MIN_DURATION_FRAMES);
    float frame[DTW_ZORA_DIM];
    int idx = 0;
    while (std::fread(frame, sizeof(float), (size_t)dim, f) == (size_t)dim) {
        float cost = 0.0f;
        int32_t consumed = 0;
        if (spot.update(frame, &cost, &consumed)) {
            std::printf("%d %.6f %d\n", idx, (double)cost, (int)consumed);
        }
        ++idx;
    }
    std::fclose(f);
    std::fprintf(stderr, "frames=%d triggers=%d\n", idx, spot.trigger_count());
    return 0;
}
