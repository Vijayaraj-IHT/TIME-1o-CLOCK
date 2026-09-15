// Track-B streaming subsequence-DTW spotter (firmware port of src/dtw/spot.py).
// O(T) per 10ms frame, float32, zero allocation after construction.
// Template (centroid + operating point) comes from a generated header such
// as dtw_template_zora.h. Parity-checked against the host on a recorded
// stream: see tools/verify_dtw_parity/.
//
// Resource note (T=98, dim=13): template 5096 B flash; working RAM two
// column sets x 99 x (4B cost + 4B steps + 4B frames) = 2376 B. steps_ and
// frames_ are int32, NOT int16: the consumed-frames counter grows with the
// stream between triggers (hours of silence), so int16 would wrap; int32
// costs 400 B and never wraps in practice.
#ifndef DTW_SPOTTER_H_
#define DTW_SPOTTER_H_

#include <float.h>
#include <stddef.h>
#include <stdint.h>

class DTWSpotter {
 public:
    // centroid: T*dim row-major floats. threshold/cooldown/min_duration: the
    // calibrated operating point (defines in the template header).
    DTWSpotter(const float* centroid, int frames, int dim, float threshold,
               int cooldown_frames, int min_duration_frames)
        : centroid_(centroid),
          frames_(frames),
          dim_(dim),
          threshold_(threshold),
          cooldown_frames_(cooldown_frames),
          min_duration_frames_(min_duration_frames),
          cooldown_left_(0),
          n_triggers_(0) {
        reset_columns();
    }

    void reset() {
        reset_columns();
        cooldown_left_ = 0;
    }

    int trigger_count() const { return n_triggers_; }

    // Advance one MFCC frame. Returns true on trigger; out_cost_per_step and
    // out_frames_consumed report the firing path (valid only if triggered).
    // Tie-break order MUST match the host (ci, then cj, else cm) or the
    // argmin paths (and hence costs) diverge on near-ties.
    bool update(const float* mfcc_frame, float* out_cost_per_step,
                int32_t* out_frames_consumed) {
        new_cost_[0] = 0.0f;  // open begin, every frame
        new_steps_[0] = 0;
        new_frames_[0] = 0;
        for (int i = 1; i <= frames_; ++i) {
            float d = 0.0f;
            const float* row = centroid_ + (i - 1) * dim_;
            for (int k = 0; k < dim_; ++k) {
                float diff = row[k] - mfcc_frame[k];
                d += diff * diff;
            }
            float ci = cost_[i];          // (i,j-1): stream advances
            float cj = new_cost_[i - 1];  // (i-1,j): stream stays
            float cm = cost_[i - 1];      // (i-1,j-1): both advance
            if (ci <= cj && ci <= cm) {
                new_cost_[i] = d + ci;
                new_steps_[i] = steps_[i] + 1;
                new_frames_[i] = frames_consumed_[i] + 1;
            } else if (cj <= cm) {
                new_cost_[i] = d + cj;
                new_steps_[i] = new_steps_[i - 1] + 1;
                new_frames_[i] = new_frames_[i - 1];
            } else {
                new_cost_[i] = d + cm;
                new_steps_[i] = steps_[i - 1] + 1;
                new_frames_[i] = frames_consumed_[i - 1] + 1;
            }
        }
        for (int i = 0; i <= frames_; ++i) {
            cost_[i] = new_cost_[i];
            steps_[i] = new_steps_[i];
            frames_consumed_[i] = new_frames_[i];
        }
        float total = new_cost_[frames_];
        int32_t steps = new_steps_[frames_];
        int32_t consumed = new_frames_[frames_];
        float norm = (steps > 0 && total < FLT_MAX) ? total / (float)steps : FLT_MAX;
        bool triggered = false;
        if (cooldown_left_ > 0) {
            --cooldown_left_;
        } else if (norm < threshold_ && consumed >= min_duration_frames_) {
            triggered = true;
            ++n_triggers_;
            cooldown_left_ = cooldown_frames_;
            reset_columns();  // same-keyword tail must not re-trigger
            *out_cost_per_step = norm;
            *out_frames_consumed = consumed;
        }
        return triggered;
    }

 private:
    void reset_columns() {
        for (int i = 0; i <= frames_; ++i) {
            cost_[i] = FLT_MAX;
            steps_[i] = 0;
            frames_consumed_[i] = 0;
        }
        cost_[0] = 0.0f;  // open begin: empty template matches anywhere
    }

    // Fixed max sizing (no heap on device): supports templates up to
    // 128 frames; larger keywords need the constant bumped (RAM +24 B/frame).
    static const int kMaxFrames = 128;
    const float* centroid_;
    int frames_;
    int dim_;
    float threshold_;
    int cooldown_frames_;
    int min_duration_frames_;
    int cooldown_left_;
    int n_triggers_;
    float cost_[kMaxFrames + 1];
    float new_cost_[kMaxFrames + 1];
    int32_t steps_[kMaxFrames + 1];
    int32_t new_steps_[kMaxFrames + 1];
    int32_t frames_consumed_[kMaxFrames + 1];
    int32_t new_frames_[kMaxFrames + 1];
};

#endif  // DTW_SPOTTER_H_
