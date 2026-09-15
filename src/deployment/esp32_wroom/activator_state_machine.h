/*
 * Dual-Threshold Hysteresis State Machine for ESP32-S3
 * SIH Problem Statement 26172
 *
 * Implements the deterministic activation lifecycle:
 * LISTENING -> VERIFYING -> ACTIVATED -> COOLDOWN
 *
 * Mirrors src/streaming/state_machine.py exactly (same defaults, same order of
 * operations). Defaults are guarded by tests/test_operating_point.py -- do not
 * change them without also updating configs/config.yaml.
 */

#ifndef ACTIVATOR_STATE_MACHINE_H_
#define ACTIVATOR_STATE_MACHINE_H_

#include <cstdint>
#include <cstddef>
#include <cmath>

enum class ActivatorState {
    LISTENING,
    VERIFYING,
    ACTIVATED,
    COOLDOWN
};

class ActivatorStateMachine {
public:
    // Defaults synced to configs/config.yaml -> detection block. Do not change these
    // without also updating config.yaml and re-running experiments/threshold/*.py,
    // or the calibration and the deployed behavior will drift apart again.
    ActivatorStateMachine(
        float tau_high = 0.89f,
        float tau_low = 0.84f,
        int persistence_count = 4,
        int smoothing_window = 8,
        uint32_t cooldown_ms = 1500,
        // Phase-1 Change 5: adaptive background-tracking threshold. Defaults
        // synced to configs/config.yaml -> detection.adapt_* (guarded by
        // tests/test_operating_point.py). Pass adaptive=false for fixed-tau.
        bool adaptive = true,
        float adapt_alpha = 0.02f,
        float adapt_k = 2.0f,
        float adapt_max = 0.95f,
        int adapt_warmup_frames = 40,
        // Phase-1 Change 3b: stream-discontinuity reset threshold (ms).
        uint32_t stale_gap_ms = 500
    ) : tau_high_(tau_high),
        tau_low_(tau_low),
        persistence_count_(persistence_count),
        smoothing_window_(smoothing_window),
        cooldown_ms_(cooldown_ms),
        adaptive_(adaptive),
        adapt_alpha_(adapt_alpha),
        adapt_k_(adapt_k),
        adapt_max_(adapt_max),
        adapt_warmup_frames_(adapt_warmup_frames),
        stale_gap_ms_(stale_gap_ms),
        state_(ActivatorState::LISTENING),
        consecutive_hits_(0),
        history_idx_(0),
        history_count_(0),
        cooldown_start_ms_(0),
        bg_mean_(0.0f),
        bg_ex2_(0.0f),
        adapt_frames_(0),
        threshold_eff_(tau_high),
        last_time_ms_(0),
        have_last_time_(false) {
        for (int i = 0; i < MAX_SMOOTHING; ++i) {
            sim_history_[i] = 0.0f;
        }
    }

    // observed=false: VAD-silence frame (no inference ran). Still decays the
    // smoothing average, but must not teach the background model (Change 3a).
    bool update(float raw_sim, uint32_t current_time_ms, bool observed = true) {
        // 0. Change 3b: stream-discontinuity reset. A gap (or a clock jump
        // backwards) means the history holds evidence about audio that is long
        // gone. Clear the FAST state; the SLOW background stats are kept (same
        // room, still valid). Unsigned subtraction is wrap-safe for the forward
        // case; the explicit backwards check catches clock jumps.
        if (have_last_time_ &&
            (current_time_ms < last_time_ms_ ||
             current_time_ms - last_time_ms_ > stale_gap_ms_)) {
            state_ = ActivatorState::LISTENING;
            consecutive_hits_ = 0;
            history_idx_ = 0;
            history_count_ = 0;
        }
        last_time_ms_ = current_time_ms;
        have_last_time_ = true;

        // Change 3b: cooldown expiry is a lifecycle transition like the gap
        // reset above: re-arm LISTENING with a clean history BEFORE this
        // frame's evidence is appended, so the exit frame decides on fresh
        // evidence only. (A frame reaching the switch below in COOLDOWN is
        // still locked out: frozen, no-op.)
        if (state_ == ActivatorState::COOLDOWN &&
            current_time_ms - cooldown_start_ms_ >= cooldown_ms_) {
            state_ = ActivatorState::LISTENING;
            consecutive_hits_ = 0;
            history_idx_ = 0;
            history_count_ = 0;
        }

        // 1. Moving average filter
        sim_history_[history_idx_] = raw_sim;
        history_idx_ = (history_idx_ + 1) % smoothing_window_;
        if (history_count_ < smoothing_window_) {
            history_count_++;
        }

        float smoothed_sim = 0.0f;
        for (int i = 0; i < history_count_; ++i) {
            smoothed_sim += sim_history_[i];
        }
        smoothed_sim /= static_cast<float>(history_count_);

        // 2. Change 5: effective threshold from PRIOR frames' background stats
        // (no same-frame feedback). Upward-only: never below calibrated tau.
        threshold_eff_ = tau_high_;
        if (adaptive_ && adapt_frames_ >= adapt_warmup_frames_) {
            float var = bg_ex2_ - bg_mean_ * bg_mean_;
            if (var < 0.0f) {
                var = 0.0f;
            }
            float tau = bg_mean_ + adapt_k_ * std::sqrt(var);
            if (tau < tau_high_) {
                tau = tau_high_;
            }
            if (tau > adapt_max_) {
                tau = adapt_max_;
            }
            threshold_eff_ = tau;
        }

        bool triggered = false;

        // 3. State machine transitions
        switch (state_) {
            case ActivatorState::LISTENING:
                if (smoothed_sim >= threshold_eff_) {
                    consecutive_hits_ = 1;
                    state_ = ActivatorState::VERIFYING;
                } else {
                    consecutive_hits_ = 0;
                    // Change 5: learn ONLY on LISTENING + non-triggering, and
                    // (Change 3a) ONLY on real model observations.
                    if (adaptive_ && observed) {
                        bg_mean_ += adapt_alpha_ * (raw_sim - bg_mean_);
                        bg_ex2_ += adapt_alpha_ * (raw_sim * raw_sim - bg_ex2_);
                        adapt_frames_++;
                    }
                }
                break;

            case ActivatorState::VERIFYING:
                if (smoothed_sim >= tau_low_) {
                    consecutive_hits_++;
                    if (consecutive_hits_ >= persistence_count_) {
                        state_ = ActivatorState::ACTIVATED;
                        cooldown_start_ms_ = current_time_ms;
                        triggered = true;
                        // Change 3b: drop the evidence that fired this
                        // trigger, so post-cooldown frames start clean.
                        history_idx_ = 0;
                        history_count_ = 0;
                    }
                } else {
                    consecutive_hits_ = 0;
                    state_ = ActivatorState::LISTENING;
                }
                break;

            case ActivatorState::ACTIVATED:
                state_ = ActivatorState::COOLDOWN;
                break;

            case ActivatorState::COOLDOWN:
                // Still locked out (expiry is handled pre-append above).
                // Frozen: history keeps appending, no transitions.
                break;
        }

        return triggered;
    }

    ActivatorState get_state() const { return state_; }
    float get_threshold_eff() const { return threshold_eff_; }
    void reset() {
        state_ = ActivatorState::LISTENING;
        consecutive_hits_ = 0;
        history_idx_ = 0;
        history_count_ = 0;
        bg_mean_ = 0.0f;
        bg_ex2_ = 0.0f;
        adapt_frames_ = 0;
        threshold_eff_ = tau_high_;
        last_time_ms_ = 0;
        have_last_time_ = false;
    }

private:
    static constexpr int MAX_SMOOTHING = 16;
    float tau_high_;
    float tau_low_;
    int persistence_count_;
    int smoothing_window_;
    uint32_t cooldown_ms_;
    bool adaptive_;
    float adapt_alpha_;
    float adapt_k_;
    float adapt_max_;
    int adapt_warmup_frames_;
    uint32_t stale_gap_ms_;

    ActivatorState state_;
    int consecutive_hits_;
    float sim_history_[MAX_SMOOTHING];
    int history_idx_;
    int history_count_;
    uint32_t cooldown_start_ms_;

    // Change-5 background stats (EMA mean + EMA x^2).
    float bg_mean_;
    float bg_ex2_;
    int adapt_frames_;
    float threshold_eff_;

    // Change-3b discontinuity tracking.
    uint32_t last_time_ms_;
    bool have_last_time_;
};

#endif // ACTIVATOR_STATE_MACHINE_H_
