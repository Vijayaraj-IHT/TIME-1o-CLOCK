/*
 * Dual-Threshold Hysteresis State Machine for ESP32-S3
 * SIH Problem Statement 26172
 *
 * Implements the deterministic activation lifecycle:
 * LISTENING -> VERIFYING -> ACTIVATED -> COOLDOWN
 */

#ifndef ACTIVATOR_STATE_MACHINE_H_
#define ACTIVATOR_STATE_MACHINE_H_

#include <cstdint>
#include <cstddef>

enum class ActivatorState {
    LISTENING,
    VERIFYING,
    ACTIVATED,
    COOLDOWN
};

class ActivatorStateMachine {
public:
    ActivatorStateMachine(
        float tau_high = 0.78f,
        float tau_low = 0.73f,
        int persistence_count = 3,
        int smoothing_window = 5,
        uint32_t cooldown_ms = 1500
    ) : tau_high_(tau_high),
        tau_low_(tau_low),
        persistence_count_(persistence_count),
        smoothing_window_(smoothing_window),
        cooldown_ms_(cooldown_ms),
        state_(ActivatorState::LISTENING),
        consecutive_hits_(0),
        history_idx_(0),
        history_count_(0),
        cooldown_start_ms_(0) {
        for (int i = 0; i < MAX_SMOOTHING; ++i) {
            sim_history_[i] = 0.0f;
        }
    }

    bool update(float raw_sim, uint32_t current_time_ms) {
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

        bool triggered = false;

        // 2. State machine transitions
        switch (state_) {
            case ActivatorState::LISTENING:
                if (smoothed_sim >= tau_high_) {
                    consecutive_hits_ = 1;
                    state_ = ActivatorState::VERIFYING;
                }
                break;

            case ActivatorState::VERIFYING:
                if (smoothed_sim >= tau_low_) {
                    consecutive_hits_++;
                    if (consecutive_hits_ >= persistence_count_) {
                        state_ = ActivatorState::ACTIVATED;
                        cooldown_start_ms_ = current_time_ms;
                        triggered = true;
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
                if (current_time_ms - cooldown_start_ms_ >= cooldown_ms_) {
                    consecutive_hits_ = 0;
                    state_ = ActivatorState::LISTENING;
                }
                break;
        }

        return triggered;
    }

    ActivatorState get_state() const { return state_; }
    void reset() {
        state_ = ActivatorState::LISTENING;
        consecutive_hits_ = 0;
        history_idx_ = 0;
        history_count_ = 0;
    }

private:
    static constexpr int MAX_SMOOTHING = 16;
    float tau_high_;
    float tau_low_;
    int persistence_count_;
    int smoothing_window_;
    uint32_t cooldown_ms_;

    ActivatorState state_;
    int consecutive_hits_;
    float sim_history_[MAX_SMOOTHING];
    int history_idx_;
    int history_count_;
    uint32_t cooldown_start_ms_;
};

#endif // ACTIVATOR_STATE_MACHINE_H_
