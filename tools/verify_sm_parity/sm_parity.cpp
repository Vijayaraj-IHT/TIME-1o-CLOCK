// Phase-1 Change 3b + Change-5 firmware completion: desktop parity test.
//
// Compiles the REAL master header (no mocks):
//   g++ -std=c++17 -Wall -Wextra -Isrc/deployment/esp32 tools/verify_sm_parity/sm_parity.cpp -o /tmp/sm_parity && /tmp/sm_parity
//
// Self-asserting (exit 1 + FAIL line on any mismatch) AND prints a trace that
// tools/verify_sm_parity/sm_parity_mirror.py must reproduce (run the mirror
// with this binary's stdout as argv[1]; it diffs with tolerance).
//
// KNOWN structural difference (documented, not a bug): firmware reports the
// transient ACTIVATED state (2) on its trigger frame and steps to COOLDOWN on
// the next update(); Python transitions to COOLDOWN within the trigger frame.
// The mirror folds fw st=2+trg=1 to st=3 before comparing; everything else
// (trigger times, tau_eff trajectory, gap/cooldown behavior) must match.
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "activator_state_machine.h"

static int g_fail = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::printf("ASSERT-FAIL: %s\n", msg);
        g_fail++;
    }
}

static void trace(const char* tag, uint32_t t, const ActivatorStateMachine& sm, bool trg) {
    std::printf("%s t=%u tau=%.4f st=%d trg=%d\n",
                tag, t, sm.get_threshold_eff(),
                static_cast<int>(sm.get_state()), trg ? 1 : 0);
}

int main() {
    // A: fixed-tau trigger + clean re-arm (proves cooldown history clear).
    {
        ActivatorStateMachine sm(0.89f, 0.84f, 4, 8, 1500, false);
        bool trg;
        uint32_t trig_at[4];
        int n_trig = 0;
        uint32_t t = 0;
        for (int i = 0; i < 6; i++) {  // 0.95 x6
            trg = sm.update(0.95f, t);
            trace("A1", t, sm, trg);
            if (trg) trig_at[n_trig++] = t;
            t += 50;
        }
        for (int i = 0; i < 30; i++) {  // 0.10 x30: ride out cooldown
            trg = sm.update(0.10f, t);
            if (t == 600 || t == 1600 || t == 1650 || t == 1700) trace("A2", t, sm, trg);
            if (trg) trig_at[n_trig++] = t;
            t += 50;
        }
        for (int i = 0; i < 12; i++) {  // 0.95 x12: re-arm from clean slate
            // (8-window must flush three 0.10s, then fresh persistence x4)
            trg = sm.update(0.95f, t);
            trace("A3", t, sm, trg);
            if (trg) trig_at[n_trig++] = t;
            t += 50;
        }
        check(n_trig == 2, "A: exactly 2 triggers");
        check(n_trig == 2 && trig_at[0] == 150, "A: first trigger @150ms");
        check(n_trig == 2 && trig_at[1] == 2300, "A: re-trigger @2300ms");
    }

    // B: adaptive rise / warmup / ceiling (Change-5 algorithm, in C++).
    // NOTE (measured Sep-2026): from a zero start the EMA variance overshoots,
    // so tau pins at the 0.95 cap for ~200 frames before settling to ~0.91
    // (~600 frames). The 40-frame warmup does NOT cover this transient: a
    // longer warmup would trade boot-time noise blindness for no overshoot.
    // Kept as-is pending MEASURE-1 data (see docs/PHASE1_CHANGES.md).
    {
        ActivatorStateMachine sm;  // all defaults: adaptive, warmup 40
        bool any_trg = false;
        float tau0 = 0, tau39 = 0, tau40 = 0, tau119 = 0, tau599 = 0;
        for (int i = 0; i < 600; i++) {
            float s = (i % 2 == 0) ? 0.87f : 0.79f;
            bool trg = sm.update(s, (uint32_t)(i * 50));
            any_trg = any_trg || trg;
            if (i == 0) tau0 = sm.get_threshold_eff();
            if (i == 39) tau39 = sm.get_threshold_eff();
            if (i == 40) tau40 = sm.get_threshold_eff();
            if (i == 119) tau119 = sm.get_threshold_eff();
            if (i == 599) tau599 = sm.get_threshold_eff();
        }
        std::printf("B tau0=%.4f tau39=%.4f tau40=%.4f tau119=%.4f tau599=%.4f\n",
                    tau0, tau39, tau40, tau119, tau599);
        check(tau0 == 0.89f, "B: frame 0 at calibrated tau");
        check(tau39 == 0.89f, "B: warmup frame 39 still fixed tau");
        check(tau40 > 0.89f, "B: tau rises right after warmup");
        check(tau119 > 0.89f, "B: tau stays raised during transient");
        check(tau599 > 0.90f && tau599 < 0.93f, "B: tau settles ~0.91");
        check(!any_trg, "B: confusing bg never triggers");
    }

    // C: unobserved frames do not advance warmup (Change 3a, in C++).
    {
        ActivatorStateMachine sm;
        for (int i = 0; i < 200; i++) sm.update(0.0f, (uint32_t)(i * 50), false);
        for (int i = 0; i < 10; i++) {
            float s = (i % 2 == 0) ? 0.87f : 0.79f;
            sm.update(s, (uint32_t)(10000 + i * 50), true);
        }
        float tau = sm.get_threshold_eff();
        std::printf("C tau_after_10obs=%.4f\n", tau);
        check(tau == 0.89f, "C: 200 unobserved + 10 observed still in warmup");
        for (int i = 0; i < 40; i++) {
            float s = (i % 2 == 0) ? 0.87f : 0.79f;
            sm.update(s, (uint32_t)(11000 + i * 50), true);
        }
        tau = sm.get_threshold_eff();
        std::printf("C tau_after_50obs=%.4f\n", tau);
        check(tau > 0.89f, "C: 50 observed frames leave warmup");
    }

    // D: gap + backwards-clock reset (Change 3b, in C++).
    {
        ActivatorStateMachine sm;
        sm.update(0.95f, 0);
        check(sm.get_state() == ActivatorState::VERIFYING, "D: hot start VERIFYING");
        bool trg = sm.update(0.50f, 2000);  // 2s stall
        trace("D1", 2000, sm, trg);
        check(sm.get_state() == ActivatorState::LISTENING, "D: gap -> LISTENING");
        trg = sm.update(0.10f, 500);  // clock jumped back
        trace("D2", 500, sm, trg);
        check(sm.get_state() == ActivatorState::LISTENING, "D: backwards -> LISTENING");
    }

    // E: cooldown-exit frame decides on fresh evidence (proves exit-clear).
    {
        ActivatorStateMachine sm(0.89f, 0.84f, 2, 4, 200);
        sm.update(0.95f, 0);
        bool trg = sm.update(0.95f, 50);  // trigger, cooldown till 250
        check(trg, "E: trigger @50ms");
        sm.update(0.95f, 100);  // cooldown frame (would linger)
        trg = sm.update(0.85f, 300);  // exit frame: 0.85 alone < tau ...
        trace("E1", 300, sm, trg);    // ... but (0.95+0.85)/2 would VERIFY
        check(sm.get_state() == ActivatorState::LISTENING, "E: exit decides fresh");
    }

    // F: Change 3d -- persistence completes ON an unobserved (silence) frame.
    {
        ActivatorStateMachine sm(0.89f, 0.84f, 8, 8, 1500);
        for (int i = 0; i < 7; i++) {
            bool trg = sm.update(0.99f, (uint32_t)(i * 50), true);
            check(!trg, "F: no early trigger");
        }
        check(sm.get_state() == ActivatorState::VERIFYING, "F: staged VERIFYING");
        // decay edge: (7x0.99 + 0.0)/8 = 0.866 >= tau_low -> count 8 -> fire
        bool trg = sm.update(0.0f, 350, false);
        trace("F1", 350, sm, trg);
        check(trg, "F: unobserved frame completes persistence");
    }

    std::printf(g_fail ? "SM_PARITY_CPP: FAIL\n" : "SM_PARITY_CPP: PASS\n");
    return g_fail ? 1 : 0;
}
