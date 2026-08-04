// AetherMoE — benchmarks/bench_telemetry_overhead.cpp
//
// Milestone 4 testing plan: "A/B comparison of throughput with telemetry
// enabled vs. disabled, asserting overhead stays under a fixed budget
// (e.g., <1%)."
//
// This drives the SAME synthetic workload (a large batch of sequences with
// randomized prompt/generation lengths, matching test_scheduler.cpp's own
// RandomizedWorkload test shape) through ContinuousBatchingScheduler twice
// -- once with TelemetryRecorder disabled, once enabled -- and reports
// wall-clock time and the resulting overhead percentage. Every decode step
// records exactly one telemetry sample when enabled (see scheduler.hpp),
// so this exercises the real hot-path integration, not a synthetic stand-in
// for it.
//
// Repeats each configuration multiple times and reports median-of-runs,
// not a single sample -- Milestone 2's evidence doc is explicit that a
// single before/after comparison can be actively misleading for exactly
// this kind of measurement (noise, thermal/scheduler jitter), so this
// benchmark doesn't repeat that mistake here.

#include "core/scheduler.hpp"
#include "core/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace aether::core;
using Clock = std::chrono::steady_clock;

namespace {

constexpr uint64_t kNumSequences = 100000;
constexpr int kRepeats = 15;

double run_once(bool telemetry_enabled) {
    TelemetryRecorder::reset_for_testing();
    TelemetryRecorder::set_enabled(telemetry_enabled);

    PagedKVCacheAllocator pt(32, 1u << 16);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 2048;
    cfg.max_prefill_chunk = 512;
    ContinuousBatchingScheduler sched(cfg, pt);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> prompt_len(1, 64);
    std::uniform_int_distribution<int> max_new(1, 16);

    for (uint64_t id = 0; id < kNumSequences; ++id) {
        sched.admit(std::make_shared<Sequence>(
            id, std::vector<uint32_t>(prompt_len(rng), 1), max_new(rng)));
    }

    auto start = Clock::now();
    uint64_t iterations = 0;
    while ((sched.num_active() > 0 || sched.num_waiting() > 0) &&
           iterations < 10'000'000) {
        sched.step();
        ++iterations;
    }
    auto elapsed = Clock::now() - start;

    TelemetryRecorder::set_enabled(true);  // restore default for the next run
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main() {
    std::vector<double> disabled_ms, enabled_ms;
    // Interleave enabled/disabled runs (not all-disabled-then-all-enabled)
    // so a slow drift over the run (thermal throttling, background load)
    // doesn't get attributed entirely to one side.
    for (int i = 0; i < kRepeats; ++i) {
        disabled_ms.push_back(run_once(/*telemetry_enabled=*/false));
        enabled_ms.push_back(run_once(/*telemetry_enabled=*/true));
    }

    double d = median(disabled_ms);
    double e = median(enabled_ms);
    double overhead_pct = (e - d) / d * 100.0;

    std::printf("AetherMoE Milestone 4 telemetry overhead benchmark\n");
    std::printf("  sequences=%llu repeats=%d\n",
                 (unsigned long long)kNumSequences, kRepeats);
    std::printf("  telemetry disabled: median %.2f ms (all runs:", d);
    for (double v : disabled_ms) std::printf(" %.2f", v);
    std::printf(")\n");
    std::printf("  telemetry enabled:  median %.2f ms (all runs:", e);
    for (double v : enabled_ms) std::printf(" %.2f", v);
    std::printf(")\n");
    std::printf("  overhead: %.3f%%  (target: <1%%)\n", overhead_pct);

    if (overhead_pct >= 1.0) {
        std::printf("  RESULT: FAIL -- overhead at or above the 1%% budget.\n");
        return 1;
    }
    std::printf("  RESULT: PASS\n");
    return 0;
}
