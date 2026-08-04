#include "core/telemetry.hpp"
#include "core/scheduler.hpp"

#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

using aether::core::ContinuousBatchingScheduler;
using aether::core::MetricKind;
using aether::core::PagedKVCacheAllocator;
using aether::core::Sequence;
using aether::core::SequencePhase;
using aether::core::SchedulerConfig;
using aether::core::TelemetryFlusher;
using aether::core::TelemetryRecorder;

namespace {
std::vector<uint32_t> make_prompt(size_t n) { return std::vector<uint32_t>(n, 1); }

// TelemetryRecorder is a process-wide singleton (deliberately -- see
// telemetry.hpp), so every test must reset it to a known state first
// rather than relying on test execution order. Restoring enabled=true
// afterwards matters too: GTest runs all TESTs in one process by default,
// so a test that disables telemetry and forgets to re-enable it would
// silently break every test that runs after it.
class TelemetryTest : public ::testing::Test {
protected:
    void SetUp() override {
        TelemetryRecorder::reset_for_testing();
        TelemetryRecorder::set_enabled(true);
    }
    void TearDown() override {
        TelemetryRecorder::set_enabled(true);
    }
};

}  // namespace

TEST_F(TelemetryTest, BasicPercentilesNearestRank) {
    TelemetryFlusher flusher;
    for (uint64_t i = 1; i <= 100; ++i) {
        TelemetryRecorder::record_ttft(i, std::chrono::nanoseconds(i * 1000));
    }
    flusher.drain_once();
    auto snap = flusher.snapshot();

    EXPECT_EQ(snap.ttft.count, 100u);
    // Nearest-rank percentile of the sequence {1000, 2000, ..., 100000}:
    // P50 -> index ceil(0.50*100)=50 -> value 50*1000.
    EXPECT_DOUBLE_EQ(snap.ttft.p50_ns, 50000.0);
    EXPECT_DOUBLE_EQ(snap.ttft.p95_ns, 95000.0);
    EXPECT_DOUBLE_EQ(snap.ttft.p99_ns, 99000.0);
    EXPECT_DOUBLE_EQ(snap.ttft.max_ns, 100000.0);
    EXPECT_DOUBLE_EQ(snap.ttft.mean_ns, 50500.0);
}

TEST_F(TelemetryTest, EmptySnapshotIsAllZero) {
    TelemetryFlusher flusher;
    auto snap = flusher.snapshot();
    EXPECT_EQ(snap.ttft.count, 0u);
    EXPECT_EQ(snap.inter_token.count, 0u);
    EXPECT_DOUBLE_EQ(snap.ttft.p50_ns, 0.0);
    EXPECT_EQ(snap.dropped_samples, 0u);
}

TEST_F(TelemetryTest, DisabledRecorderNeverPushesOrDrops) {
    TelemetryRecorder::set_enabled(false);
    for (int i = 0; i < 500; ++i) {
        TelemetryRecorder::record_ttft(i, std::chrono::nanoseconds(1));
        TelemetryRecorder::record_inter_token(i, std::chrono::nanoseconds(1));
    }
    TelemetryFlusher flusher;
    flusher.drain_once();
    auto snap = flusher.snapshot();
    EXPECT_EQ(snap.ttft.count, 0u);
    EXPECT_EQ(snap.inter_token.count, 0u);
    // Disabled means the sample never attempted a push at all -- it must
    // not show up as "dropped" either. Dropped is specifically for "we
    // tried and the ring was full", a distinct, more concerning condition.
    EXPECT_EQ(snap.dropped_samples, 0u);
}

TEST_F(TelemetryTest, TogglingEnabledMidStreamOnlyAffectsSamplesWhileDisabled) {
    TelemetryFlusher flusher;
    TelemetryRecorder::record_ttft(1, std::chrono::nanoseconds(10));
    TelemetryRecorder::set_enabled(false);
    TelemetryRecorder::record_ttft(2, std::chrono::nanoseconds(20));  // dropped on the floor
    TelemetryRecorder::set_enabled(true);
    TelemetryRecorder::record_ttft(3, std::chrono::nanoseconds(30));

    flusher.drain_once();
    auto snap = flusher.snapshot();
    EXPECT_EQ(snap.ttft.count, 2u);
    EXPECT_EQ(snap.dropped_samples, 0u);
}

// Forces more samples than the ring buffer's fixed capacity WITHOUT
// draining in between, proving the hot path drops (never blocks, never
// silently loses count of how many) once the ring is full -- the same
// "accounted for, never silently lost" principle the M4 spec cares about
// for chaos testing, just applied here to telemetry's own backpressure.
TEST_F(TelemetryTest, RingOverflowDropsAreCountedExactly) {
    constexpr uint64_t kRingCapacity = 65536;  // must match MetricRing's Capacity
    constexpr uint64_t kOverfill = kRingCapacity + 5000;
    for (uint64_t i = 0; i < kOverfill; ++i) {
        TelemetryRecorder::record_ttft(i, std::chrono::nanoseconds(1));
    }
    uint64_t dropped = TelemetryRecorder::dropped_samples();
    EXPECT_EQ(dropped, kOverfill - kRingCapacity);

    TelemetryFlusher flusher;
    flusher.drain_once();
    auto snap = flusher.snapshot();
    EXPECT_EQ(snap.ttft.count + dropped, kOverfill);
    EXPECT_EQ(snap.dropped_samples, dropped);
}

TEST_F(TelemetryTest, FlusherStopIsIdempotentAndFinalDrainCompletes) {
    TelemetryFlusher flusher;
    flusher.start();
    TelemetryRecorder::record_ttft(1, std::chrono::nanoseconds(5));
    flusher.stop();
    flusher.stop();  // must not double-join or crash
    auto snap = flusher.snapshot();
    EXPECT_EQ(snap.ttft.count, 1u);
}

// Many producer threads, one background flusher -- exactly the MPSC shape
// telemetry.hpp is designed for. Every sample must be accounted for as
// either collected or dropped; none may vanish.
TEST_F(TelemetryTest, ConcurrentProducersAccountForEverySample) {
    TelemetryFlusher flusher;
    flusher.start();
    constexpr int kThreads = 16;
    constexpr int kPerThread = 2000;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t] {
            for (int i = 0; i < kPerThread; ++i) {
                TelemetryRecorder::record_ttft(t * kPerThread + i, std::chrono::nanoseconds(1));
                TelemetryRecorder::record_inter_token(t * kPerThread + i, std::chrono::nanoseconds(2));
            }
        });
    }
    for (auto& th : threads) th.join();
    flusher.stop();

    auto snap = flusher.snapshot();
    uint64_t total = snap.ttft.count + snap.inter_token.count + snap.dropped_samples;
    EXPECT_EQ(total, static_cast<uint64_t>(kThreads) * kPerThread * 2);
}

// ===================== Scheduler integration =====================
//
// Confirms scheduler.hpp's decode-loop instrumentation records exactly the
// samples it should: one TTFT for the first generated token, one
// inter-token sample for every subsequent one -- never more, never fewer,
// and never any effect on the scheduling decisions themselves.

TEST_F(TelemetryTest, SchedulerRecordsExpectedSampleCounts) {
    PagedKVCacheAllocator pt(32, 256);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 128;
    cfg.max_prefill_chunk = 64;
    ContinuousBatchingScheduler sched(cfg, pt);

    constexpr uint32_t kMaxNew = 7;
    auto seq = std::make_shared<Sequence>(1, make_prompt(10), kMaxNew);
    sched.admit(seq);
    while (seq->phase != SequencePhase::FINISHED) sched.step();

    TelemetryFlusher flusher;
    flusher.drain_once();
    auto snap = flusher.snapshot();
    EXPECT_EQ(snap.ttft.count, 1u);
    EXPECT_EQ(snap.inter_token.count, kMaxNew - 1);
    EXPECT_EQ(snap.dropped_samples, 0u);
}

TEST_F(TelemetryTest, DisablingTelemetryDoesNotChangeSchedulingOutcome) {
    TelemetryRecorder::set_enabled(false);
    PagedKVCacheAllocator pt(32, 256);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 128;
    cfg.max_prefill_chunk = 64;
    ContinuousBatchingScheduler sched(cfg, pt);

    auto seq = std::make_shared<Sequence>(1, make_prompt(10), 5);
    sched.admit(seq);
    int iterations = 0;
    while (seq->phase != SequencePhase::FINISHED && iterations < 1000) {
        sched.step();
        ++iterations;
    }
    EXPECT_EQ(seq->phase, SequencePhase::FINISHED);

    TelemetryFlusher flusher;
    flusher.drain_once();
    auto snap = flusher.snapshot();
    EXPECT_EQ(snap.ttft.count, 0u);
    EXPECT_EQ(snap.inter_token.count, 0u);
}

// Randomized multi-sequence workload (same shape as
// test_scheduler.cpp's RandomizedWorkload test) -- proves the per-sequence
// counting invariant holds in aggregate under FIFO interleaving of many
// concurrent in-flight sequences, not just a single isolated one.
TEST_F(TelemetryTest, RandomizedMultiSequenceWorkloadSampleCountsMatchExpected) {
    PagedKVCacheAllocator pt(32, 256);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 128;
    cfg.max_prefill_chunk = 64;
    ContinuousBatchingScheduler sched(cfg, pt);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> prompt_len(1, 300);
    std::uniform_int_distribution<int> max_new(1, 20);

    uint64_t expected_ttft = 0, expected_inter = 0;
    constexpr uint64_t kNumSeqs = 50;
    for (uint64_t id = 0; id < kNumSeqs; ++id) {
        int mn = max_new(rng);
        sched.admit(std::make_shared<Sequence>(id, make_prompt(prompt_len(rng)), mn));
        expected_ttft += 1;
        expected_inter += static_cast<uint64_t>(mn - 1);
    }

    int iterations = 0;
    while ((sched.num_active() > 0 || sched.num_waiting() > 0) && iterations < 100000) {
        sched.step();
        ++iterations;
    }
    ASSERT_LT(iterations, 100000);

    TelemetryFlusher flusher;
    flusher.drain_once();
    auto snap = flusher.snapshot();
    EXPECT_EQ(snap.ttft.count, expected_ttft);
    EXPECT_EQ(snap.inter_token.count, expected_inter);
    EXPECT_EQ(snap.dropped_samples, 0u);
}
