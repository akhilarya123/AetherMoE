// AetherMoE — src/core/telemetry.hpp
//
// Milestone 4, Step 1: low-overhead telemetry for Time-to-First-Token
// (TTFT) and inter-token latency.
//
// The spec's core constraint is avoiding the "observer effect" -- the act
// of measuring must not meaningfully slow down the thing being measured.
// The design that follows from that:
//
//   - The hot path (scheduler threads recording a sample) does exactly one
//     relaxed atomic load (the enabled check) plus one lock-free
//     try_push() into a shared ring buffer. No mutex, no syscall, no
//     allocation, on the hot path, ever.
//   - This deliberately REUSES Milestone 1's existing
//     aether::core::LockFreeRingBuffer (Vyukov MPMC) rather than building a
//     second lock-free structure from scratch -- many producer threads
//     (every scheduler/worker thread recording samples), one consumer (the
//     background flusher below), which is exactly the MPMC queue's designed
//     use case with room to spare.
//   - The actual aggregation (sorting into percentile buckets) happens on a
//     SEPARATE background thread (TelemetryFlusher) that drains the ring
//     buffer on its own schedule -- this is the "flushed asynchronously"
//     part of the spec, and it's what keeps any O(log n)-or-worse
//     bookkeeping off every hot-path call.
//   - If the ring buffer is momentarily full (producer thread outrunning
//     the flusher), a sample is dropped and a counter incremented, rather
//     than blocking the hot path to wait for space -- the same
//     backpressure-over-blocking philosophy the M1 ingress ring buffer
//     already uses for requests. Telemetry must never be the reason a
//     request gets slower.
//
// Known, deliberate limitation: TelemetrySnapshot keeps every retained
// sample in memory to compute exact percentiles (nearest-rank method), not
// an approximate streaming quantile sketch (e.g. t-digest). Fine for the
// benchmark-run durations Milestone 4 targets; an explicit, documented
// scope boundary if this is ever pointed at a multi-day production
// deployment, same spirit as Milestone 1's page-table mutex tradeoff and
// Milestone 2's decision to stop tuning at a hardware ceiling -- a
// conscious choice, not an oversight.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "spmc_ring_buffer.hpp"

namespace aether::core {

enum class MetricKind : uint8_t {
    kTTFT = 0,        // admission -> first generated token
    kInterToken = 1,  // time between two consecutive decode steps
};

struct MetricSample {
    uint64_t seq_id = 0;
    MetricKind kind = MetricKind::kTTFT;
    uint64_t latency_ns = 0;

    MetricSample() = default;
    MetricSample(uint64_t sid, MetricKind k, uint64_t ns)
        : seq_id(sid), kind(k), latency_ns(ns) {}
};

// Percentile summary for one metric kind. Nearest-rank method: sorted,
// index = ceil(p * n) - 1, clamped -- standard, simple, and matches what
// most latency-reporting tools mean by "P99" without needing an
// interpolation-method footnote.
struct MetricStats {
    uint64_t count = 0;
    double mean_ns = 0.0;
    double p50_ns = 0.0;
    double p95_ns = 0.0;
    double p99_ns = 0.0;
    double max_ns = 0.0;
};

struct TelemetrySnapshot {
    MetricStats ttft;
    MetricStats inter_token;
    uint64_t dropped_samples = 0;  // ring buffer was full when recorded
};

namespace detail {

inline MetricStats compute_stats(std::vector<uint64_t>& latencies_ns) {
    MetricStats s;
    s.count = latencies_ns.size();
    if (s.count == 0) return s;

    std::sort(latencies_ns.begin(), latencies_ns.end());

    double sum = 0.0;
    for (uint64_t v : latencies_ns) sum += static_cast<double>(v);
    s.mean_ns = sum / static_cast<double>(s.count);
    s.max_ns = static_cast<double>(latencies_ns.back());

    auto pct = [&](double p) -> double {
        size_t idx = static_cast<size_t>(
            std::ceil(p * static_cast<double>(s.count)));
        if (idx == 0) idx = 1;
        if (idx > s.count) idx = s.count;
        return static_cast<double>(latencies_ns[idx - 1]);
    };
    s.p50_ns = pct(0.50);
    s.p95_ns = pct(0.95);
    s.p99_ns = pct(0.99);
    return s;
}

}  // namespace detail

// Shared MPMC ring buffer feeding the flusher. Capacity must stay a power
// of two (LockFreeRingBuffer's own requirement) -- 65536 gives ~a few
// seconds of headroom at tens of thousands of samples/sec before the
// flusher must keep up, comfortably above anything Milestone 1's own load
// test (10k req/s) implies for a per-request handful of samples.
using MetricRing = LockFreeRingBuffer<MetricSample, 65536>;

// TelemetryRecorder is the hot-path-facing API. It is intentionally a set
// of free functions over a process-wide singleton ring buffer (like a
// counter/metrics library), not an object every caller needs to thread
// through -- the scheduler only needs to call `record_ttft`/
// `record_inter_token`, it doesn't own or manage telemetry lifecycle.
class TelemetryRecorder {
public:
    // Global on/off switch -- what the overhead A/B test flips between
    // runs. A single relaxed load on the hot path; no other cost when
    // disabled.
    static void set_enabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_relaxed);
    }
    static bool enabled() { return enabled_.load(std::memory_order_relaxed); }

    static void record_ttft(uint64_t seq_id, std::chrono::nanoseconds latency) {
        record(seq_id, MetricKind::kTTFT, latency);
    }

    static void record_inter_token(uint64_t seq_id, std::chrono::nanoseconds latency) {
        record(seq_id, MetricKind::kInterToken, latency);
    }

    static MetricRing& ring() { return ring_; }

    static uint64_t dropped_samples() {
        return dropped_.load(std::memory_order_relaxed);
    }

    // Test/shutdown hook: undoes accumulated drop-counter state between
    // isolated test cases sharing this process-wide singleton.
    static void reset_for_testing() {
        dropped_.store(0, std::memory_order_relaxed);
        MetricSample discard;
        while (ring_.try_pop(discard)) {
        }
    }

private:
    static void record(uint64_t seq_id, MetricKind kind, std::chrono::nanoseconds latency) {
        if (!enabled()) return;  // the ONE hot-path branch when disabled
        if (!ring_.try_push(MetricSample(seq_id, kind, static_cast<uint64_t>(latency.count())))) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    static inline std::atomic<bool> enabled_{true};
    static inline std::atomic<uint64_t> dropped_{0};
    static inline MetricRing ring_{};
};

// Background consumer: drains TelemetryRecorder's ring buffer on its own
// thread and maintains a running aggregate that `snapshot()` can read at
// any time. This is the "asynchronous flush" the spec asks for -- the
// sorting/bucketing work for percentiles never happens on a caller's hot
// path, only here.
class TelemetryFlusher {
public:
    // drain_interval: how often the background thread wakes up to drain
    // the ring buffer. Shorter intervals mean snapshot() reflects more
    // recent activity at the cost of more wakeups; 5ms is a reasonable
    // default for a benchmarking/dev context, not tuned for a specific
    // production SLA.
    explicit TelemetryFlusher(std::chrono::milliseconds drain_interval =
                                   std::chrono::milliseconds(5))
        : drain_interval_(drain_interval) {}

    ~TelemetryFlusher() { stop(); }

    TelemetryFlusher(const TelemetryFlusher&) = delete;
    TelemetryFlusher& operator=(const TelemetryFlusher&) = delete;

    void start() {
        if (running_.exchange(true, std::memory_order_acq_rel)) return;  // idempotent
        thread_ = std::thread([this] { run(); });
    }

    void stop() {
        if (!running_.exchange(false, std::memory_order_acq_rel)) return;  // idempotent
        if (thread_.joinable()) thread_.join();
        drain_once();  // final drain so a snapshot right after stop() is complete
    }

    // Snapshot of everything drained so far (does NOT include samples still
    // sitting in the ring buffer that the background thread hasn't drained
    // yet -- call stop() first, or accept that a snapshot mid-run is a
    // point-in-time view of what's been flushed, same tradeoff any
    // eventually-consistent metrics pipeline makes deliberately).
    TelemetrySnapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<uint64_t> ttft_copy = ttft_ns_;
        std::vector<uint64_t> inter_copy = inter_token_ns_;
        TelemetrySnapshot snap;
        snap.ttft = detail::compute_stats(ttft_copy);
        snap.inter_token = detail::compute_stats(inter_copy);
        snap.dropped_samples = TelemetryRecorder::dropped_samples();
        return snap;
    }

    // Test hook: forces one drain pass without needing to wait a full
    // drain_interval -- keeps unit tests fast and deterministic instead of
    // sleeping and hoping the background thread got there first.
    void drain_once() {
        MetricSample sample;
        std::lock_guard<std::mutex> lock(mutex_);
        while (TelemetryRecorder::ring().try_pop(sample)) {
            (sample.kind == MetricKind::kTTFT ? ttft_ns_ : inter_token_ns_)
                .push_back(sample.latency_ns);
        }
    }

private:
    void run() {
        while (running_.load(std::memory_order_acquire)) {
            drain_once();
            std::this_thread::sleep_for(drain_interval_);
        }
    }

    std::chrono::milliseconds drain_interval_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    mutable std::mutex mutex_;
    std::vector<uint64_t> ttft_ns_;
    std::vector<uint64_t> inter_token_ns_;
};

}  // namespace aether::core
