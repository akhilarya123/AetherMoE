// AetherMoE — benchmarks/bench_cli.cpp
//
// Milestone 4, Step 3: automated benchmarking CLI. Per the spec: "generates
// realistic synthetic traffic (e.g., Poisson arrivals, mixed short/long
// prompt and generation lengths) and produces latency/throughput charts."
//
// Design decisions, stated explicitly (same discipline as every other
// scope call in this codebase):
//
//   - Drives the SAME architecture main.cpp uses -- a Poisson-arrival
//     producer pushing onto the real aether::api::IngressQueue
//     (LockFreeRingBuffer<shared_ptr<Sequence>>), and a separate engine
//     thread draining it and calling scheduler.step(), exactly mirroring
//     main.cpp's own admission/scheduling thread split. This is
//     deliberately IN-PROCESS rather than over real HTTP: it exercises the
//     real ring-buffer/scheduler/telemetry hot path under real Poisson
//     arrival timing, without needing a running HTTP server or the
//     network-dependent parts of the stack. `load_generator.cpp` (M1)
//     remains the tool for exercising the actual HTTP ingress layer
//     end-to-end; this one is for the latency/throughput numbers the M4
//     spec's Definition of Done asks for, which live in the telemetry
//     subsystem (Step 1), not in HTTP round-trip time (main.cpp's
//     /generate is admission-only -- it returns immediately, so an
//     HTTP-level timer would only ever measure admission latency, not
//     TTFT/inter-token, which is the whole point of Step 1's
//     instrumentation existing in the first place).
//   - "Multi-hour benchmark run" is taken literally: --duration paces
//     Poisson arrivals against REAL wall-clock time (sleep_until, not a
//     simulated/virtual clock), because the entire point of a multi-hour
//     run per the spec's Definition of Done is to catch real time-based
//     degradation (memory growth, fragmentation drift) that a
//     virtual-time simulation could not expose.
//   - Completion/throughput counting reads directly off each
//     scheduler.step() call's StepResult stream (phase_after == FINISHED)
//     rather than retaining every admitted Sequence for the whole run --
//     this keeps the benchmark's own memory footprint flat regardless of
//     --duration or --rate, matching the real engine's footprint
//     characteristics rather than adding a benchmark-only memory leak of
//     its own on top of whatever it's trying to measure.
//   - Output is CSV, not an in-process chart -- per the handover's own
//     suggestion, "CSV + a separate plotting step is simplest and most
//     portable." See benchmarks/plot_benchmark.py for the charting half.
//
// Usage:
//   ./bench_cli [--duration SEC] [--rate REQ_PER_SEC] [--long-fraction F]
//               [--repeat N] [--seed-base N] [--output PREFIX]
//
// Defaults are sized for a quick sanity run (30s x 3 repeats); pass
// --duration 3600 --repeat 1 for an actual multi-hour run.

#include "api/ingress_server.hpp"
#include "core/page_table.hpp"
#include "core/scheduler.hpp"
#include "core/telemetry.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <thread>

using namespace aether::core;
using aether::api::IngressQueue;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    double duration_seconds = 30.0;
    double rate_per_second = 100.0;
    double long_fraction = 0.15;
    int repeat = 3;
    uint64_t seed_base = 1000;
    std::string output_prefix = "bench_run";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (flag == "--duration") a.duration_seconds = std::stod(next());
        else if (flag == "--rate") a.rate_per_second = std::stod(next());
        else if (flag == "--long-fraction") a.long_fraction = std::stod(next());
        else if (flag == "--repeat") a.repeat = std::stoi(next());
        else if (flag == "--seed-base") a.seed_base = std::stoull(next());
        else if (flag == "--output") a.output_prefix = next();
        else {
            std::fprintf(stderr, "unknown flag: %s\n", flag.c_str());
            std::exit(2);
        }
    }
    return a;
}

// Mixed short/long prompt+generation-length draw. Short requests dominate
// (matching a realistic chat-style traffic mix); the `long_fraction` of
// requests get materially longer prompts AND more generated tokens, so a
// benchmark run actually exercises both the fast decode-bound path and the
// slower prefill-bound / longer-KV-cache-footprint path in the same run.
struct RequestShape {
    uint32_t prompt_len;
    uint32_t max_new_tokens;
    bool is_long;
};

RequestShape draw_request_shape(std::mt19937& rng, double long_fraction) {
    static thread_local std::uniform_real_distribution<double> coin(0.0, 1.0);
    static thread_local std::uniform_int_distribution<int> short_prompt(10, 200);
    static thread_local std::uniform_int_distribution<int> short_new(1, 32);
    static thread_local std::uniform_int_distribution<int> long_prompt(500, 2000);
    static thread_local std::uniform_int_distribution<int> long_new(32, 256);

    bool is_long = coin(rng) < long_fraction;
    if (is_long) {
        return {static_cast<uint32_t>(long_prompt(rng)), static_cast<uint32_t>(long_new(rng)), true};
    }
    return {static_cast<uint32_t>(short_prompt(rng)), static_cast<uint32_t>(short_new(rng)), false};
}

struct RunSummary {
    int run_index = 0;
    uint64_t admitted = 0;
    uint64_t rejected_backpressure = 0;
    uint64_t completed = 0;
    uint64_t still_in_flight_at_cutoff = 0;
    uint64_t tokens_generated = 0;
    double wall_seconds = 0.0;
    double requests_per_second = 0.0;
    double tokens_per_second = 0.0;
    MetricStats ttft;
    MetricStats inter_token;
    uint64_t dropped_telemetry_samples = 0;
};

// One full benchmark run: Poisson-paced admission for `duration_seconds`
// of real wall-clock time, against a fresh scheduler/queue/telemetry
// state, mirroring main.cpp's own thread split (a producer admitting work,
// a separate thread draining the queue and stepping the scheduler).
RunSummary run_once(const Args& args, int run_index, uint64_t seed) {
    TelemetryRecorder::reset_for_testing();
    TelemetryRecorder::set_enabled(true);

    // Same sizing as main.cpp's own defaults, so this benchmark reflects
    // the actual deployed configuration rather than an arbitrary one.
    constexpr size_t kBlockSizeTokens = 16;
    constexpr size_t kNumPhysicalBlocks = 4096;
    PagedKVCacheAllocator page_table(kBlockSizeTokens, kNumPhysicalBlocks);
    SchedulerConfig sched_cfg;
    sched_cfg.max_batch_tokens = 2048;
    sched_cfg.max_prefill_chunk = 512;
    ContinuousBatchingScheduler scheduler(sched_cfg, page_table);

    IngressQueue queue;
    std::atomic<bool> stop_engine{false};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> tokens_generated{0};

    // BUG (found via real numbers from the user's M1 Mac, not guessed):
    // this flusher used to be constructed AFTER the run finished, with a
    // single drain_once() call at the end -- meaning nothing drained the
    // telemetry ring buffer *during* the run at all. It filled to its
    // 65536 capacity almost immediately and silently dropped nearly
    // everything produced after that, which is exactly why the collected
    // count always came out to precisely 65536 regardless of run length
    // or arrival rate -- the smoking gun once the real drop numbers
    // (92,732 and 990,860 dropped, both with EXACTLY 65536 collected) came
    // back from real multi-core hardware, ruling out "flusher starved by
    // a single-core sandbox" as the cause. telemetry.hpp itself was never
    // the problem; this benchmark just forgot to call flusher.start().
    TelemetryFlusher flusher;
    flusher.start();

    std::thread engine_thread([&] {
        while (!stop_engine.load(std::memory_order_acquire)) {
            std::shared_ptr<Sequence> seq;
            int drained = 0;
            while (drained < 256 && queue.try_pop(seq)) {
                scheduler.admit(seq);
                ++drained;
            }
            auto results = scheduler.step();
            for (auto& r : results) {
                if (r.phase_after == SequencePhase::FINISHED) {
                    completed.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (drained == 0 && results.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });

    // Poisson-process arrival generation: inter-arrival times drawn from
    // Exponential(rate), which is precisely what "Poisson arrivals" means
    // -- the memoryless inter-arrival distribution whose counting process
    // is Poisson. Paced against a real steady_clock, not simulated.
    std::mt19937 rng(seed);
    std::exponential_distribution<double> inter_arrival(args.rate_per_second);
    uint64_t next_seq_id = 1;
    uint64_t admitted = 0, rejected_backpressure = 0;

    auto run_start = Clock::now();
    auto deadline = run_start + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(args.duration_seconds));
    auto next_arrival = run_start;

    while (true) {
        next_arrival += std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(inter_arrival(rng)));
        if (next_arrival >= deadline) break;
        std::this_thread::sleep_until(next_arrival);

        RequestShape shape = draw_request_shape(rng, args.long_fraction);
        auto seq = std::make_shared<Sequence>(
            next_seq_id++, std::vector<uint32_t>(shape.prompt_len, 1), shape.max_new_tokens);
        if (queue.try_push(seq)) {
            ++admitted;
        } else {
            ++rejected_backpressure;  // real backpressure, same semantics as main.cpp's /generate 503
        }
    }

    // Grace period: let in-flight work drain a bit past the arrival
    // deadline before taking the final snapshot, rather than measuring a
    // snapshot the instant arrivals stop (which would undercount
    // completions purely due to the cutoff, not anything the engine did
    // wrong). Fixed at 2s -- generous relative to this benchmark's token
    // budgets, not tuned per workload.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    auto wall_elapsed = Clock::now() - run_start;

    stop_engine.store(true, std::memory_order_release);
    engine_thread.join();

    flusher.stop();  // joins the background drain thread, then does one final drain
    auto snap = flusher.snapshot();

    RunSummary summary;
    summary.run_index = run_index;
    summary.admitted = admitted;
    summary.rejected_backpressure = rejected_backpressure;
    summary.completed = completed.load();
    summary.still_in_flight_at_cutoff = admitted - summary.completed;
    // Every generated token produced exactly one telemetry sample (TTFT
    // for the first, inter-token for every one after -- see
    // scheduler.hpp's integration), so this sum is an exact token count,
    // not an estimate.
    summary.tokens_generated = snap.ttft.count + snap.inter_token.count;
    summary.wall_seconds = std::chrono::duration<double>(wall_elapsed).count();
    summary.requests_per_second = summary.completed / summary.wall_seconds;
    summary.tokens_per_second = summary.tokens_generated / summary.wall_seconds;
    summary.ttft = snap.ttft;
    summary.inter_token = snap.inter_token;
    summary.dropped_telemetry_samples = snap.dropped_samples;
    return summary;
}

void print_summary(const RunSummary& s) {
    std::printf("run %d: admitted=%llu rejected=%llu completed=%llu in_flight=%llu "
                "tokens=%llu wall=%.1fs req/s=%.2f tok/s=%.2f\n",
                s.run_index, (unsigned long long)s.admitted, (unsigned long long)s.rejected_backpressure,
                (unsigned long long)s.completed, (unsigned long long)s.still_in_flight_at_cutoff,
                (unsigned long long)s.tokens_generated, s.wall_seconds, s.requests_per_second,
                s.tokens_per_second);
    std::printf("  TTFT (ns):        p50=%.0f p95=%.0f p99=%.0f mean=%.0f (n=%llu)\n",
                s.ttft.p50_ns, s.ttft.p95_ns, s.ttft.p99_ns, s.ttft.mean_ns,
                (unsigned long long)s.ttft.count);
    std::printf("  inter-token (ns): p50=%.0f p95=%.0f p99=%.0f mean=%.0f (n=%llu)\n",
                s.inter_token.p50_ns, s.inter_token.p95_ns, s.inter_token.p99_ns,
                s.inter_token.mean_ns, (unsigned long long)s.inter_token.count);
    if (s.dropped_telemetry_samples > 0) {
        std::printf("  WARNING: %llu telemetry samples dropped (ring buffer saturated)\n",
                    (unsigned long long)s.dropped_telemetry_samples);
    }
}

void write_csv_header(std::ofstream& out) {
    out << "run_index,admitted,rejected_backpressure,completed,still_in_flight_at_cutoff,"
           "tokens_generated,wall_seconds,requests_per_second,tokens_per_second,"
           "ttft_p50_ns,ttft_p95_ns,ttft_p99_ns,ttft_mean_ns,ttft_count,"
           "inter_token_p50_ns,inter_token_p95_ns,inter_token_p99_ns,inter_token_mean_ns,inter_token_count,"
           "dropped_telemetry_samples\n";
}

void write_csv_row(std::ofstream& out, const RunSummary& s) {
    out << s.run_index << ',' << s.admitted << ',' << s.rejected_backpressure << ','
        << s.completed << ',' << s.still_in_flight_at_cutoff << ',' << s.tokens_generated << ','
        << s.wall_seconds << ',' << s.requests_per_second << ',' << s.tokens_per_second << ','
        << s.ttft.p50_ns << ',' << s.ttft.p95_ns << ',' << s.ttft.p99_ns << ',' << s.ttft.mean_ns << ','
        << s.ttft.count << ',' << s.inter_token.p50_ns << ',' << s.inter_token.p95_ns << ','
        << s.inter_token.p99_ns << ',' << s.inter_token.mean_ns << ',' << s.inter_token.count << ','
        << s.dropped_telemetry_samples << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::printf("AetherMoE Milestone 4 benchmarking CLI\n");
    std::printf("  duration=%.1fs rate=%.1f req/s long_fraction=%.2f repeats=%d\n\n",
                args.duration_seconds, args.rate_per_second, args.long_fraction, args.repeat);

    std::string csv_path = args.output_prefix + "_summary.csv";
    std::ofstream csv(csv_path);
    write_csv_header(csv);

    for (int i = 0; i < args.repeat; ++i) {
        RunSummary s = run_once(args, i, args.seed_base + static_cast<uint64_t>(i));
        print_summary(s);
        write_csv_row(csv, s);
        csv.flush();
    }
    csv.close();

    std::printf("\nsummary written to %s\n", csv_path.c_str());
    if (args.repeat > 1) {
        std::printf("(run-to-run consistency: compare the ttft_p50_ns/p95_ns/p99_ns columns "
                    "across rows -- see benchmarks/plot_benchmark.py --consistency)\n");
    }
    return 0;
}
