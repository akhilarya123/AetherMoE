# Milestone 4 — Progress notes (Step 1 of 4)

Following the phasing suggested in `HANDOVER_MILESTONE4.md` §7: Step 1
(telemetry) is built and self-verified as far as this sandbox allows.
Steps 2–4 (chaos-testing layer, benchmarking CLI, end-to-end regression
suite) are not started yet — picking those up next, one at a time, after
Step 1 is confirmed on real hardware.

**Sandbox constraints this round (same shape as every prior milestone,
see `HANDOVER_MILESTONE4.md` §2):** this environment has a working `g++`
but **no network access**, so — same as Milestone 3 — I couldn't use the
project's actual CMake/GTest `FetchContent` setup directly. I self-verified
everything with throwaway dependency-free harnesses (real threads, real
`-fsanitize=thread` / `-fsanitize=address,undefined` runs) before writing
the real GTest-based `tests/test_telemetry.cpp` for your build. That file
itself has **not** been compiled by me — it needs your machine's GTest to
confirm it builds and passes cleanly, same as every `.cpp` test file in
this project going back to Milestone 3.

---

## 1. What's built

**`src/core/telemetry.hpp`** (new) — the telemetry subsystem itself:
- Reuses Milestone 1's existing `LockFreeRingBuffer` (Vyukov MPMC) as the
  hot-path collection mechanism, rather than inventing a second lock-free
  structure — many producer threads (scheduler/worker threads recording
  samples), one background consumer, which is exactly what that queue was
  built for.
- Hot-path cost per recorded sample: one relaxed atomic load (the enabled
  check) + one lock-free `try_push`. No mutex, no allocation, no syscall,
  ever, on the recording side.
- `TelemetryFlusher` drains the ring buffer on its own background thread
  and maintains running latency vectors for exact P50/P95/P99 (nearest-rank
  method) — this is the "flushed asynchronously" part of the spec: sorting
  and bucketing never happens on a caller's hot path.
- Backpressure, not blocking: if the ring is momentarily full, a sample is
  dropped and a counter incremented rather than stalling whatever's trying
  to record it. Every sample is accounted for either way — never silently
  lost, same principle the M4 spec cares about for chaos testing later,
  applied here to telemetry's own overflow behavior.

**`src/core/scheduler.hpp`** and **`src/core/sequence.hpp`** (both edited,
additively) — real integration, not a synthetic stand-in: every decode
step now records a TTFT sample (first generated token) or an inter-token
latency sample (every one after), via two new `Sequence` timestamp fields
(`admitted_at`, `last_decode_at`) that don't affect any existing behavior,
field layout expectations, or `StepResult`.

**`tests/test_telemetry.cpp`** (new, GTest, **unverified by me** — see
constraints above) — 11 tests: percentile correctness, empty-snapshot
zero-state, enable/disable gating (including mid-stream toggling),
ring-overflow drop accounting, flusher start/stop idempotency, a 16-thread
concurrent-producer accounting test, and four tests specifically on the
scheduler integration (exact sample counts for a single sequence, for a
randomized 50-sequence workload, and confirmation that disabling telemetry
never changes a scheduling outcome).

**`benchmarks/bench_telemetry_overhead.cpp`** (new) — the spec's required
overhead A/B test: runs the same synthetic 100,000-sequence randomized
workload through the scheduler twice (telemetry disabled vs. enabled,
15 interleaved repeats, reporting median-of-runs rather than a single
comparison — see §3 for why that matters here specifically), and asserts
overhead stays under the 1% budget.

**`CMakeLists.txt`** — updated: `test_telemetry.cpp` added to `aether_tests`;
new `bench_telemetry_overhead` executable target.

**`README.md`** — status table row updated to point here.

---

## 2. Self-verified in this sandbox (real runs, real output, not guessed)

Since GTest itself isn't available here, I wrote plain-`assert`
dependency-free harnesses reproducing what the real tests check, the same
discipline used for Milestone 3's orchestration layer:

- **Telemetry core correctness** (percentiles, enable/disable, overflow
  drop-counting, flusher idempotency): clean under `-fsanitize=thread`
  and, separately, `-fsanitize=address,undefined`.
  (Running TSan and UBSan *together* in this sandbox produced a data-race
  warning **inside the sanitizer runtime's own `pipe()`-based diagnostics**,
  not in any of this code — confirmed by isolating them: TSan alone is
  clean, ASan+UBSan together is clean. Worth knowing if this ever comes up
  again, in the same spirit as the Instruments-table gotcha from
  Milestone 3.)
- **16-thread concurrent producer stress**: 32,000 TTFT + 32,000
  inter-token samples recorded from 16 threads racing against one
  background flusher — exactly accounted for, zero drops, zero races.
- **Ring overflow**: pushed 70,536 samples into a 65,536-capacity ring with
  no draining in between — exactly 65,536 collected, exactly 5,000 counted
  as dropped, nothing missing, nothing double-counted.
- **Scheduler integration, reproducing all 5 of `test_scheduler.cpp`'s
  existing cases** verbatim as plain asserts: all pass unchanged with the
  telemetry instrumentation active, confirming this integration doesn't
  alter Milestone 1's scheduling behavior. Plus two new checks: a single
  7-max-new-token sequence produces exactly 1 TTFT + 6 inter-token samples;
  disabling telemetry produces a scheduling outcome that reaches
  `FINISHED` identically to the enabled case.

All of the above: clean under both `-fsanitize=thread` and
`-fsanitize=address,undefined` (run separately, per the note above).

---

## 3. The one open finding — overhead benchmark fails its own 1% budget

Real numbers, not a guess, from `bench_telemetry_overhead` run in this
sandbox (100,000 sequences, 15 interleaved repeats, median-of-runs):

```
telemetry disabled: median 112.04 ms
telemetry enabled:  median 114.78 ms
overhead: 2.45%   (target: <1%)
```

I deliberately ran this more than once before concluding anything, per
the handover's own §10 caution about a single data point being misleading:
first at 20,000 sequences / 7 repeats (1.63% overhead), then at 100,000 /
15 repeats (2.45%) — overhead went *up*, not down, with a larger, less
noisy sample, so this isn't just measurement jitter resolving itself with
more repeats. There's a real, currently unexplained cost somewhere between
`steady_clock::now()` overhead and the ring buffer's CAS-loop `try_push`,
and I haven't root-caused it yet.

**Important caveat I want to be explicit about:** this measurement is from
an x86 Linux sandbox container, not your actual M1 Mac — clock-read cost
and atomic-CAS cost can both differ meaningfully between the two, so this
number is a signal that something is worth investigating, not necessarily
the number that will reproduce on your machine. I'm flagging it rather
than either hiding it or overclaiming it's fixed.

**What I'd like from you before I go further on this:**
1. Build and run `tests/test_telemetry.cpp` on your machine (see §4) and
   paste back the exact GTest output — pass/fail counts, and the full
   text of any failure.
2. Run `bench_telemetry_overhead` and paste back the exact printed numbers
   (not "it passed/failed" — the actual ms figures and overhead
   percentage), so I have a real number from the actual target hardware
   to diagnose against instead of guessing from a Linux proxy.

Once I have that, I'll either confirm the overhead clears budget on real
hardware (in which case Step 1 is done and we move to chaos testing) or
profile and fix the real cost using your actual numbers as ground truth —
same pattern that found all 7 bugs across Milestones 1–3.

---

## 4. How to run it

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

./build/aether_tests            # should now include the new telemetry tests
                                 # alongside M1's 18 and M3's 28
./build/bench_telemetry_overhead
```

Concurrency check (the telemetry tests include a real multi-threaded
stress test, worth running under TSan once like the rest of the suite):
```bash
cmake -B build-tsan -DAETHER_SANITIZE=thread
cmake --build build-tsan -j$(sysctl -n hw.ncpu) --target aether_tests
./build-tsan/aether_tests
```

Please paste back the exact terminal output from both `aether_tests` and
`bench_telemetry_overhead` — pass/fail counts and the literal numbers, not
a summary — so anything unexpected can be root-caused precisely rather
than guessed at.
