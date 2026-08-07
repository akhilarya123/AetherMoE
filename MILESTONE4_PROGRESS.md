# Milestone 4 — Progress notes (Step 1 of 4)

## 0. Step 1 status: CONFIRMED on real hardware

Ran on the actual M1 Mac (not the Linux sandbox proxy from earlier):

```
38 tests from 5 test suites ran. [ PASSED ] 38 tests.  (10 new TelemetryTest cases, all green)

telemetry disabled: median 120.59 ms
telemetry enabled:  median 121.48 ms
overhead: 0.738%  (target: <1%)
RESULT: PASS
```

The 2.45% figure from the Linux sandbox in §3 below was, as flagged at the
time, a proxy measurement that didn't hold on the real target hardware —
clock-read/CAS cost evidently differ enough between the two that it
mattered. No fix was needed; keeping §3 below as-written for the record
rather than editing it away, since the caveat about not trusting the
sandbox number turned out to be the right call.

**Step 1 (telemetry) is done.** Step 2 (chaos-testing layer) starts below.

---

## Step 2: Chaos-testing layer

**Same sandbox constraints as Step 1** — no network, so `tests/test_chaos.cpp`
(GTest) is written but **not compiled by me**; I self-verified the
underlying logic with a dependency-free plain-`assert` harness first,
same discipline as Step 1 and as Milestone 3 before it.

### What's built

**`src/orchestration/chaos.hpp`** (new) — two independent tools, matching
the M4 spec's testing-plan item:
- **`DelayInjectingTransport`** — a decorator over `ICollectiveTransport`
  that sleeps a configurable duration before `send()`/`receive()`. Reuses
  the interface abstraction exactly as `collective_transport.hpp`'s own
  header comment intends: `Router` only ever holds an
  `ICollectiveTransport*`, so substituting a delay-injecting wrapper for
  one shard's entry needs zero changes to `Router` or
  `UnixSocketTransport`.
- **`ChaosScript`** — schedules a real `SIGKILL` of a real worker process
  at a random point within a time window, on a background thread, while
  the caller keeps driving traffic on its own thread. This is Milestone
  3's `KilledWorkerIsDetectedNotHung` test (kill *before* traffic starts)
  generalized to kill *during* active traffic, which is what the spec's
  "at random points during active traffic" phrase actually requires.

**`src/orchestration/process_spawn.hpp`** (edited, additive) — added a
`shard_pids` map to `SpawnedWorkers` (`shard_id -> pid`) so chaos code can
target "kill shard 2" without the caller separately tracking which index
into `child_pids` belongs to which shard.

**`src/orchestration/routed_token.hpp`** (edited, additive) — added
`failed` and `failure_reason` fields to `RoutedResult`, following the
exact same convention as the existing `effective_expert` field
(router-only bookkeeping, defaulted, never sent over the wire).

**`src/orchestration/router.hpp`** (edited, additive) — new
`route_batch_tolerant()` method plus `is_shard_unhealthy()` /
`unhealthy_shard_count()`. **`route_batch()` itself is completely
untouched** — Milestone 3's own test depends on it still throwing on a
dead shard, and that contract stays exactly as-is. `route_batch_tolerant`
is the new graceful-degradation path: every token gets exactly one
`RoutedResult` back — a real result, or `failed=true` with a reason —
never a throw, never a silently missing `batch_position`. A shard that
fails once is remembered so later calls skip re-attempting a socket
that's already known dead, and healthy shards in the same batch are
unaffected.

**`tests/test_chaos.cpp`** (new, GTest, **unverified by me**) — 6 tests:
`shard_pids` correctness, delay-injection latency (both nonzero and the
zero-delay no-op case), tolerant-routing matching strict routing on a
healthy cluster, tolerant routing isolating a dead shard without
throwing/hanging (including a third batch proving the isolation actually
skips re-attempting the dead shard, not just detects it once), and the
main scripted-kill-during-active-traffic test.

### Self-verified in this sandbox

Reproduced all 6 tests as a plain-`assert` harness against real forked
worker processes and a real background `SIGKILL` thread:

- **Delay injection**: an 80ms send-delay + 80ms receive-delay wrapper
  measured a real ≥150ms round trip (target 160ms accounting for
  scheduling slack) — the delay is real, not a no-op.
- **Tolerant routing on a healthy cluster**: byte-for-byte matches strict
  `route_batch`'s results, `failed=false` everywhere, zero shards marked
  unhealthy.
- **Chaos during active traffic** (the main scenario): killed shard 1 at
  a random point within the first 150ms while sending 40 batches of 200
  tokens each (8,000 tokens total, spanning ~450ms of wall time) on the
  main thread concurrently. Every run: **all 8,000 tokens accounted for**
  (`succeeded + failed == sent`, exactly, every time), the victim shard
  correctly detected and marked unhealthy, and — critically — the three
  surviving shards kept serving successful results the *entire* run, not
  just before the kill.
- Ran this 3x under `-fsanitize=address,undefined` and 3x under
  `-fsanitize=thread`: clean, deterministic invariants every time
  (`8000 sent / ~6440 succeeded / ~1560 failed` in the TSan/plain runs —
  small run-to-run variance in the exact split is expected and fine, since
  it depends on exactly when within the random window the kill lands
  relative to in-flight batches; the *invariant* that succeeded+failed
  always equals sent held in every single run).
- Confirmed the untouched `route_batch()` (strict) still throws on a dead
  shard exactly as Milestone 3's test expects — full regression check
  against the edited `router.hpp`/`routed_token.hpp`/`process_spawn.hpp`.

**One sandbox-only artifact worth flagging, not a real bug:** running the
harness under ThreadSanitizer specifically (not ASan/UBSan, not plain)
produced duplicated stdout lines — buffered-but-unflushed `std::cout`
output getting copied into forked child processes and echoed again, a
known category of `fork()` + buffered-stdio + sanitizer-runtime
interaction, same flavor as the TSan+UBSan `pipe()` false-positive
flagged in Step 1's notes. All assertions still passed correctly every
single time; only the diagnostic printing was affected, not any actual
process/router/chaos behavior. Real GTest binaries don't buffer output
the same way `std::cout`-in-a-loop does, so this is very unlikely to
reproduce in your actual test run — flagging it here so it isn't a
surprise if it ever does.

### What to run and report back

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

./build/aether_tests   # now includes 6 new Chaos.* tests alongside
                        # the 38 from Step 1
```

Please paste back the exact GTest output again — this one has real timing
assertions (`EXPECT_LT`/`EXPECT_GE` on wall-clock milliseconds) that could
in principle be tighter or looser than your machine needs; if anything is
flaky or fails, the exact numbers matter for tuning the bounds rather than
guessing at them.

Once this is confirmed, remaining per the phasing in
`HANDOVER_MILESTONE4.md` §7: Step 3 (benchmarking CLI) — done below —
and Step 4 (end-to-end regression suite tying Milestones 1–4 together).

---

## Step 3: Benchmarking CLI

**`benchmarks/bench_cli.cpp`** (new) — drives the same architecture
`main.cpp` uses (a Poisson-arrival producer pushing onto the real
`aether::api::IngressQueue`, a separate engine thread draining it and
calling `scheduler.step()`), deliberately in-process rather than over real
HTTP — see the file's header comment for why: `main.cpp`'s `/generate` is
admission-only (returns immediately), so an HTTP-level timer can only ever
measure admission latency, not TTFT/inter-token latency, which is the
whole reason Step 1's telemetry instrumentation exists. `load_generator.cpp`
(Milestone 1) remains the tool for exercising the real HTTP ingress path;
this one is for the actual P50/P95/P99 numbers the spec's Definition of
Done asks for.

- **Poisson arrivals**: inter-arrival times drawn from
  `std::exponential_distribution`, paced against real `steady_clock` time
  (not a simulated/virtual clock) — "multi-hour run" is taken literally,
  since the whole point of a multi-hour run is catching real time-based
  degradation a virtual clock couldn't expose.
- **Mixed short/long lengths**: 85% short (10–200 prompt tokens, 1–32
  generated), 15% long (500–2,000 prompt tokens, 32–256 generated) by
  default, configurable via `--long-fraction`.
- **Reports**: admitted/completed/rejected/in-flight-at-cutoff counts,
  tokens/sec, requests/sec, and TTFT + inter-token P50/P95/P99 straight
  from Step 1's `TelemetryFlusher::snapshot()` — to a CSV
  (`--output PREFIX` → `PREFIX_summary.csv`), one row per `--repeat`.
- **`benchmarks/plot_benchmark.py`** (new) — reads that CSV and produces
  three PNG charts (latency percentiles across runs, throughput across
  runs, backpressure/telemetry-drop accounting) via matplotlib, plus a
  `--consistency` mode that flags any metric whose run-to-run coefficient
  of variation exceeds 25% — the testing plan's explicit "checked for
  run-to-run consistency" ask, made into an actual pass/fail check rather
  than an eyeballed chart.
- **`requirements-m4.txt`** (new) — `pandas`, `matplotlib`, matching the
  Milestone 2 requirements-file convention.
- **`CMakeLists.txt`** — new `bench_cli` executable target.

### Self-verified in this sandbox

Ran `bench_cli` directly (real runs, real numbers) across a range of
rates, and ran `plot_benchmark.py` against the real CSVs it produced —
including visually inspecting the rendered PNGs, not just checking they
exist:

```
--duration 5  --rate 50   (x2 repeats): 262/248 completed, 0 rejected, 0 telemetry drops
                                          TTFT p50 ~555µs, both runs within ~2% of each other
--duration 20 --rate 20              : 415 completed, 0 rejected, 0 telemetry drops
--duration 20 --rate 75              : 1531 completed, 0 rejected, 0 telemetry drops
--duration 45 --rate 150             : 6784 completed, 0 rejected, 170,401 telemetry samples DROPPED
--duration 8  --rate 2000            : 16005 completed, 0 rejected, 819,961 telemetry samples DROPPED
```
`plot_benchmark.py --consistency` on the 2-repeat run: all 8 metrics
(TTFT/inter-token P50/P95/P99, req/s, tok/s) within 0.2–20% coefficient of
variation, well under the 25% threshold — reproducible.

### RESOLVED: telemetry drops were a real bug in bench_cli.cpp, not the ring buffer or the sandbox

Your two real runs on the M1 Mac confirmed the drops reproduce on real
multi-core hardware — which ruled out my single-core-sandbox theory and
sent me back to find the actual cause rather than guess again:

```
run 1 (duration=30, rate=150):  92,732 dropped,  65,536 collected
run 2 (duration=15, rate=2000): 990,860 dropped,  65,536 collected
```

**The smoking gun: collected count was exactly 65,536 — the ring
buffer's fixed capacity — in both runs, despite wildly different total
volumes.** That's only possible if nothing ever drained the ring *during*
the run at all. Went back to `bench_cli.cpp` and found exactly that: the
`TelemetryFlusher` was constructed *after* the run finished, with a single
`drain_once()` call — meaning the ring filled to capacity almost
immediately and silently dropped nearly everything produced after that,
for the entire run. `telemetry.hpp` itself was never the problem (Step 1's
own tests, including a live background flusher, passed cleanly); this
benchmark tool simply forgot to call `flusher.start()`.

**Fix:** moved the `TelemetryFlusher` construction and `.start()` call to
before the run begins, and replaced the post-hoc `drain_once()` with a
proper `.stop()` (which joins the background thread and does one final
drain) after the engine thread stops.

**Re-ran the exact same conditions in this sandbox** (still single-core,
so if anything a harder environment than your Mac) after the fix:

```
duration=30 rate=150:  4548 completed, 0 rejected, 0 telemetry drops (was 92,732)
duration=15 rate=2000: 29996 completed, 0 rejected, 0 telemetry drops (was 990,860)
```

Zero drops in both, even at the higher rate that previously lost >90% of
samples. Fixed.

**Confirmed on real hardware.** Ran on the M1 Mac:

```
duration=30 rate=150:  dropped_telemetry_samples=0  (was 92,732)
duration=15 rate=2000: dropped_telemetry_samples=0  (was 990,860 -- over 1M samples collected this time, zero dropped)
```

Fix holds. **Step 3 is done.**

**The caveat about no real model/GPU compute yet still stands** — the
token-generation rate (and therefore telemetry sample volume) this
benchmark produces is almost certainly far higher than a real GPU-backed
decode loop would sustain, since every decode step here is a near-instant
placeholder push. That's a property of testing before Milestone 2's
kernels are wired in, not something today's fix changes.

### How to run it

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

./build/bench_cli --duration 30 --rate 100 --repeat 3 --output my_run
pip install -r requirements-m4.txt
python3 benchmarks/plot_benchmark.py my_run_summary.csv --consistency
```

For an actual multi-hour run (the literal DoD ask):
```bash
./build/bench_cli --duration 3600 --rate 100 --repeat 1 --output overnight_run
```

`benchmarks/sample_output/` has the real CSV and PNGs generated during
self-verification above (one from a healthy low-rate run, one from the
2000 req/s drop-heavy stress run), if you want to see what the output
actually looks like before running it yourself.

---

## Step 4: End-to-end regression suite

**`tests/test_e2e_regression.cpp`** (new, GTest, **unverified by me as a
GTest binary** — same sandbox constraint as every other step; verified via
the same mechanical plain-assert translation used for Steps 1–3, see
below) — 3 tests that thread a real pipeline through ingress → scheduling
→ simulated multi-process routing → output, tying Milestones 1, 3, and 4
together for real rather than testing each layer in isolation:

1. **`IngressThroughSchedulerAllSequencesAccountedFor`** — 60 sequences
   through the real `IngressQueue` → `ContinuousBatchingScheduler`,
   confirms every one finishes and Step 1's telemetry recorded exactly the
   right sample counts for what actually ran.
2. **`GeneratedTokensRouteAndReassembleCorrectly`** — takes tokens a real
   scheduler run generated, feeds them as real `RoutedToken`s into a real
   Milestone 3 multi-process cluster, confirms the reassembled output
   accounts for every one.
3. **`FullPipelineSurvivesWorkerFailureDuringRouting`** — the strongest
   one: ingress → scheduling → routing, with a worker killed mid-routing
   via Step 2's `chaos.hpp`, using Step 2's `route_batch_tolerant`.
   Confirms the whole pipeline survives, every token is accounted for
   (completed or cleanly failed), and the surviving shards keep serving.

**One scope decision stated explicitly, not glossed over:** this suite
does **not** include Milestone 2's real MLX/Metal gating kernel. Two
independent reasons, both already true before this step started: that
kernel needs real Apple GPU hardware this sandbox doesn't have, AND —
per `README.md`'s own documented Milestone 3 "Known limitations" —
`double_buffer_demo` (M2's kernel work) is **not yet wired into this C++
engine at all**, regardless of hardware. There is no real integration
point to call here yet. Faking one would produce a green checkmark that
catches nothing. The scheduler's decode loop stands in for "the stage
that produces tokens gating would route," clearly commented as such in
the file, exactly the same stand-in `bench_cli.cpp` already uses. When
Milestone 2's kernel eventually gets wired into `main.cpp`, that
integration seam is exactly where this suite should gain a real gating
stage — tracked here explicitly, not silently skipped.

**`.github/workflows/ci.yml`** (new) — runs the full `aether_tests` suite
(all 47 tests across every milestone) plus benchmark smoke tests (not
performance gates — just "does it run without crashing") on GitHub's
`macos-14` Apple Silicon runners on every push/PR, plus a separate
ThreadSanitizer job using the project's own existing `-DAETHER_SANITIZE=thread`
CMake option. Same scope caveat as the kernel: **Milestone 2's MLX/Metal
work is not in this CI path**, both because hosted macOS runners don't
reliably expose real Metal GPU compute for CI and because — again — it
isn't wired into the buildable project yet regardless.

**`CMakeLists.txt`** — `test_e2e_regression.cpp` added to `aether_tests`.

### Self-verified in this sandbox

Used the same discipline as every prior step: mechanically translated the
actual GTest file's `TEST_F`/`EXPECT_*`/`ASSERT_*` calls into a
plain-`assert` harness (via a small Python script doing the textual
substitution, so the test *bodies* being run are byte-for-byte the same
logic as what's in the real file, not a hand-rewritten approximation of
it) and ran that:

- Clean under `-fsanitize=address,undefined`.
- Clean under `-fsanitize=thread`, 3 repeated runs.
- Clean under plain `-O2`, 5 repeated runs (the third test has real
  timing — a scheduled kill during routing — so repeatability mattered
  more here than for a purely deterministic test).
- All 3 scenarios passed every single time: full ingress→scheduler
  accounting for 60/60 sequences with exact telemetry sample counts;
  scheduler-generated tokens routing and reassembling correctly through a
  real 4-process cluster; and the full pipeline surviving a worker kill
  mid-routing with every token accounted for, the killed shard correctly
  isolated, and the three surviving shards still serving.

**`.github/workflows/ci.yml` itself is unverified** — this sandbox has no
network access to actually trigger a GitHub Actions run. Every individual
command in it is one already confirmed working on your real Mac
(`cmake -B build`, `cmake --build build`, `./build/aether_tests`, the
`-DAETHER_SANITIZE=thread` option), but the YAML itself, and
runner-specific behavior on `macos-14`, has not been exercised.

### What to run and report back

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
./build/aether_tests   # now 47 tests: the 44 from Steps 1-2 plus 3 new EndToEndRegression tests
```

If you push this to an actual GitHub repo, please also check that the
Actions tab shows a run and paste back whether it passed — that's the one
piece of this step I genuinely could not verify myself.

---

## Milestone 4 status: all 4 steps built and self-verified; awaiting your confirmation on Step 4

Steps 1–3 are confirmed on your real M1 Mac with real numbers, including
one real bug found and fixed from your data (the telemetry-flusher
lifecycle bug in `bench_cli.cpp`, Step 3). Step 4 is built and
self-verified in this sandbox the same way Steps 1–3 were before your
confirmation; nothing about it has been contradicted by real-hardware
numbers yet since it hasn't been run there.





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